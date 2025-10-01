#!/usr/bin/env python3
"""
init_embeddings.py
─────────────────
Initialize embeddings for all nodes in the dataset using MiniCPM model
and save them to pickle files for faster training.
"""

import os
import pickle
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Sequence, Tuple, cast
import torch
from torch.cuda.amp import autocast
# Fix transformers import issues
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM, AutoModel
from tqdm import tqdm
import networkx as nx
import sys
# NEW: needed for vLLM
import numpy as np

# Add FlagEmbedding import for BGEM3FlagModel
try:
    from FlagEmbedding import BGEM3FlagModel
except (ImportError, NameError):
    BGEM3FlagModel = None
    print("Warning: FlagEmbedding import failed. Standard reranker type may not work.")

# Remove hardcoded paths
# Constants for default model paths
DEFAULT_MINCPM_PATH_BIRD = "/home/datht/graph-schema/embedder/finetuned-reranker-v2-minicpm-layerwise-bird-lora/merged_model/"
DEFAULT_MINCPM_PATH_SPIDER = ""  # TODO: set if available
DEFAULT_ENCODER_PATH_BIRD = "/home/datht/graph-schema/embedder/finetuned-bge-m3-bird/checkpoint-18736"
DEFAULT_ENCODER_PATH_SPIDER = "/home/datht/graph-schema/embedder/output/finetuned-reranker-v2-m3-spider-v2-all-cols/checkpoint-131"
# Default Qwen reranker (HF hub id); you can override with --model_path
DEFAULT_QWEN_RERANKER = "Qwen/Qwen3-Reranker-0.6B"

# Qwen prompt template (match wip_evaluate_qwen_reranker.py)
QWEN_PREFIX = """<|im_start|>system
Judge whether the column (Document) is necessary to use when writing the SQL query, based on the provided Query and Instruct. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
"""
QWEN_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
DEFAULT_QWEN_INSTRUCT = """Given a natural-language database question (Query) and a Column description (Document), decide if the column may be necessary to answer the question."""

DATASET_PKL = {
    "bird": {
        "train": Path("data/bird_train_samples_graph.pkl"),
        "dev": Path("data/bird_dev_samples_graph.pkl"),
    },
    "spider": {
        "train": Path("data/spider_train_samples_graph.pkl"),
        "dev": Path("data/spider_dev_samples_graph.pkl"),
    },
}

# Spider 2.0 has only dev split with both evidence versions
SPIDER2_PKL = {
    "dev": {
        "with_evidence": Path("data/spider2_dev_samples_graph_with_evidence.pkl"),
        "no_evidence": Path("data/spider2_dev_samples_graph_no_evidence.pkl"),
    },
}

def _infer_qwen_size_tag(model_path: str) -> str:
    mp = str(model_path)
    if "0.6B" in mp or "0_6B" in mp:
        return "0.6B"
    if "4B" in mp:
        return "4B"
    if "8B" in mp:
        return "8B"
    return "unknown"

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Initialize embeddings for graph nodes")
    p.add_argument("--reranker_type", choices=["standard", "layerwise", "qwen"], default="layerwise",
                   help="Which reranker to use: standard (BGEM3FlagModel), layerwise (MiniCPM), or qwen (Qwen3-Reranker)")
    p.add_argument("--model_path", type=str, default=None,
                   help="Path to model (default: based on reranker_type and dataset)")
    p.add_argument("--dataset", choices=["bird", "spider", "spider2"], default="bird")
    p.add_argument("--split", choices=["train", "dev", "both"], default="both")
    p.add_argument("--evidence_version", choices=["with_evidence", "no_evidence"], default="with_evidence",
                   help="For Spider 2.0: which evidence version to use")
    p.add_argument("--sample_ids", nargs="+", type=int, default=None,
                   help="Specific sample IDs to process (default: all samples)")
    p.add_argument("--suffix", type=str, default="",
                   help="Suffix to add to output filenames")
    p.add_argument("--cut_layer", type=int, default=39,
                   help="Which hidden layer CLS to use (-1 => last, only for layerwise)")
    p.add_argument("--batch_size", type=int, default=8,
                   help="Batch size for encoding (use smaller values for large models)")
    p.add_argument("--max_length", type=int, default=1024,
                   help="Maximum sequence length (for layerwise/qwen)")
    p.add_argument("--output_dir", type=Path,
                   default=Path("frozen_encoder_trainer/embeddings"))
    return p.parse_args()

def embed_dim() -> int:
    """Resolve embedding size from MiniCPM config."""
    if not hasattr(embed_dim, "value"):
        with open(Path(DEFAULT_MINCPM_PATH_BIRD) / "config.json", "r") as f:
            cfg_hidden = json.load(f)
        embed_dim.value = cfg_hidden.get("hidden_size", 4096)
    return embed_dim.value

def make_desc(node: Dict[str, Any]) -> str:
    """Create description string for a node."""
    col = node.get("node_name", "")
    # meaning = concat(table_meaning, column_meaning)
    table_meaning = node.get("table_meaning", "")
    column_meaning = node.get("meaning", "")
    meaning = f"Table meaning: {table_meaning} ; Column meaning: {column_meaning}"
    col_type = node.get("type", "")
    vals = " , ".join(list(map(str, node.get("similar_values", [])))[:2])
    has_null = node.get("has_null", False)
    val_desc = node.get("value_desc", "")
    # Split by dots: first element is table, rest is column
    elms = col.split(".")
    table = elms[0]
    column = ".".join(elms[1:])  # Join the rest back together
    parts = [f"{table}.{column}", meaning,
             f"type {col_type}", f"has values {vals}",
             f"has_null = {has_null}"]
    if val_desc.strip():
        parts.append(f"Value description: {val_desc.strip()}")
    return " ; ".join(parts)

def get_inputs(pairs: Sequence[Tuple[str, str]], tokenizer, 
               prompt: str | None = None, max_length: int = 1024):
    """Tokenize query/passage pairs following MiniCPM-Layerwise format."""
    if prompt is None:
        prompt = ("Given a query A and a passage B, determine whether the passage "
                  "contains an answer to the query by providing a prediction of either "
                  "'Yes' or 'No'.")
    sep = "\n"
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    sep_ids = tokenizer(sep, add_special_tokens=False).input_ids
    
    inputs = []
    for query, passage in pairs:
        q_ids = tokenizer(f"A: {query}", add_special_tokens=False,
                          truncation=True, max_length=max_length).input_ids
        p_ids = tokenizer(f"B: {passage}", add_special_tokens=False,
                          truncation=True, max_length=max_length).input_ids
        item = tokenizer.prepare_for_model(
            [tokenizer.bos_token_id] + q_ids,
            sep_ids + p_ids,
            truncation="longest_first",
            max_length=max_length,
            padding=False,
            return_token_type_ids=False,
            add_special_tokens=False
        )
        item["input_ids"] += sep_ids + prompt_ids
        item["attention_mask"] = [1] * len(item["input_ids"])
        inputs.append(item)
    return tokenizer.pad(inputs, padding=True,
                         max_length=max_length + len(sep_ids) + len(prompt_ids),
                         pad_to_multiple_of=8, return_tensors="pt")

# Refactor EmbeddingInitializer to support both reranker types
class EmbeddingInitializer:
    def __init__(self, reranker_type: str, model_path: str, cut_layer: Optional[int], batch_size: int, max_length: int, device: str):
        self.reranker_type = reranker_type
        self.model_path = model_path
        self.cut_layer = cut_layer
        self.batch_size = batch_size
        self.max_length = max_length
        if isinstance(device, torch.device):
            self.device = str(device) if device.type == "cpu" else "cuda:0"
        else:
            self.device = device
        self.torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if reranker_type == "layerwise":
            print(f"Loading MiniCPM model from {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
            ).to(self.torch_device)
            for param in self.model.parameters():
                param.requires_grad = False
            print(f"Model loaded. Embedding dimension: {self.embed_dim()}")
        elif reranker_type == "standard":
            print(f"Loading encoder model from {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path).to(self.torch_device)
            for param in self.model.parameters():
                param.requires_grad = False
            print(f"Model loaded. Embedding dimension: {self.embed_dim()}")
        elif reranker_type == "st":
            print(f"Loading sentence-transformers CrossEncoder from {model_path}")
            from sentence_transformers.cross_encoder import CrossEncoder
            # CrossEncoder expects a device string or None
            self.model = CrossEncoder(model_path, max_length=self.max_length, device=str(self.torch_device))
            # Do not set self.tokenizer here to avoid Optional type confusion
            self.tokenizer = None  # type: ignore[assignment]
            print(f"CrossEncoder loaded. Embedding dimension: {self.embed_dim()}")
        elif reranker_type == "qwen":
            print(f"Loading Qwen3-Reranker with vLLM from {model_path}")
            try:
                from vllm import LLM
            except Exception as e:
                raise RuntimeError(
                    "vLLM is required for the 'qwen' reranker_type. Please install vllm."
                ) from e

            # Initialize a pooling runner and use task-specific APIs (embed)
            gpu_mem_util = float(os.getenv("VLLM_GPU_MEM_UTIL", "0.7"))
            tp_size = int(os.getenv("VLLM_TP_SIZE", "1"))
            self.vllm = LLM(
                model=model_path,
                runner="pooling",
                dtype="bfloat16",
                trust_remote_code=True,
                max_model_len=max_length,  # lets vLLM handle truncation internally
                enable_chunked_prefill=False,  # Avoid issues with pooling + encode
                gpu_memory_utilization=gpu_mem_util,
                tensor_parallel_size=tp_size,
            )

            # Keep these for formatting the exact same prompt text as before
            self.qwen_instruction = DEFAULT_QWEN_INSTRUCT
            # IMPORTANT: Use the *string* prefix/suffix directly with vLLM.
            self.qwen_prefix_text = QWEN_PREFIX
            self.qwen_suffix_text = QWEN_SUFFIX

            # For type consistency with other branches
            self.tokenizer = None  # type: ignore[assignment]
            self.model = None      # vLLM handles the model internally
            print(f"vLLM ready. Embedding dimension: {self.embed_dim()}")
        else:
            raise ValueError(f"Unknown reranker_type: {reranker_type}")

    def _qwen_format_instruction(self, query: str, doc: str) -> str:
        # same as before, but we'll feed this *text* to vLLM
        return f"<Instruct>: {self.qwen_instruction}\n<Query>: {query}\n<Document>: {doc}"

    def _qwen_process_inputs(self, pair_strings: List[str]) -> List[str]:
        # With vLLM, just wrap text with the same prefix/suffix strings
        # (vLLM will tokenize internally).
        return [self.qwen_prefix_text + s + self.qwen_suffix_text for s in pair_strings]

    def _vllm_extract_vector(self, output_obj: Any) -> np.ndarray:
        """Extract embedding vector from vLLM pooling output across versions.
        Tries output.outputs.data, output.outputs.embedding, output.data, output.embedding.
        """
        candidate = None
        inner = getattr(output_obj, "outputs", None)
        if inner is not None:
            candidate = getattr(inner, "data", None)
            if candidate is None:
                candidate = getattr(inner, "embedding", None)
        if candidate is None:
            candidate = getattr(output_obj, "data", None)
        if candidate is None:
            candidate = getattr(output_obj, "embedding", None)
        if candidate is None:
            raise RuntimeError("Failed to extract embedding vector from vLLM output")
        return np.asarray(candidate, dtype=np.float32)

    def qwen_last_hidden_states(
        self, pair_strings: List[str]
    ) -> List[np.ndarray]:
        """
        Returns a pooled hidden state vector per input using vLLM pooling runner.
        Uses the embed API which returns one vector for each input.
        """
        assert hasattr(self, "vllm"), "qwen reranker_type must be initialized with vLLM."
        prompts = self._qwen_process_inputs(pair_strings)

        # Use the pooling runner's embed API to obtain vectors
        outputs = self.vllm.embed(prompts, use_tqdm=False, truncate_prompt_tokens=self.max_length)
        last_states: List[np.ndarray] = []
        for out in outputs:
            vec = self._vllm_extract_vector(out)
            last_states.append(vec)
        return last_states

    def embed_dim(self) -> int:
        if self.reranker_type == "layerwise":
            with open(Path(self.model_path) / "config.json") as f:
                return json.load(f)["hidden_size"]
        elif self.reranker_type == "standard":
            dummy = self.tokenizer([("dummyQ", "dummyD")], padding=True,
                                   truncation=True, max_length=32,
                                   return_tensors='pt').to(self.torch_device)  # type: ignore[union-attr]
            with torch.no_grad():
                out = self.model(**dummy, output_hidden_states=True)
            return out.hidden_states[-1].shape[-1]
        elif self.reranker_type == "st":
            # Get embedding dimension by running a dummy sentence through the underlying transformer
            dummy_pairs = [["dummy query", "dummy description"]]
            with torch.no_grad():
                # Access the underlying model/tokenizer from CrossEncoder
                underlying_model = self.model.model if hasattr(self.model, 'model') else self.model  # type: ignore[assignment]
                ce_tokenizer = getattr(self.model, 'tokenizer', None)
                if ce_tokenizer is None:
                    raise RuntimeError("CrossEncoder tokenizer not available")
                ce_tokenizer = cast(Any, ce_tokenizer)
                inputs = ce_tokenizer(dummy_pairs, padding=True, truncation=True,
                                      max_length=512, return_tensors='pt').to(self.torch_device)
                out = underlying_model(**inputs, output_hidden_states=True)
                return out.hidden_states[-1].shape[-1]
        elif self.reranker_type == "qwen":
            # Use vLLM pooling runner to get an embedded vector and read its dimension
            pair_str = self._qwen_format_instruction("dummy query", "dummy doc")
            prompt = self._qwen_process_inputs([pair_str])[0]
            (output,) = self.vllm.embed(prompt, use_tqdm=False)
            vec = self._vllm_extract_vector(output)
            return int(vec.shape[-1])
        else:
            raise ValueError(f"Unknown reranker_type: {self.reranker_type}")

    def encode_pairs(self, queries: List[str], node_descs_list: List[List[str]]) -> torch.Tensor:
        if self.reranker_type == "layerwise":
            # ───── layerwise branch ───────────────────────────────────────────
            pairs = [(q, d) for q, descs in zip(queries, node_descs_list)
                            for d in descs]

            outputs = []
            for i in range(0, len(pairs), self.batch_size):
                batch_pairs = pairs[i : i + self.batch_size]

                inputs = get_inputs(
                    batch_pairs,
                    self.tokenizer,  # type: ignore[arg-type]
                    max_length=self.max_length
                ).to(self.torch_device)

                # we only need the encoder, not the scoring head
                encoder = getattr(self.model, "model", self.model)  # fallback
                kwargs: Dict[str, Any] = {"output_hidden_states": True, "return_dict": True}
                # add cutoff_layer to kwargs
                if self.cut_layer is not None:
                    kwargs['cutoff_layers'] = [self.cut_layer]
                with torch.no_grad(), autocast():
                    enc_out = encoder(**inputs, **kwargs)

                # pick the hidden_states list and the required layer
                hidden_states = enc_out.hidden_states[0]
                cls_vec = hidden_states[:, -1, :].float()  # for decoder only model, take the last token

                # Move to CPU immediately to free GPU memory
                outputs.append(cls_vec.cpu())
                
                # Clear memory
                del inputs, enc_out, hidden_states, cls_vec
                torch.cuda.empty_cache()
        
            return torch.cat(outputs, dim=0)
        elif self.reranker_type == "standard":
            pairs = []
            for q, descs in zip(queries, node_descs_list):
                pairs.extend([(q, d) for d in descs])
            outputs = []
            for i in range(0, len(pairs), self.batch_size):
                batch_pairs = pairs[i:i + self.batch_size]
                # Tokenize as list of (query, desc) pairs
                batch_texts = [(q, d) for q, d in batch_pairs]
                # HuggingFace tokenizer can take list of tuples for pair encoding
                inputs = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt').to(self.torch_device)  # type: ignore[union-attr]
                with torch.no_grad():
                    out = self.model(**inputs, output_hidden_states=True)
                hidden_states = out.hidden_states
                cls = hidden_states[-1][:, 0, :].to(torch.float32)
                # Move to CPU immediately to free GPU memory
                outputs.append(cls.cpu())
                del inputs, out, cls
                torch.cuda.empty_cache()
            return torch.cat(outputs, dim=0)
        elif self.reranker_type == "st":
            # For CrossEncoder, we need to get embeddings from the underlying model
            pairs = []
            for q, descs in zip(queries, node_descs_list):
                pairs.extend([(q, d) for d in descs])
            
            outputs = []
            for i in range(0, len(pairs), self.batch_size):
                batch_pairs = pairs[i:i + self.batch_size]
                batch_texts = [[q, d] for q, d in batch_pairs]
                with torch.no_grad():
                    underlying_model = self.model.model if hasattr(self.model, 'model') else self.model  # type: ignore[assignment]
                    ce_tokenizer = getattr(self.model, 'tokenizer', None)
                    if ce_tokenizer is None:
                        raise RuntimeError("CrossEncoder tokenizer not available")
                    ce_tokenizer = cast(Any, ce_tokenizer)
                    inputs = ce_tokenizer(batch_texts, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt').to(self.torch_device)
                    out = underlying_model(**inputs, output_hidden_states=True)
                    hidden_states = out.hidden_states
                    cls = hidden_states[-1][:, 0, :].to(torch.float32)
                    outputs.append(cls.cpu())
                del inputs, out, cls
                torch.cuda.empty_cache()
            return torch.cat(outputs, dim=0)
        elif self.reranker_type == "qwen":
            # Use vLLM to get embeddings
            pairs = [(q, d) for q, descs in zip(queries, node_descs_list)
                            for d in descs]
            outputs = []
            for i in range(0, len(pairs), self.batch_size):
                batch_pairs = pairs[i : i + self.batch_size]
                pair_strings = [self._qwen_format_instruction(q, d) for q, d in batch_pairs]
                # Get last hidden states using vLLM
                last_vecs = self.qwen_last_hidden_states(pair_strings)
                # Convert numpy arrays to torch tensors
                batch_tensor = torch.stack([torch.from_numpy(vec) for vec in last_vecs])
                outputs.append(batch_tensor)
            return torch.cat(outputs, dim=0)
        else:
            raise ValueError(f"Unknown reranker_type: {self.reranker_type}")

    def process_dataset(self, pkl_path: Path, sample_ids: Optional[List[int]] = None) -> Dict[str, torch.Tensor]:
        print(f"Processing {pkl_path}")
        triples = pickle.load(open(pkl_path, "rb"))
        all_embeddings = {}
        node_count = 0
        
        # Filter triples by sample IDs if specified
        if sample_ids is not None:
            sample_ids_set = set(sample_ids)
            filtered_triples = []
            for i, (q, G, positives, sample_id, *_) in enumerate(triples):
                if sample_id in sample_ids_set:
                    filtered_triples.append((q, G, positives, sample_id))
            triples = filtered_triples
            print(f"Filtered to {len(triples)} samples from {len(sample_ids)} requested sample IDs")
        
        for q, G, positives, sample_id, *_ in tqdm(triples, desc=f"Processing {pkl_path.name}"):
            if not positives:
                continue
            names = sorted(G.nodes())
            node_descs = [make_desc(G.nodes[n]) for n in names]
            embeddings = self.encode_pairs([q] * len(names), [node_descs])
            query_key = f"{q}_{sample_id}"  # Use sample_id instead of node_count for better tracking
            all_embeddings[query_key] = {
                'query': q,
                'node_names': names,
                'embeddings': embeddings,
                'G': G,
                'positives': positives,
                'sample_id': sample_id
            }
            node_count += 1
        print(f"Processed {len(all_embeddings)} queries from {pkl_path.name}")
        return all_embeddings

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Set model_path default
    if args.model_path is None:
        if args.reranker_type == "layerwise":
            if args.dataset == "bird":
                model_path = DEFAULT_MINCPM_PATH_BIRD
            elif args.dataset == "spider":
                model_path = DEFAULT_MINCPM_PATH_SPIDER
            elif args.dataset == "spider2":
                model_path = "embedder/finetuned-reranker-v2-minicpm-layerwise-bird-lora-merged-group-size-8/merged_model"
            else:
                raise ValueError(f"Unknown dataset: {args.dataset}")
        elif args.reranker_type == "standard":
            if args.dataset == "bird":
                model_path = DEFAULT_ENCODER_PATH_BIRD
            elif args.dataset == "spider":
                model_path = DEFAULT_ENCODER_PATH_SPIDER
            elif args.dataset == "spider2":
                model_path = "embedder/finetuned-reranker-v2-minicpm-layerwise-bird-lora-merged-group-size-8/merged_model"
            else:
                raise ValueError(f"Unknown dataset: {args.dataset}")
        elif args.reranker_type == "qwen":
            # Default to hub model; for local checkpoints, pass --model_path
            model_path = DEFAULT_QWEN_RERANKER
        else:
            raise ValueError(f"Unknown reranker_type: {args.reranker_type}")
    else:
        model_path = args.model_path

    print(f"Initializing embeddings with reranker_type={args.reranker_type}, dataset={args.dataset}, model_path={model_path}")
    print(f"Output directory: {args.output_dir}")

    initializer = EmbeddingInitializer(
        reranker_type=args.reranker_type,
        model_path=model_path,
        cut_layer=args.cut_layer if args.reranker_type == "layerwise" else None,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device_str
    )

    # Handle Spider 2.0 specially since it only has dev split and evidence versions
    if args.dataset == "spider2":
        if args.split == "both" or args.split == "train":
            print("Warning: Spider 2.0 only has dev split. Processing dev split only.")
        splits = ["dev"]
        for split in splits:
            pkl_path = SPIDER2_PKL[split][args.evidence_version]
            print(f"\nProcessing Spider 2.0 {split} set ({args.evidence_version})...")
            # Always use default naming (no suffix)
            output_path = args.output_dir / f"{args.dataset}_{split}_{args.evidence_version}_embeddings_{args.reranker_type}.pkl"
            
            embeddings = initializer.process_dataset(pkl_path, args.sample_ids)
            with open(output_path, 'wb') as f:
                pickle.dump(embeddings, f)
            print(f"Saved {split} embeddings to {output_path}")
    else:
        splits = ["train", "dev"] if args.split == "both" else [args.split]
        for split in splits:
            pkl_path = DATASET_PKL[args.dataset][split]
            print(f"\nProcessing {split} set...")
            # Always use default naming (no suffix)
            if args.reranker_type == "qwen":
                size_tag = _infer_qwen_size_tag(model_path)
                output_path = args.output_dir / f"{args.dataset}_{split}_embeddings_{args.reranker_type}_{size_tag}.pkl"
            else:
                output_path = args.output_dir / f"{args.dataset}_{split}_embeddings_{args.reranker_type}.pkl"
            
            embeddings = initializer.process_dataset(pkl_path, args.sample_ids)
            with open(output_path, 'wb') as f:
                pickle.dump(embeddings, f)
            print(f"Saved {split} embeddings to {output_path}")

    # Save metadata
    metadata = {
        'reranker_type': args.reranker_type,
        'cut_layer': args.cut_layer if args.reranker_type == "layerwise" else None,
        'embed_dim': initializer.embed_dim(),
        'batch_size': args.batch_size,
        'max_length': args.max_length,
        'model_path': model_path,
        'dataset': args.dataset,
        'qwen_size': _infer_qwen_size_tag(model_path) if args.reranker_type == "qwen" else None,
    }
    if args.reranker_type == "qwen":
        size_tag = _infer_qwen_size_tag(model_path)
        metadata_path = args.output_dir / f"metadata_{args.dataset}_{args.reranker_type}_{size_tag}.json"
    else:
        metadata_path = args.output_dir / f"metadata_{args.dataset}_{args.reranker_type}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")
    print("\nEmbedding initialization complete!")

if __name__ == "__main__":
    main()
