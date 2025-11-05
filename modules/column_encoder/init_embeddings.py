#!/usr/bin/env python3
"""
init_embeddings.py
─────────────────
Initialize embeddings for all nodes in the dataset using Qwen3-Reranker via vLLM
and save them to pickle files for faster training.
"""

import os
import pickle
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch
from tqdm import tqdm
import numpy as np
import importlib
from vllm import LLM

# Qwen prompt template (match wip_evaluate_qwen_reranker.py)
QWEN_PREFIX = """<|im_start|>system
Judge whether the column (Document) is necessary to use when writing the SQL query, based on the provided Query and Instruct. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
"""
QWEN_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
DEFAULT_QWEN_INSTRUCT = """Given a natural-language database question (Query) and a Column description (Document), decide if the column may be necessary to answer the question."""


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
    p = argparse.ArgumentParser("Initialize embeddings for graph nodes (Qwen-only)")
    p.add_argument("graphs_pkl", type=Path, help="Path to the graphs pickle file to process")
    p.add_argument("--model_path", type=str, required=True, help="Model ID or local path")
    p.add_argument("--sample_ids", nargs="+", type=int, default=None,
                   help="Specific sample IDs to process (default: all samples)")
    p.add_argument("--batch_size", type=int, default=8,
                   help="Batch size for encoding")
    p.add_argument("--max_length", type=int, default=1024,
                   help="Maximum sequence length for the prompt")
    p.add_argument("--output_dir", type=Path,
                   default=Path("frozen_encoder_trainer/embeddings"))
    return p.parse_args()


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


class EmbeddingInitializer:
    def __init__(self, model_path: str, batch_size: int, max_length: int, device: str):
        self.model_path = model_path
        self.batch_size = batch_size
        self.max_length = max_length
        # Normalize device string
        self.device = device if isinstance(device, str) else ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading Qwen3-Reranker with vLLM from {model_path}")
        # Initialize a pooling runner and use task-specific APIs (embed)
        self.vllm = LLM(
            model=model_path,
            runner="pooling",
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=max_length,  # lets vLLM handle truncation internally
            enable_chunked_prefill=False,  # Avoid issues with pooling + encode
            gpu_memory_utilization=0.2,
            tensor_parallel_size=1,
        )

        # Keep these for formatting the exact same prompt text as before
        self.qwen_instruction = DEFAULT_QWEN_INSTRUCT
        self.qwen_prefix_text = QWEN_PREFIX
        self.qwen_suffix_text = QWEN_SUFFIX

        print(f"vLLM ready. Embedding dimension: {self.embed_dim()}")

    def _qwen_format_instruction(self, query: str, doc: str) -> str:
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

    def qwen_last_hidden_states(self, pair_strings: List[str]) -> List[np.ndarray]:
        """
        Returns a pooled hidden state vector per input using vLLM pooling runner.
        Uses the embed API which returns one vector for each input.
        """
        prompts = self._qwen_process_inputs(pair_strings)
        outputs = self.vllm.embed(prompts, use_tqdm=False, truncate_prompt_tokens=self.max_length)
        last_states: List[np.ndarray] = []
        for out in outputs:
            vec = self._vllm_extract_vector(out)
            last_states.append(vec)
        return last_states

    def embed_dim(self) -> int:
        pair_str = self._qwen_format_instruction("dummy query", "dummy doc")
        prompt = self._qwen_process_inputs([pair_str])[0]
        (output,) = self.vllm.embed(prompt, use_tqdm=False)
        vec = self._vllm_extract_vector(output)
        return int(vec.shape[-1])

    def encode_pairs(self, queries: List[str], node_descs_list: List[List[str]]) -> torch.Tensor:
        pairs = [(q, d) for q, descs in zip(queries, node_descs_list) for d in descs]
        outputs: List[torch.Tensor] = []
        for i in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[i: i + self.batch_size]
            pair_strings = [self._qwen_format_instruction(q, d) for q, d in batch_pairs]
            last_vecs = self.qwen_last_hidden_states(pair_strings)
            batch_tensor = torch.stack([torch.from_numpy(vec) for vec in last_vecs])
            outputs.append(batch_tensor)
        return torch.cat(outputs, dim=0) if outputs else torch.empty(0)

    def process_dataset(self, pkl_path: Path, sample_ids: Optional[List[int]] = None) -> Dict[str, Any]:
        print(f"Processing {pkl_path}")
        triples = pickle.load(open(pkl_path, "rb"))
        all_embeddings: Dict[str, Any] = {}

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
            query_key = f"{q}_{sample_id}"
            all_embeddings[query_key] = {
                'query': q,
                'node_names': names,
                'embeddings': embeddings,
                'G': G,
                'positives': positives,
                'sample_id': sample_id
            }
        print(f"Processed {len(all_embeddings)} queries from {pkl_path.name}")
        return all_embeddings


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"

    model_path = args.model_path
    print(f"Initializing embeddings with Qwen reranker, graphs_pkl={args.graphs_pkl}, model_path={model_path}")
    print(f"Output directory: {args.output_dir}")

    initializer = EmbeddingInitializer(
        model_path=model_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device_str,
    )

    pkl_path: Path = args.graphs_pkl
    stem = pkl_path.stem
    print(f"\nProcessing graphs file: {pkl_path}...")
    size_tag = _infer_qwen_size_tag(model_path)
    output_path = args.output_dir / f"{stem}_embeddings_qwen_{size_tag}.pkl"

    embeddings = initializer.process_dataset(pkl_path, args.sample_ids)
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"Saved embeddings to {output_path}")

    # Save metadata
    metadata = {
        'reranker_type': 'qwen',
        'embed_dim': initializer.embed_dim(),
        'batch_size': args.batch_size,
        'max_length': args.max_length,
        'model_path': model_path,
        'graphs_pkl': str(pkl_path),
        'qwen_size': size_tag,
    }
    metadata_path = args.output_dir / f"metadata_{stem}_qwen_{size_tag}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")
    print("\nEmbedding initialization complete!")


if __name__ == "__main__":
    main()
