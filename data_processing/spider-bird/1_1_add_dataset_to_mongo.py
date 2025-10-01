#!/usr/bin/env python3
"""
Bulk-import Bird-062024 or Spider train/dev JSON into MongoDB.

Examples
--------
python import_bird.py train                 # loads data/bird-062024/train/train.json
python import_bird.py dev --base-dir ./data/bird-062024
python import_bird.py train --dataset spider # loads data/sft_data_collections/spider/train/train.json
python import_bird.py dev --dataset spider --base-dir ./data/sft_data_collections/spider
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
from pymongo import MongoClient, UpdateOne


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Import Bird-062024 or Spider split into MongoDB.")
    p.add_argument("split", choices=["train", "dev"],
                   help="Which split to import (train | dev)")
    p.add_argument("--dataset", choices=["bird", "spider"], default="bird",
                   help="Which dataset to import (bird | spider, default: %(default)s)")
    p.add_argument("--base-dir", type=Path, 
                   help="Root directory that contains <split>/<split>.json "
                        "(default: ../data/bird-062024 for bird, ../data/sft_data_collections/spider for spider)")
    p.add_argument("--mongo-uri",
                   default=os.getenv("MONGODB_URI", "mongodb://192.168.1.108:27017"),
                   help="MongoDB URI (env MONGODB_URI overrides)")
    return p.parse_args()


def get_default_base_dir(dataset: str) -> Path:
    """Get the default base directory for the specified dataset."""
    if dataset == "bird":
        return Path("../data/bird-062024")
    elif dataset == "spider":
        return Path("../data/sft_data_collections/spider")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_json_path(base_dir: Path, dataset: str, split: str) -> Path:
    """Get the correct JSON file path based on dataset and split."""
    if dataset == "bird":
        # Bird dataset: ../data/bird-062024/train/train.json
        return base_dir / split / f"{split}.json"
    elif dataset == "spider":
        # Spider dataset: ../data/sft_data_collections/spider/train.json
        return base_dir / f"{split}.json"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def main() -> None:
    args = parse_args()

    # Set default base directory if not provided
    if args.base_dir is None:
        args.base_dir = get_default_base_dir(args.dataset)

    json_path = get_json_path(args.base_dir, args.dataset, args.split)
    if not json_path.is_file():
        raise FileNotFoundError(f"Cannot find {json_path}")

    client = MongoClient(args.mongo_uri)
    # Use dataset-specific collection names
    collection_name = f"{args.dataset}_{args.split}_samples"
    coll = client["mats"][collection_name]

    with json_path.open(encoding="utf-8") as fp:
        data = json.load(fp)

    ops: list[UpdateOne] = [
        UpdateOne({"_id": idx},
                  {"$setOnInsert": {**sample, "_id": idx}},
                  upsert=True)
        for idx, sample in enumerate(data)
    ]

    if ops:
        res = coll.bulk_write(ops, ordered=False)
        print(f"[{coll.name}] inserted {res.upserted_count} new docs; "
              f"{len(data) - res.upserted_count} already existed.")

    client.close()


if __name__ == "__main__":
    main()
