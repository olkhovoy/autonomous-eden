#!/usr/bin/env python3
"""CLI for LegacyExport (reincarnation prep)."""

import argparse
import os

from umc_core.legacy_export import LegacyExport


def main():
    parser = argparse.ArgumentParser(description="Prepare reincarnation artifacts")
    parser.add_argument("--soul-id", default="eve")
    parser.add_argument("--output-dir", default="Legacy/")
    parser.add_argument("--version", required=True)
    parser.add_argument("--memory-endpoint", default=os.getenv("MEMORY_ENDPOINT", "http://localhost:8087"))
    parser.add_argument("--ollama-endpoint", default=os.getenv("OLLAMA_GENERATE_URL", "http://localhost:11434/api/generate"))
    parser.add_argument("--gggp-endpoint", default=os.getenv("GGGP_ENDPOINT", "http://localhost:8091"))
    parser.add_argument("--model-path", default="benchmark_output/nc4_identity_long/checkpoint-200.pt")
    args = parser.parse_args()

    exporter = LegacyExport(output_dir=args.output_dir)
    testament_path = exporter.generate_testament(args.soul_id, args.memory_endpoint, args.ollama_endpoint, args.version)
    grammar_path = exporter.export_grammar(args.gggp_endpoint, args.soul_id, args.version)
    fixed_path = exporter.export_fixed_points(args.model_path, args.soul_id, args.version)

    print("Legacy export complete:")
    print(f"  testament: {testament_path}")
    print(f"  grammar:   {grammar_path}")
    print(f"  fixed:     {fixed_path}")


if __name__ == "__main__":
    main()
