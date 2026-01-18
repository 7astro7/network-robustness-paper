from __future__ import annotations

import argparse
from pathlib import Path
import glob

try:
    # Works when invoked as a module: python -m runner.prepare_caida_edges
    from runner.caida_loader import load_caida_as_rel2, export_edge_list, degree_sanity_from_edges
except ModuleNotFoundError:  # pragma: no cover
    # Works when invoked as a script from repo root: python runner/prepare_caida_edges.py
    from caida_loader import load_caida_as_rel2, export_edge_list, degree_sanity_from_edges


def _resolve_input(pattern_or_path: str) -> Path:
    p = Path(pattern_or_path)
    if p.exists():
        return p

    # Treat as glob pattern.
    matches = sorted(glob.glob(pattern_or_path))
    if not matches:
        raise FileNotFoundError(f"No files matched: {pattern_or_path!r}")
    return Path(matches[0])


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Parse CAIDA *.as-rel2.txt and export a unique undirected edge list for experiments."
    )
    ap.add_argument(
        "--input",
        default="*.as-rel2.txt",
        help="Path or glob for CAIDA as-rel2 file (default: *.as-rel2.txt). Use *.bz2 if needed.",
    )
    ap.add_argument(
        "--out",
        default="caida_as_edges.txt",
        help="Output edge list path (default: caida_as_edges.txt).",
    )
    ap.add_argument(
        "--keep-self-loops",
        action="store_true",
        help="Keep self-loops (as1==as2) in the undirected edge set (default: dropped).",
    )
    ap.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Top-K ASNs to display by degree (default: 10).",
    )
    args = ap.parse_args()

    in_path = _resolve_input(args.input)

    res = load_caida_as_rel2(in_path, drop_self_loops=(not args.keep_self_loops))

    # --- quick dataset summary (print once) ---
    print(f"CAIDA input: {in_path}")
    print(f"total lines: {res.n_total_lines:,}")
    print(f"comment lines skipped (#): {res.n_comment_lines:,}")
    print(f"blank lines skipped: {res.n_blank_lines:,}")
    print(f"parse errors: {res.n_parse_errors:,}")
    print("")
    print(f"unique ASNs (nodes): {len(res.nodes):,}")
    print(f"unique undirected edges: {len(res.edges_undirected):,}")
    print(
        "rel counts: "
        + ", ".join(f"{k}={int(res.rel_counts.get(k, 0)):,}" for k in (-1, 0, 1))
        + (
            ""
            if set(res.rel_counts.keys()).issubset({-1, 0, 1})
            else f" (also saw rel values: {sorted(res.rel_counts.keys())})"
        )
    )
    print(f"self-loops seen (as1==as2): {res.n_self_loops:,}")
    print(f"duplicate undirected edges skipped: {res.n_duplicates_undirected:,}")

    # --- export edge list (cacheable) ---
    out_path = export_edge_list(res.edges_undirected, args.out)
    print("")
    print(f"wrote edge list: {out_path} ({len(res.edges_undirected):,} edges)")

    # --- sanity check: degree distribution, top-10 by degree ---
    stats = degree_sanity_from_edges(res.edges_undirected, top_k=args.topk)
    print("")
    print("degree sanity:")
    print(
        "n_nodes={n_nodes:,}, n_edges={n_edges:,}, "
        "min={min_degree}, median={median_degree}, p90={p90_degree}, p99={p99_degree}, max={max_degree}".format(
            **stats
        )
    )
    print(f"top-{args.topk} ASNs by degree:")
    for asn, d in stats["top_by_degree"]:
        print(f"  AS{asn}: {d:,}")

    if stats["n_edges"] <= 0 or stats["n_nodes"] <= 0:
        raise RuntimeError("Sanity check failed: graph is empty.")


if __name__ == "__main__":
    main()


