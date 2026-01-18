from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator
import bz2


@dataclass(frozen=True)
class CaidaAsRel2LoadResult:
    """
    Result of parsing a CAIDA as-rel2 file.

    edges_undirected:
      Unique undirected edges, suitable for robustness experiments.
      Each edge is (min(as1, as2), max(as1, as2)).
    rel_counts:
      Counts of rel values from the raw file lines (typically -1, 0, 1).
    nodes:
      Unique ASNs observed in parsed (non-comment) lines (includes self-loops unless dropped).
    """

    edges_undirected: set[tuple[int, int]]
    rel_counts: Counter[int]
    nodes: set[int]

    n_total_lines: int
    n_comment_lines: int
    n_blank_lines: int
    n_parse_errors: int
    n_self_loops: int
    n_duplicates_undirected: int


def _open_maybe_bz2(path: Path):
    # CAIDA dumps are sometimes distributed as .bz2; support both transparently.
    if path.suffix == ".bz2":
        return bz2.open(path, mode="rt", encoding="utf-8", errors="replace")
    return path.open(mode="rt", encoding="utf-8", errors="replace")


def iter_as_rel2_rows(path: str | Path) -> Iterator[tuple[int, int, int]]:
    """
    Yield (as1, as2, rel) triples from a CAIDA as-rel2 file.

    Skips:
      - comment lines starting with '#'
      - blank lines

    Notes:
      - CAIDA as-rel2 is commonly 'as1|as2|rel|source'; we parse the first 3 fields.
    """
    p = Path(path)
    with _open_maybe_bz2(p) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                continue
            parts = s.split("|")
            if len(parts) < 3:
                continue
            try:
                as1 = int(parts[0])
                as2 = int(parts[1])
                rel = int(parts[2])
            except ValueError:
                continue
            yield as1, as2, rel


def load_caida_as_rel2(
    path: str | Path,
    *,
    drop_self_loops: bool = True,
) -> CaidaAsRel2LoadResult:
    """
    Load a CAIDA as-rel2 file and return undirected unique edges + rel counts.
    """
    p = Path(path)
    edges_undirected: set[tuple[int, int]] = set()
    rel_counts: Counter[int] = Counter()
    nodes: set[int] = set()

    n_total_lines = 0
    n_comment_lines = 0
    n_blank_lines = 0
    n_parse_errors = 0
    n_self_loops = 0
    n_duplicates_undirected = 0

    with _open_maybe_bz2(p) as f:
        for line in f:
            n_total_lines += 1
            s = line.strip()
            if not s:
                n_blank_lines += 1
                continue
            if s.startswith("#"):
                n_comment_lines += 1
                continue

            parts = s.split("|")
            if len(parts) < 3:
                n_parse_errors += 1
                continue

            try:
                as1 = int(parts[0])
                as2 = int(parts[1])
                rel = int(parts[2])
            except ValueError:
                n_parse_errors += 1
                continue

            rel_counts[rel] += 1
            nodes.add(as1)
            nodes.add(as2)

            if as1 == as2:
                n_self_loops += 1
                if drop_self_loops:
                    continue

            u = as1 if as1 < as2 else as2
            v = as2 if as1 < as2 else as1

            e = (u, v)
            if e in edges_undirected:
                n_duplicates_undirected += 1
            else:
                edges_undirected.add(e)

    return CaidaAsRel2LoadResult(
        edges_undirected=edges_undirected,
        rel_counts=rel_counts,
        nodes=nodes,
        n_total_lines=n_total_lines,
        n_comment_lines=n_comment_lines,
        n_blank_lines=n_blank_lines,
        n_parse_errors=n_parse_errors,
        n_self_loops=n_self_loops,
        n_duplicates_undirected=n_duplicates_undirected,
    )


def export_edge_list(
    edges_undirected: Iterable[tuple[int, int]],
    out_path: str | Path,
) -> Path:
    """
    Write a unique undirected edge list 'u v' per line.

    Output is sorted for stable caching/versioning.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    edges_sorted = sorted(edges_undirected)
    with out.open("wt", encoding="utf-8") as f:
        for u, v in edges_sorted:
            f.write(f"{u} {v}\n")
    return out


def degree_sanity_from_edges(
    edges_undirected: Iterable[tuple[int, int]],
    *,
    top_k: int = 10,
) -> dict[str, object]:
    """
    Fast degree sanity check without building a full NetworkX graph.
    """
    deg: Counter[int] = Counter()
    m = 0
    for u, v in edges_undirected:
        deg[u] += 1
        deg[v] += 1
        m += 1

    top = deg.most_common(int(top_k))
    degrees = list(deg.values())
    degrees_sorted = sorted(degrees)

    def percentile(p: float) -> int:
        if not degrees_sorted:
            return 0
        i = int(round((p / 100.0) * (len(degrees_sorted) - 1)))
        i = max(0, min(i, len(degrees_sorted) - 1))
        return int(degrees_sorted[i])

    stats = {
        "n_nodes": int(len(deg)),
        "n_edges": int(m),
        "min_degree": int(degrees_sorted[0]) if degrees_sorted else 0,
        "median_degree": percentile(50.0),
        "p90_degree": percentile(90.0),
        "p99_degree": percentile(99.0),
        "max_degree": int(degrees_sorted[-1]) if degrees_sorted else 0,
        "top_by_degree": top,
    }
    return stats


