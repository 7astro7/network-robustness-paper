from __future__ import annotations

"""
Generate a CAIDA (2026-01-01) random-failure "figure" for the paper *without*
third-party Python deps. This environment doesn't guarantee numpy/matplotlib/networkx,
so we output plot-ready data files and render using LaTeX/pgfplots.
"""

from pathlib import Path
import argparse
import math
import random
from collections import Counter, deque

from runner.export import export_caida_summary


def _linspace(n: int, q_max: float) -> list[float]:
    if n < 2:
        raise ValueError("n must be >= 2")
    q_max = float(q_max)
    if not (0.0 < q_max <= 1.0):
        raise ValueError("q_max must be in (0, 1]")
    step = q_max / (n - 1)
    return [i * step for i in range(n)]


def _ewma(xs: list[float], alpha: float) -> list[float]:
    if not xs:
        return []
    out = [0.0] * len(xs)
    out[0] = float(xs[0])
    for i in range(1, len(xs)):
        out[i] = alpha * xs[i] + (1.0 - alpha) * out[i - 1]
    return out


def _kl_divergence_bits(P: list[float], Q: list[float]) -> float:
    if len(P) != len(Q):
        raise ValueError("KL: length mismatch")
    s = 0.0
    for p, q in zip(P, Q):
        # with epsilon smoothing, both are > 0; keep guard anyway
        if p > 0.0 and q > 0.0:
            s += p * math.log(p / q, 2)
    return float(s)


def _mean_std(xs: list[float]) -> tuple[float, float]:
    if len(xs) < 2:
        return float("nan"), float("nan")
    mu = sum(xs) / len(xs)
    var = sum((x - mu) ** 2 for x in xs) / len(xs)
    return float(mu), float(math.sqrt(var))


def _read_edges_and_nodes(edge_list_path: str) -> tuple[list[tuple[int, int]], list[int], int]:
    edges: list[tuple[int, int]] = []
    nodes_set: set[int] = set()
    deg0: Counter[int] = Counter()

    with open(edge_list_path, "rt", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            a, b = s.split()
            u = int(a)
            v = int(b)
            edges.append((u, v))
            nodes_set.add(u)
            nodes_set.add(v)
            deg0[u] += 1
            deg0[v] += 1

    nodes = sorted(nodes_set)
    k_max0 = max(deg0.values()) if deg0 else 0
    return edges, nodes, int(k_max0)


class _DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [0] * n

    def make_active(self, x: int) -> None:
        self.parent[x] = x
        self.size[x] = 1

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> int:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return self.size[ra]
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return self.size[ra]


def _build_indexed_adjacency(edges: list[tuple[int, int]], nodes: list[int]) -> tuple[list[list[int]], dict[int, int]]:
    idx = {asn: i for i, asn in enumerate(nodes)}
    adj: list[list[int]] = [[] for _ in range(len(nodes))]
    for u, v in edges:
        iu = idx[u]
        iv = idx[v]
        adj[iu].append(iv)
        adj[iv].append(iu)
    return adj, idx


def compute_S_curve_random_failure_dsu(
    *,
    edges: list[tuple[int, int]],
    nodes: list[int],
    seed: int,
    q_max: float,
    num_q: int,
) -> tuple[list[float], list[float]]:
    """
    Efficiently compute S(q) for random node removal using reverse percolation + DSU.

    Returns:
      qs: evenly spaced in [0, q_max]
      S_values: S(q) = |GCC| / |V_remaining|
    """
    n = len(nodes)
    rng = random.Random(int(seed))
    order = nodes[:]  # removal order forward
    rng.shuffle(order)

    # reverse order = addition order
    add_order = list(reversed(order))

    # adjacency on indexed nodes
    adj, idx = _build_indexed_adjacency(edges, nodes)

    active = [False] * n
    dsu = _DSU(n)

    # S_by_remaining[t] where t=number of active nodes (remaining), 0..n
    S_by_remaining = [0.0] * (n + 1)
    max_cc = 0

    t = 0
    for asn in add_order:
        i = idx[asn]
        active[i] = True
        dsu.make_active(i)
        t += 1
        if max_cc < 1:
            max_cc = 1
        for j in adj[i]:
            if active[j]:
                new_size = dsu.union(i, j)
                if new_size > max_cc:
                    max_cc = new_size
        S_by_remaining[t] = max_cc / t

    qs = _linspace(int(num_q), float(q_max))
    S_values: list[float] = []
    for q in qs:
        n_remove = int(q * n)
        remaining = n - n_remove
        if remaining <= 0:
            S_values.append(0.0)
        else:
            S_values.append(float(S_by_remaining[remaining]))

    return qs, S_values


def find_q_collapse(qs: list[float], S_values: list[float], *, threshold: float = 0.1) -> float | None:
    for q, s in zip(qs, S_values):
        if s < threshold:
            return float(q)
    return None


def _ewma_list(xs: list[float], alpha: float) -> list[float]:
    return _ewma(xs, float(alpha))


def detect_q_warn_baseline_rule(
    *,
    qs: list[float],
    dkl_smooth: list[float],
    baseline_q: float = 0.15,
    k: float = 2.0,
) -> float | None:
    """
    Baseline deviation rule on the midpoint grid:
      q_warn = smallest q_mid > baseline_q such that dkl_smooth(q_mid) > mu0 + k*sigma0,
    where mu0, sigma0 are computed over q_mid <= baseline_q.
    """
    qs_mid = [0.5 * (qs[i] + qs[i + 1]) for i in range(len(qs) - 1)]
    baseline_vals = [v for q, v in zip(qs_mid, dkl_smooth) if q <= baseline_q]
    if len(baseline_vals) < 2:
        return None
    mu, sigma = _mean_std(baseline_vals)
    if not (mu == mu and sigma == sigma):
        return None
    threshold = mu + float(k) * sigma
    for q, v in zip(qs_mid, dkl_smooth):
        if q > baseline_q and v > threshold:
            return float(q)
    return None


def compute_dkl_smooth_curve_random_failure_incremental(
    *,
    adj: list[list[int]],
    seed: int,
    qs: list[float],
    alpha: float,
    eps: float = 1e-12,
) -> list[float]:
    """
    Efficiently compute the EWMA-smoothed successive KL signal on the qs midpoint grid
    for a *single random-removal realization* (seed).

    Implementation: start from the full graph and remove nodes in random order, updating
    degrees + the degree-count histogram incrementally.
    """
    n = len(adj)
    rng = random.Random(int(seed))
    order = list(range(n))
    rng.shuffle(order)

    deg = [len(adj[i]) for i in range(n)]
    k_max0 = max(deg) if deg else 0
    counts = [0] * (k_max0 + 1)
    for d in deg:
        counts[d] += 1

    active = [True] * n
    removed = 0

    targets = [int(q * n) for q in qs]

    def pmf_from_counts(remaining: int) -> list[float]:
        if remaining <= 0:
            P0 = [0.0] * (k_max0 + 1)
            P0[0] = 1.0
            return P0
        inv = 1.0 / float(remaining)
        P = [c * inv for c in counts]
        if eps and eps > 0.0:
            P = [p + eps for p in P]
            z = sum(P)
            P = [p / z for p in P]
        return P

    prev_P: list[float] | None = None
    dkl_successive: list[float] = []

    for target_removed in targets:
        while removed < target_removed:
            u = order[removed]
            du = deg[u]
            counts[du] -= 1
            active[u] = False
            for v in adj[u]:
                if active[v]:
                    dv = deg[v]
                    counts[dv] -= 1
                    deg[v] = dv - 1
                    counts[dv - 1] += 1
            deg[u] = 0
            removed += 1

        remaining = n - removed
        P = pmf_from_counts(remaining)
        if prev_P is not None:
            dkl_successive.append(_kl_divergence_bits(P, prev_P))
        prev_P = P

    return _ewma_list(dkl_successive, float(alpha))


def run_caida_random_failure_one_seed(
    *,
    edge_list_path: str,
    outdir: str,
    seed: int,
    alpha: float,
    q_max: float,
    num_q: int,
) -> dict[str, object]:
    """
    One-seed CAIDA run on a configurable grid, reporting:
      - q_warn via baseline deviation rule on smoothed successive KL
      - q_collapse via S(q) < 0.1 (if observed within grid)
      - S(q_max)
    Also writes pgfplots-ready data files for seed 0 (paper figure).
    """
    edges, nodes, k_max0 = _read_edges_and_nodes(edge_list_path)
    adj, idx = _build_indexed_adjacency(edges, nodes)
    n = len(nodes)

    qs = _linspace(int(num_q), float(q_max))

    # --- successive KL signal ---
    dkl_smooth = compute_dkl_smooth_curve_random_failure_incremental(
        adj=adj, seed=seed, qs=qs, alpha=float(alpha), eps=1e-12
    )
    q_warn = detect_q_warn_baseline_rule(qs=qs, dkl_smooth=dkl_smooth, baseline_q=0.15, k=2.0)

    # --- S(q) via reverse-percolation DSU (same seed-defined removal order) ---
    rng = random.Random(int(seed))
    order_nodes = nodes[:]
    rng.shuffle(order_nodes)
    add_order = list(reversed(order_nodes))

    active = [False] * n
    dsu = _DSU(n)
    max_cc = 0
    S_by_remaining = [0.0] * (n + 1)

    t = 0
    for asn in add_order:
        i = idx[asn]
        active[i] = True
        dsu.make_active(i)
        t += 1
        if max_cc < 1:
            max_cc = 1
        for j in adj[i]:
            if active[j]:
                new_size = dsu.union(i, j)
                if new_size > max_cc:
                    max_cc = new_size
        S_by_remaining[t] = max_cc / t

    S_values: list[float] = []
    for q in qs:
        n_remove = int(q * n)
        remaining = n - n_remove
        if remaining <= 0:
            S_values.append(0.0)
        else:
            S_values.append(float(S_by_remaining[remaining]))

    q_collapse = find_q_collapse(qs, S_values, threshold=0.1)
    S_qmax = float(S_values[-1]) if S_values else 0.0

    # --- Write plot data for seed 0 (paper figure) ---
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    s_path = out / "caida_random_S_20260101_seed0.dat"
    dkl_path = out / "caida_random_dkl_20260101_seed0_alpha0.20.dat"
    const_path = out / "caida_random_constants_20260101_seed0.tex"
    if int(seed) == 0:
        with s_path.open("wt", encoding="utf-8") as f:
            f.write("q S\n")
            for q, s in zip(qs, S_values):
                f.write(f"{q:.6f} {s:.12g}\n")

        qs_mid = [0.5 * (qs[i] + qs[i + 1]) for i in range(len(qs) - 1)]
        with dkl_path.open("wt", encoding="utf-8") as f:
            f.write("q_mid dkl_smooth\n")
            for q, v in zip(qs_mid, dkl_smooth):
                f.write(f"{q:.6f} {v:.12g}\n")

        # Write a tiny constants file for pgfplots annotations (no hardcoding in paper.tex).
        def _fmt_num(x: float | None) -> str:
            return f"{float(x):.6f}" if isinstance(x, float) else "nan"

        with const_path.open("wt", encoding="utf-8") as f:
            f.write("% Auto-generated by runner/make_caida_fig.py\n")
            f.write(r"\def\CAIDAQWarn{" + _fmt_num(q_warn) + "}\n")
            f.write(r"\def\CAIDAQCollapse{" + _fmt_num(q_collapse) + "}\n")

    return {
        "edge_list_path": str(edge_list_path),
        "seed": int(seed),
        "alpha": float(alpha),
        "q_max": float(q_max),
        "num_q": int(num_q),
        "n_nodes": int(len(nodes)),
        "n_edges": int(len(edges)),
        "max_degree": int(k_max0),
        "q_warn": q_warn,
        "q_collapse": q_collapse,
        "S_qmax": S_qmax,
        "s_path": str(s_path),
        "dkl_path": str(dkl_path),
        "constants_path": str(const_path),
    }

def _simulate_one_q(
    *,
    edges: list[tuple[int, int]],
    nodes: list[int],
    k_max0: int,
    q: float,
    rng: random.Random,
    eps: float = 1e-12,
) -> tuple[float, list[float]]:
    n0 = len(nodes)
    n_remove = int(q * n0)
    if n_remove >= n0:
        return 0.0, [1.0] + [0.0] * k_max0

    removed = set(rng.sample(nodes, n_remove)) if n_remove > 0 else set()
    remaining = [n for n in nodes if n not in removed]
    remaining_set = set(remaining)

    # Build induced adjacency + degree counts in one pass over edges.
    adj: dict[int, list[int]] = {}
    for u, v in edges:
        if u in remaining_set and v in remaining_set:
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)

    # GCC size (count isolates as components of size 1).
    visited: set[int] = set()
    max_cc = 0
    for n in remaining:
        if n in visited:
            continue
        visited.add(n)
        if n not in adj:
            if max_cc < 1:
                max_cc = 1
            continue

        qd: deque[int] = deque([n])
        size = 0
        while qd:
            x = qd.popleft()
            size += 1
            for y in adj.get(x, ()):
                if y not in visited:
                    visited.add(y)
                    qd.append(y)
        if size > max_cc:
            max_cc = size

    S = max_cc / len(remaining) if remaining else 0.0

    # Degree PMF on fixed support 0..k_max0 (include isolates).
    counts = [0] * (k_max0 + 1)
    isolates = len(remaining) - len(adj)
    if isolates > 0:
        counts[0] += isolates
    for neigh in adj.values():
        d = len(neigh)
        # degrees only decrease under removal, so d <= k_max0 should hold
        if d <= k_max0:
            counts[d] += 1
        else:
            # safety: clamp into last bin
            counts[k_max0] += 1

    total = float(len(remaining)) if remaining else 1.0
    P = [c / total for c in counts]
    # epsilon smoothing + renorm
    if eps and eps > 0.0:
        P = [p + eps for p in P]
        z = sum(P)
        P = [p / z for p in P]

    return float(S), P


def generate_caida_random_failure_data(
    *,
    edge_list_path: str,
    outdir: str,
    seed: int = 0,
    alpha: float = 0.2,
    q_max: float = 0.9,
    num_q: int = 100,
) -> dict[str, object]:
    return run_caida_random_failure_one_seed(
        edge_list_path=edge_list_path,
        outdir=outdir,
        seed=seed,
        alpha=alpha,
        q_max=q_max,
        num_q=num_q,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="CAIDA (2026-01-01) random-failure: figure-data + collapse sweep.")
    ap.add_argument(
        "--edges",
        default="caida_as_edges.txt",
        help="Undirected edge list path (u v per line). Default: caida_as_edges.txt",
    )
    ap.add_argument("--outdir", default="paper/figures")
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0, help="Seed for single-seed data generation.")
    ap.add_argument("--q-max", type=float, default=0.99)
    ap.add_argument("--num-q", type=int, default=200)
    ap.add_argument(
        "--collapse-sweep",
        action="store_true",
        help="Run S(q) collapse check via DSU for 40 seeds (0..39) at higher q resolution.",
    )
    ap.add_argument("--q-max-sweep", type=float, default=0.99)
    ap.add_argument("--num-q-sweep", type=int, default=200)
    ap.add_argument(
        "--run-5-seeds",
        action="store_true",
        help="Run full CAIDA random-failure experiment across multiple seeds and report q_warn/q_collapse/S(q_max).",
    )
    ap.add_argument(
        "--num-seeds",
        type=int,
        default=5,
        help="Number of seeds to use with --run-5-seeds (default: 5).",
    )
    args = ap.parse_args()

    if args.collapse_sweep:
        edges, nodes, _ = _read_edges_and_nodes(args.edges)
        q_max = float(args.q_max_sweep)
        num_q = int(args.num_q_sweep)
        print(f"CAIDA collapse sweep (random failure): q_max={q_max}, num_q={num_q}, seeds=0..39")
        collapses: list[float] = []
        for seed in range(40):
            qs, S_vals = compute_S_curve_random_failure_dsu(
                edges=edges,
                nodes=nodes,
                seed=seed,
                q_max=q_max,
                num_q=num_q,
            )
            qc = find_q_collapse(qs, S_vals, threshold=0.1)
            print(f"  seed {seed}: q_collapse={qc}")
            if qc is not None:
                collapses.append(float(qc))

        if collapses:
            mean = sum(collapses) / len(collapses)
            mn = min(collapses)
            mx = max(collapses)
            print(f"collapsed in {len(collapses)}/40 seeds within q<= {q_max}: min={mn:.4f}, mean={mean:.4f}, max={mx:.4f}")
        else:
            print(f"no collapse observed in 40 seeds within q<= {q_max} (criterion S(q)<0.1)")
        return

    if args.run_5_seeds:
        q_max = float(args.q_max)
        num_q = int(args.num_q)
        alpha = float(args.alpha)
        n_seeds = int(args.num_seeds)
        print(f"CAIDA random-failure (full) {n_seeds}-seed run: q_max={q_max}, num_q={num_q}, alpha={alpha}")
        q_warns: list[float] = []
        q_collapses: list[float] = []
        for seed in range(n_seeds):
            res = run_caida_random_failure_one_seed(
                edge_list_path=args.edges,
                outdir=args.outdir,
                seed=seed,
                alpha=alpha,
                q_max=q_max,
                num_q=num_q,
            )
            qc = res["q_collapse"]
            qc_str = f"{qc:.6f}" if isinstance(qc, float) else f"> {q_max:.3f} (not observed)"
            qw = res["q_warn"]
            qw_str = f"{qw:.6f}" if isinstance(qw, float) else "None"
            print(f"  seed {seed}: q_warn={qw_str}, q_collapse={qc_str}, S(q_max)={res['S_qmax']:.4f}")
            q_warns.append(float(qw) if isinstance(qw, float) else float("nan"))
            q_collapses.append(float(qc) if isinstance(qc, float) else float("nan"))

        export_caida_summary(
            out_path="paper/tables/caida_summary.tex",
            q_warns=q_warns,
            q_collapses=q_collapses,
            n_total=n_seeds,
        )
        print("Wrote: paper/tables/caida_summary.tex")
        return

    res = generate_caida_random_failure_data(
        edge_list_path=args.edges,
        outdir=args.outdir,
        alpha=args.alpha,
        seed=args.seed,
        q_max=float(args.q_max),
        num_q=int(args.num_q),
    )
    print("CAIDA random-failure (data for pgfplots):")
    print(f"  S(q) data: {res['s_path']}")
    print(f"  dKL data:  {res['dkl_path']}")
    print(f"  constants: {res.get('constants_path')}")
    print(f"  q_warn: {res['q_warn']}")
    print(f"  q_collapse: {res['q_collapse']}")
    print(f"  q_max: {res['q_max']}, num_q: {res['num_q']}")
    print(f"  n_nodes: {res['n_nodes']}, n_edges: {res['n_edges']}, max_degree: {res['max_degree']}")
    print(f"  S(q_max): {res['S_qmax']}")


if __name__ == "__main__":
    main()


