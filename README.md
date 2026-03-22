# Network Robustness: Early Warning via Successive KL Divergence

Reproducible experiments and paper for:
> *Structural Early Warning of Connectivity Collapse in Heavy-Tailed Networks*

## One-command reproduction

```bash
bash run_docker.sh
```

Requires Docker. Builds the environment, runs all experiments, and compiles `paper.pdf`.

**Outputs:**
- `paper.pdf` — compiled manuscript
- `paper/tables/` — LaTeX table fragments
- `paper/figures/` — PDF/PNG figures
- `paper/data/` — raw CSVs

## CAIDA empirical network data

The CAIDA AS-relationship dataset is required for the empirical network results (Step 9).
It is not bundled in this repository due to size. To include it:

1. Register at [https://www.caida.org/data/request_user_info_forms/as-relationships.xml](https://www.caida.org/data/request_user_info_forms/as-relationships.xml)
2. Download the `20260101.as-rel2.txt.bz2` snapshot
3. Extract and convert to an undirected edge list:

```bash
bunzip2 20260101.as-rel2.txt.bz2
awk '!/^#/ {print $1, $2}' 20260101.as-rel2.txt > caida_as_edges.txt
```

4. Place `caida_as_edges.txt` in the repo root before running `bash run_docker.sh`.

If `caida_as_edges.txt` is absent, the pipeline will skip Step 9 and produce `paper.pdf` without the CAIDA figure (all synthetic results remain intact).

## Running without Docker

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash remake.sh
```

LaTeX (`pdflatex`, `bibtex`) must be installed separately.

## Repository structure

```
core/           — graph models, failure models, metrics, experiment runner
runner/         — sweep orchestration, export, figure generation
experiments/    — control experiments (null control, low-damage control)
scripts/        — plotting scripts
paper/          — generated artifacts (tables, figures, data)
paper.tex       — manuscript source
references.bib  — bibliography
remake.sh       — full pipeline script
run_docker.sh   — one-command Docker entry point
```

## License

Apache 2.0 — see `paper.tex` for full license notice.
