#!/usr/bin/env bash
# remake.sh — regenerate all paper outputs from scratch.
# Run from the repo root: bash remake.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

CAIDA_EDGES="caida_as_edges.txt"

echo "================================================================"
echo " Step 1/16  Chung–Lu gamma sweep (tables + CSVs + kappa control)"
echo "================================================================"
python main.py

echo ""
echo "================================================================"
echo " Step 2/16  Chung–Lu sensitivity table (alpha x z grid)"
echo "================================================================"
python main.py --sensitivity-only

echo ""
echo "================================================================"
echo " Step 3/16  Configuration-model gamma sweep (tables + CSVs)"
echo "================================================================"
python main.py --graph-model config

echo ""
echo "================================================================"
echo " Step 4/16  Baseline noise comparison table"
echo "================================================================"
python -m runner.make_baseline_noise_table

echo ""
echo "================================================================"
echo " Step 5/16  Configuration-model baseline check (exponent-matched)"
echo "================================================================"
python -m runner.config_model_check

echo ""
echo "================================================================"
echo " Step 6/16  Null control table (false-trigger rate under no damage)"
echo "================================================================"
python experiments/null_control_random_failure.py

echo ""
echo "================================================================"
echo " Step 7/16  Low-damage control table (false-trigger under q<=0.25)"
echo "================================================================"
python experiments/control_low_damage_random_failure.py

echo ""
echo "================================================================"
echo " Step 8/16  Representative figures (Fig 1 random, Fig 2 targeted)"
echo "================================================================"
python runner/run_experiment.py --make-fig1-random
python runner/run_experiment.py --make-fig2-targeted

echo ""
echo "================================================================"
echo " Step 9/16  KL decomposition by degree class (Fig 3)"
echo "================================================================"
python -m runner.make_kl_decomp_fig

echo ""
echo "================================================================"
echo " Step 10/16  Gamma sweep summary figure"
echo "================================================================"
python scripts/plot_gamma_sweep_random.py

echo ""
echo "================================================================"
echo " Step 11/16  Degree-matched CM diagnostic + baseline-window sensitivity"
echo "             + EWS comparator table + D_KL vs removed-degree diagnostic"
echo "================================================================"
python -m runner.degree_matched_config_check
python -m runner.make_degree_matched_table
python -m runner.baseline_window_sensitivity
python -m runner.ews_comparison
python -m runner.make_dkl_vs_removed_degree_fig

echo ""
echo "================================================================"
echo " Step 13/16  Analytical KL signal + Fisher information figures"
echo "================================================================"
python -m runner.analytical_kl
python -m runner.make_fisher_info_fig

echo ""
echo "================================================================"
echo " Step 14/16  KL Lorenz curve (degree-class structure figure)"
echo "================================================================"
python -m runner.make_kl_lorenz_fig

echo ""
echo "================================================================"
echo " Step 15/16  CAIDA figure + summary table"
echo "================================================================"
if [ -f "$CAIDA_EDGES" ]; then
    python -m runner.make_caida_fig --edges "$CAIDA_EDGES"
else
    echo "WARNING: $CAIDA_EDGES not found — skipping CAIDA step."
    echo "See README.md for download instructions."
    echo "All synthetic results are unaffected."
fi

echo ""
echo "================================================================"
echo " Step 16/16  Compile paper.tex -> paper.pdf"
echo "           (pdflatex → bibtex → pdflatex → pdflatex)"
echo "================================================================"
pdflatex -interaction=nonstopmode paper.tex
bibtex paper
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex

echo ""
echo "All done. Outputs:"
echo "  paper/tables/   — LaTeX tables"
echo "  paper/data/     — raw CSVs"
echo "  paper/figures/  — PDF/PNG figures"
echo "  paper.pdf       — compiled manuscript"
