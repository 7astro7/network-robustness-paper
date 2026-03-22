FROM python:3.13-slim

# ── System deps: TeX Live (full for pgfplots/siunitx/physics/booktabs/lmodern)
RUN apt-get update && apt-get install -y --no-install-recommends \
        texlive-latex-extra \
        texlive-science \
        texlive-fonts-recommended \
        texlive-fonts-extra \
        texlive-plain-generic \
        lmodern \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# ── Python deps (pinned)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Source code and data
COPY . .

# ── Allow non-root user (passed via --user at runtime) to write outputs
RUN chmod -R a+rw /workspace

# ── Outputs land in /workspace/paper/{tables,data,figures} and paper.pdf
CMD ["bash", "remake.sh"]
