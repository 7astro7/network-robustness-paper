#!/bin/sh

# Ensure output directories are writable by the current user.
# This handles the case where a previous root-owned Docker run left files behind.
find paper/ -not -user "$(id -u)" -print 2>/dev/null | grep -q . && \
  echo "Fixing root-owned files in paper/ (may require sudo)..." && \
  sudo chown -R "$(id -u):$(id -g)" paper/

# Build the image (once, or after any code change)
docker build -t network-robustness .

# Run the full pipeline; outputs land in ./paper/ and ./paper.pdf on your host
# :z relabels the volume for SELinux hosts; safe to use on non-SELinux hosts too
# --user ensures files are written as the host user, not root
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e MPLCONFIGDIR=/tmp/matplotlib \
  -v "$(pwd)/paper":/workspace/paper:z \
  -v "$(pwd)":/workspace/out:z \
  network-robustness bash -c "bash remake.sh && cp paper.pdf /workspace/out/paper.pdf"


