#!/usr/bin/env bash
# Push kaggle/kernel.py as a private GPU kernel and start it.
# Usage: scripts/push_kaggle_kernel.sh trial|full OWNER/SCAN_DATASET_SLUG
set -euo pipefail

stage=${1:?usage: $0 trial|full OWNER/SCAN_DATASET_SLUG}
dataset=${2:?usage: $0 trial|full OWNER/SCAN_DATASET_SLUG}
owner=${dataset%%/*}
slug="dg-tcav-classifier-${stage}"
kaggle=${KAGGLE_CLI:-kaggle}

build=$(mktemp -d)
trap 'rm -rf "$build"' EXIT
sed "s/^STAGE = \"trial\"$/STAGE = \"${stage}\"/" "$(dirname "$0")/../kaggle/kernel.py" > "$build/kernel.py"
grep -q "^STAGE = \"${stage}\"$" "$build/kernel.py"

cat > "$build/kernel-metadata.json" <<EOF
{
  "id": "${owner}/${slug}",
  "title": "${slug}",
  "code_file": "kernel.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": ["${dataset}"],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
EOF

"$kaggle" kernels push -p "$build"
echo "Status: $kaggle kernels status ${owner}/${slug}"
echo "Output: $kaggle kernels output ${owner}/${slug} -p outputs/kaggle_${stage}"
