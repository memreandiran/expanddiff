#!/bin/bash
# Fetch the released checkpoints into this directory and verify their hashes.
#
#   bash checkpoints/download.sh            # all seven
#   bash checkpoints/download.sh p b        # just ExpandDiff-P and -B
#
# The checkpoints are licensed CC BY-NC 4.0 (non-commercial use only); see
# LICENSE-WEIGHTS.md. The code in this repository is Apache 2.0.
#
# Override REPO or TAG to pull from a fork or an older release.

set -eu
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-memreandiran/expanddiff}
TAG=${TAG:-checkpoints-v1}
BASE=https://github.com/$REPO/releases/download/$TAG

declare -A FILES=(
  [p]=expanddiff_p_150k.ckpt
  [b]=expanddiff_b_75k.ckpt
  [s]=expanddiff_s_150k.ckpt
  [d]=expanddiff_d_150k.ckpt
  [pu21_unbounded]=ablation_pu21_unbounded_150k.ckpt
  [linear_target]=ablation_linear_target_150k.ckpt
  [linear_unbounded]=ablation_linear_unbounded_150k.ckpt
)

keys=("$@")
[ ${#keys[@]} -eq 0 ] && keys=("${!FILES[@]}")

cd "$HERE"
for k in "${keys[@]}"; do
  f=${FILES[$k]:-}
  [ -n "$f" ] || { echo "unknown arm: $k (choose from: ${!FILES[*]})"; exit 1; }
  if [ -f "$f" ]; then
    echo "have $f"
  else
    echo "fetching $f"
    curl -fL --progress-bar -o "$f" "$BASE/$f"
  fi
done

# Verify whatever is present. Files not downloaded are simply not checked.
if command -v sha256sum >/dev/null 2>&1; then
  echo
  while read -r sum name; do
    [ -f "$name" ] && echo "$sum  $name" | sha256sum -c -
  done < SHA256SUMS.txt
fi
