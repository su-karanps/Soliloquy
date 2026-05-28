#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

mkdir -p paper/figures
cp results/plots/paper_figures/*.png paper/figures/

echo "Synced paper figures into paper/figures/"
