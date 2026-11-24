#!/usr/bin/env bash
# One-shot deployment of the current committed source; no recurring monitor.
set -euo pipefail
cd "$(dirname "$0")"
study_host="${1:-tc}"
study_commit="$(git rev-parse --verify HEAD)"
[[ "$study_commit" =~ ^[0-9a-f]{40}$ ]]
study_root="/projects/tealab/rohit/context-attribution-stability-${study_commit:0:12}"
study_archive="$(mktemp "${TMPDIR:-/tmp}/context-stability.XXXXXX")"
trap 'rm -f "$study_archive"' EXIT
git archive "$study_commit" | gzip > "$study_archive"
study_sha="$(shasum -a 256 "$study_archive" | awk '{print $1}')"
study_ssh=(-4 -o BatchMode=yes -o ConnectTimeout=20 -o ControlPath=/tmp/arc-stability-live-%C -o ControlMaster=auto -o ControlPersist=600 -o ServerAliveInterval=10 -o ServerAliveCountMax=2)
ssh "${study_ssh[@]}" "$study_host" "mkdir -p '$study_root'; cat > '$study_root/source.tar.gz'" < "$study_archive"
ssh "${study_ssh[@]}" "$study_host" "set -e
cd '$study_root'
printf '%s  source.tar.gz\n' '$study_sha' | sha256sum -c -
tar xzf source.tar.gz
mkdir -p logs
cp /projects/tealab/rohit/context-attribution-generalization-b11f39e/logs/model_paths.json logs/model_paths.json
nohup /projects/tealab/rohit/qwen35-neuron-probe-20261026/.venv/bin/python -u submit_stability.py --project '$study_root' --paths '$study_root/logs/model_paths.json' --python-bin /projects/tealab/rohit/qwen35-neuron-probe-20261026/.venv/bin/python --recomp-path /projects/tealab/rohit/recomp-nq-46f98c1 --source-commit '$study_commit' > logs/deployment.log 2>&1 < /dev/null &
printf 'Submission process started at %s\n' '$study_root'
"
