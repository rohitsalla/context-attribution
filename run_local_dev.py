"""Launch a bounded persistent MLX development run with a configurable model."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=os.environ.get('MLX_MODEL_PATH'))
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--limit', type=int, default=3)
    parser.add_argument('--cap', type=int, default=128)
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--output', default='results/local-dev')
    args = parser.parse_args()
    if not args.model:
        parser.error('supply --model or set MLX_MODEL_PATH to an MLX checkpoint')
    log_dir = root / 'logs'
    log_dir.mkdir(exist_ok=True)
    log, pidfile = log_dir / 'local-dev.log', log_dir / 'local-dev-process.json'
    if pidfile.exists():
        previous = json.loads(pidfile.read_text())
        try:
            os.kill(previous['pid'], 0)
        except ProcessLookupError:
            pass
        else:
            raise SystemExit(f"Prior local development process is still present: {previous['pid']}")
    env = {**os.environ, 'PYTHONPATH': str(root/'src'),
           'HF_HUB_OFFLINE': os.environ.get('HF_HUB_OFFLINE', '1'), 'ATTRIBUTION_PROGRESS': '1'}
    cmd = [args.python, '-u', '-m', 'context_attribution.experiment', '--backend', 'mlx',
           '--model', args.model, '--start', str(args.start), '--limit', str(args.limit),
           '--caps', str(args.cap), '--samples', str(args.samples), '--output', args.output]
    with log.open('ab', buffering=0) as stream:
        process = subprocess.Popen(cmd, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    pidfile.write_text(json.dumps({'pid': process.pid, 'command': cmd, 'log': str(log)}, indent=2))
    print(json.dumps({'pid': process.pid, 'log': str(log)}))


if __name__ == '__main__':
    main()
