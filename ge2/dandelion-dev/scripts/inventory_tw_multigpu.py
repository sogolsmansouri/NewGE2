#!/usr/bin/env python3
"""Inventory log evidence by content; filenames alone never establish a run."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess


def inspect(path):
    epochs, broadcasts, hints, failures = [], set(), [], []
    tw = bool(re.search(r'twitter|(?:^|[/_])tw(?:[0-9/_]|$)', str(path), re.I))
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for raw in source:
            digest.update(raw)
            line = raw.decode('utf-8', errors='replace').strip()
            tw |= bool(re.search(r'twitter|41652230|1321528663|1468345182', line, re.I))
            broadcasts.update(map(int, re.findall(r'Broadcasting model to: ([24]) GPUs', line)))
            epochs.extend(int(x)/1000 for x in re.findall(r'Epoch Runtime:\s*(\d+)ms', line))
            if re.search(r'Stateful|Stateflow multi-GPU selected family|SynchronousMultiGPUTrainer|batch_size[:=]|source_version=|commit=|NVIDIA|num_train:', line):
                if line not in hints and len(hints) < 30:
                    hints.append(line)
            if re.search(r'Traceback|CUDA error|out of memory|terminate called|Segmentation fault', line):
                failures.append(line[:500])
    if not tw or not broadcasts:
        return None
    return dict(path=str(path), sha256=digest.hexdigest(), gpus=sorted(broadcasts),
                epochs_observed=len(epochs), epoch_times_s=epochs,
                average_epoch_s=statistics.mean(epochs) if epochs else None,
                steady_epoch_s=statistics.mean(epochs[1:]) if len(epochs)>1 else None,
                hints=hints, failures=failures[:10],
                qualification='historical only; match config, data, hardware and source separately')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', action='append', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    roots = [str(p) for p in args.root if p.exists()]
    command = ['rg', '--files', '--hidden', '-g', '*.log', '-g', '*.out',
               '-g', '!**/envs/**', '-g', '!**/site-packages/**', '-g', '!**/.git/**',
               '-g', '!**/build*/**', '-g', '!**/node_modules/**', *roots]
    listing = subprocess.run(command, text=True, capture_output=True, check=False)
    errors, rows, seen = [], [], set()
    if listing.stderr:
        errors.append(listing.stderr)
    scanned = 0
    for name in listing.stdout.splitlines():
        path = Path(name)
        try:
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            row = inspect(path)
            scanned += 1
            if row:
                rows.append(row)
        except OSError as exc:
            errors.append(dict(path=name, error=str(exc)))
    report = dict(roots=roots, missing_roots=[str(p) for p in args.root if not p.exists()],
                  log_files_scanned=scanned, errors=errors, runs=rows,
                  limitation='No inference of batch size or hardware from timing; inaccessible remote logs excluded.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(log_files_scanned=scanned, multigpu_tw_logs=len(rows), errors=len(errors))))


if __name__ == '__main__':
    main()
