"""Bounded monitoring and small, persistent failure evidence for ARC jobs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


class AllocationDeadline(RuntimeError):
    pass


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def save_failure_evidence(source, destination, per_file=1 << 20, total=8 << 20):
    """Copy bounded text tails, never checkpoints or symlink targets."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents:
        raise ValueError('Evidence destination must be outside the source tree')
    if min(per_file, total) < 1:
        raise ValueError('Evidence budgets must be positive')
    destination.mkdir(parents=True, exist_ok=True)
    report = dict(source=str(source), files=[], skipped=[], errors=[])
    candidates = []
    for root, dirs, files in os.walk(source, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not (Path(root)/d).is_symlink()
                         and d not in ('models', 'model', '__pycache__', '.git'))
        for name in files:
            path = Path(root)/name
            if not path.is_symlink() and path.is_file() and path.suffix in (
                    '.log', '.json', '.jsonl', '.yaml', '.yml', '.txt'):
                candidates.append(path)
    # Preserve the actual exception and child status before large telemetry files.
    priority = {'status.json': 0, 'driver.log': 1, 'resolved_config_gate.log': 2,
                'train.log': 3}
    candidates.sort(key=lambda p: (priority.get(p.name, 4), str(p)))
    remaining = total
    for path in candidates:
        rel = path.relative_to(source)
        if remaining <= 0 or len(report['files']) >= 128:
            report['skipped'].append(str(rel))
            continue
        try:
            size = path.stat().st_size
            length = min(size, per_file, remaining)
            with path.open('rb') as stream:
                stream.seek(size-length)
                data = stream.read(length)
            target = destination/rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            remaining -= len(data)
            report['files'].append(dict(path=str(rel), original_bytes=size,
                saved_bytes=len(data), tail_only=size > len(data),
                saved_sha256=hashlib.sha256(data).hexdigest()))
        except OSError as error:
            report['errors'].append(dict(path=str(rel), error=repr(error)))
    report['saved_bytes'] = total-remaining
    write_json(destination/'evidence_manifest.json', report)
    return report


def probe(command, deadline, env, events, attempts=3):
    """Retry transient queries, but never extend the workload deadline."""
    for attempt in range(attempts):
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise AllocationDeadline('Allocation safety deadline reached')
        try:
            return subprocess.check_output(command, env=env, text=True,
                stderr=subprocess.PIPE, timeout=min(20, remaining))
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
            events.append(dict(time=time.time(), command=list(command),
                               attempt=attempt+1, error=repr(error)))
            if time.monotonic() >= deadline:
                raise AllocationDeadline('Allocation safety deadline reached') from error
            if attempt+1 == attempts:
                raise RuntimeError('Monitoring query failed after retries: ' +
                                   repr(command)) from error


def stop_child(child):
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()


def run_logged(command, env, log, timeout, monitor=None):
    log = Path(log)
    deadline = time.monotonic() + timeout
    report = dict(command=list(map(str, command)), status='starting', probe_errors=[])
    child = None
    with log.open('a') as output:
        try:
            if timeout <= 0:
                raise AllocationDeadline('Allocation safety deadline reached before launch')
            child = subprocess.Popen(command, env=env, stdout=output, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
            report.update(status='running', pid=child.pid)
            while child.poll() is None:
                events = []
                try:
                    host = os.uname().nodename.split('.')[0]
                    if env.get('SLURM_JOB_ID'):
                        active = probe(['squeue', '-h', '-j', env['SLURM_JOB_ID'],
                                        '-o', '%T %N'], deadline, env, events).split()
                        if active != ['RUNNING', host]:
                            raise RuntimeError('Allocation is not RUNNING on this node: '+repr(active))
                    if monitor:
                        gpu = probe(['nvidia-smi', '--query-gpu=index,uuid,power.limit,power.draw,clocks.sm,clocks.mem,utilization.gpu,memory.used',
                                     '--format=csv,noheader'], deadline, env, events)
                        apps = probe(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                                      '--format=csv,noheader'], deadline, env, events)
                        jobs = probe(['squeue', '-h', '-t', 'RUNNING,COMPLETING', '-w', host,
                                      '-o', '%A %u %j'], deadline, env, events)
                        others = []
                        for row in apps.splitlines():
                            try:
                                if os.getpgid(int(row.split(',')[1])) != child.pid:
                                    others.append(row)
                            except ProcessLookupError:
                                pass
                        with Path(monitor).open('a') as stream:
                            stream.write(json.dumps(dict(time=time.time(), gpu=gpu, apps=apps,
                                other_processes=others, probe_errors=events,
                                other_jobs=[r for r in jobs.splitlines()
                                            if r.split()[0] != env.get('SLURM_JOB_ID')]))+'\n')
                finally:
                    report['probe_errors'].extend(events)
                    for event in events:
                        output.write('\n[arc_monitor] '+json.dumps(event)+'\n')
                    output.flush()
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise AllocationDeadline('Allocation safety deadline reached')
                try:
                    child.wait(timeout=min(15, remaining))
                except subprocess.TimeoutExpired:
                    continue
            report.update(status='exited', exit_code=child.returncode)
            return child.returncode
        except AllocationDeadline as error:
            report.update(status='allocation_deadline', error=repr(error), exit_code=124)
            return 124
        except BaseException as error:
            report.update(status='failed', error=repr(error))
            raise
        finally:
            if child is not None:
                stop_child(child)
            # A monitor timeout is never mislabeled as a training deadline.
            write_json(log.with_suffix('.run_status.json'), report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    args = parser.parse_args()
    if not args.source.is_dir():
        parser.error('Source directory does not exist')
    report = save_failure_evidence(args.source, args.destination)
    print(json.dumps(dict(saved_files=len(report['files']), saved_bytes=report['saved_bytes'],
                          errors=report['errors'])))
