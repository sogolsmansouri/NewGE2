"""Stop only the control's process group if a foreign process enters its GPU."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def foreign_gpu_pids(gpu, process_group=None):
    output = subprocess.check_output(
        ['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid', '--format=csv,noheader'],
        text=True, timeout=20)
    foreign = []
    for row in output.splitlines():
        pid = int(row.strip())
        try:
            if process_group is None or os.getpgid(pid) != process_group:
                foreign.append(pid)
        except ProcessLookupError:
            continue
    return foreign


def guarded_run(command, env, log, timeout, gpu, monitor):
    if timeout <= 0:
        raise RuntimeError('Allocation deadline reached')
    foreign = foreign_gpu_pids(gpu)
    if foreign:
        raise RuntimeError('GPU contention before launch: '+repr(foreign))
    with Path(log).open('a') as output:
        child = subprocess.Popen(command, env=env, stdout=output, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        deadline = time.monotonic()+timeout
        try:
            while child.poll() is None:
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise RuntimeError('Allocation deadline reached')
                job = env['SLURM_JOB_ID']
                active = subprocess.check_output(['squeue', '-h', '-j', job, '-o', '%T %N'],
                                                 text=True, timeout=20).strip()
                if not active.startswith('RUNNING ') or os.uname().nodename.split('.')[0] not in active.split()[1:]:
                    raise RuntimeError('Allocation is no longer active: '+active)
                foreign = foreign_gpu_pids(gpu, child.pid)
                with Path(monitor).open('a') as stream:
                    stream.write(json.dumps(dict(time=time.time(), gpu=gpu,
                                                 process_group=child.pid, foreign_gpu_pids=foreign))+'\n')
                if foreign:
                    raise RuntimeError('GPU contention during accuracy control: '+repr(foreign))
                try:
                    return child.wait(timeout=min(5, remaining))
                except subprocess.TimeoutExpired:
                    pass
            return child.returncode
        except BaseException:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            raise
