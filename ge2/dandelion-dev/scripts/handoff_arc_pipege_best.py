#!/usr/bin/env python3
"""Replace an allocation's serial supervisor only after its active case exits."""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--old-script', type=Path, required=True)
    parser.add_argument('--old-attempt', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('Replacement command required')
    proc = Path('/proc')/str(args.pid)
    if proc.stat().st_uid != os.getuid() or str(args.old_script) not in proc.joinpath('cmdline').read_bytes().decode().split('\0'):
        raise RuntimeError('Predecessor PID/owner/command mismatch')
    job = os.environ['SLURM_JOB_ID']
    allocation = subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
    deadline = datetime.datetime.fromisoformat(re.search(r'\bEndTime=(\S+)',allocation)[1]).timestamp()-240
    state = dict(job=job, predecessor_pid=args.pid, command=command)
    def update(**changes):
        state.update(changes, updated=datetime.datetime.now().isoformat())
        temp = args.report.with_suffix('.tmp')
        temp.write_text(json.dumps(state,indent=2)+'\n')
        temp.replace(args.report)
    paused = False
    def interrupted(sig, frame):
        raise KeyboardInterrupt('Signal '+str(sig))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGUSR1, interrupted)
    try:
        # The child has its own session; stop only the dispatcher, not training
        # or evaluation. This prevents another case starting during the handoff.
        os.kill(args.pid, signal.SIGSTOP)
        paused = True
        time.sleep(.2)
        supervisor = json.loads((args.old_attempt/'supervisor.json').read_text())
        if supervisor['job'] != job or supervisor['status'] != 'running':
            raise RuntimeError('Predecessor is not running in this allocation')
        case = supervisor['case']
        status_path = args.old_attempt/case/'status.json'
        update(status='waiting_for_active_case', case=case)
        while time.time() < deadline:
            if status_path.exists():
                current = json.loads(status_path.read_text())
                child = Path('/proc')/str(current['pid'])/'stat'
                exited = not child.exists() or child.read_text().split(') ',1)[1].startswith('Z ')
                if current['status'] in ('done','failed') and exited:
                    break
            time.sleep(5)
        else:
            raise RuntimeError('Allocation too close to deadline; restoring predecessor')
        os.kill(args.pid, signal.SIGTERM)
        os.kill(args.pid, signal.SIGCONT)
        paused = False
        for _ in range(100):
            stat = proc/'stat'
            if not stat.exists() or stat.read_text().split(') ',1)[1].startswith('Z '):
                break
            time.sleep(.2)
        else:
            raise RuntimeError('Predecessor did not exit; refusing concurrent supervisor')
        update(status='starting_replacement', previous_case_status=current['status'])
        os.execv(command[0], command)
    except BaseException as error:
        if paused:
            os.kill(args.pid, signal.SIGCONT)
        update(status='failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
