#!/usr/bin/env python3
"""Queue early FB accuracy comparisons or an explicit remap regression control."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess

from arc_job_support import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'binary', 'source', 'gate'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--comparison', choices=('baseline', 'remap'), default='baseline')
    args = parser.parse_args()
    host = 'eb2-3224-lin01.csc.ncsu.edu'
    if os.uname().nodename != host:
        raise RuntimeError('Queue requires the explicitly authorized workstation')
    root = args.root.resolve()
    tag = args.commit[:7]
    label = 'fb_baseline_comparison' if args.comparison == 'baseline' else 'remap_confirmation'
    ledger = root/(label+'_'+tag+'.json')
    if ledger.exists():
        raise RuntimeError('Refusing to overwrite an existing queue')
    state = dict(status='starting', commit=args.commit, gpu=args.gpu,
                 epochs=3, eval_queries=1000, q=4, batch_size=50000,
                 comparison=args.comparison, paper_ready=False, cases={}, removed_optimizer_states=[])
    child = None

    def save(**changes):
        state.update(changes, updated=datetime.datetime.now().isoformat())
        write_json(ledger, state)

    def stop(signum, frame):
        # Leave Popen.wait before waiting again; its waitpid lock is not reentrant.
        raise KeyboardInterrupt('Supervisor signal '+str(signum))

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    runner = Path(__file__).with_name('run_workstation_fb_accuracy_control.py')
    prefix = root/'ge2-a6000-cuda121'
    common = [str(prefix/'bin/python'), '-B', '-u', str(runner),
              '--binary', str(args.binary), '--source', str(args.source),
              '--prefix', str(prefix), '--gate', str(args.gate),
              '--data16', str(root/'data/full_fb_p16'), '--data32', str(root/'data/full_fb_p32'),
              '--tools', str(root/'tools'), '--template', str(root/'templates/config.yaml'),
              '--flags', str(root/'templates/flags.json'), '--workstation-host', host,
              '--commit', args.commit, '--gpu', str(args.gpu), '--visible', '4',
              '--epochs', '3', '--eval-queries', '1000']
    if args.comparison == 'baseline':
        # The matched three-epoch released p16 DistMult reference is already saved.
        reference = root/'evidence/full_fb_distmult_p16_released_3ep_832639f/progress.json'
        row = json.loads(reference.read_text())
        if (row.get('status') != 'done' or row.get('engine') != 'zenodo'
                or row.get('partitions') != 16 or row.get('decoder') != 'distmult'
                or row.get('visible_frames') != 4 or row.get('epochs_completed') != 3
                or row.get('requested_eval_queries') != 1000
                or row.get('evaluation_panel', {}).get('query_sha256') !=
                   'feb300c4745d9f8c4d47781d4d3e8d20c139dff958624b5d7e5d03c360eedae1'):
            raise RuntimeError('Existing GE2 p16 DistMult reference does not match this early comparison')
        state['reference'] = dict(path=str(reference), mrr=row['mrr'], hits_at_10=row['hits_at_10'])
        cases = [('pipege', 'distmult', 32, 'on', 'manual', 'bounded'),
                 ('ge2', 'complex', 16, 'off', 'autograd', 'legacy-random'),
                 ('pipege', 'complex', 32, 'on', 'manual', 'bounded')]
    else:
        cases = [('pipege', 'distmult', p, 'off', 'manual', 'legacy-random') for p in (16, 32)]
    save()
    for index, (system, decoder, partitions, pipeline, gradients, schedule) in enumerate(cases):
        name = 'fb_'+decoder+'_'+system+'_p'+str(partitions)+'_'+args.comparison+'_3ep_'+tag
        work = Path('/dev/shm')/name
        evidence = root/'evidence'/name
        if work.exists() or evidence.exists():
            raise RuntimeError('Refusing to overwrite an existing control: '+name)
        command = common+['--partitions', str(partitions), '--work', str(work), '--evidence', str(evidence),
                          '--decoder', decoder, '--pipeline', pipeline, '--gradients', gradients,
                          '--schedule', schedule, '--engine', 'zenodo' if system == 'ge2' else 'optimized']
        save(status='running', active_case=name, command=command)
        with (root/(name+'.supervisor.log')).open('x') as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            save(child_pid=child.pid)
            try:
                code = child.wait()
            except BaseException:
                if child.poll() is None:
                    child.send_signal(signal.SIGTERM)
                    child.wait(timeout=120)
                save(status='interrupted', child_pid=None)
                raise
        child = None
        progress = json.loads((evidence/'progress.json').read_text()) if (evidence/'progress.json').exists() else {}
        state['cases'][name] = dict(exit_code=code, work=str(work), evidence=str(evidence),
                                   mrr=progress.get('mrr'), hits_at_10=progress.get('hits_at_10'))
        if code != 0 or progress.get('status') != 'done':
            save(status='failed', active_case=name, child_pid=None)
            raise SystemExit(code or 1)
        if index < len(cases)-1:
            # Only this newly completed diagnostic loses resumability; weights remain.
            optimizer = work/'model/embeddings_state.bin'
            expected_bytes = 86054151*100*4
            if not (work/'exact_eval.json').is_file() or optimizer.stat().st_size != expected_bytes:
                raise RuntimeError('Completed-checkpoint cleanup guard failed')
            optimizer.unlink()
            state['removed_optimizer_states'].append(str(optimizer))
        save(child_pid=None)
    save(status='complete', active_case=None)


if __name__ == '__main__':
    main()
