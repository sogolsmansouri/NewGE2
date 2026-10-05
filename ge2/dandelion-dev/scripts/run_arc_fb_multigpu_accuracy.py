#!/usr/bin/env python3
"""Compare FB ComplEx peer/host transports without authorizing final timing."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

from arc_job_support import save_failure_evidence, write_json
from run_arc_multigpu_campaign import idle_devices, run_case
from run_arc_multigpu_control import verify_payload
from run_arc_multigpu_finish import (apply_engine_override, apply_final_cohort,
                                     freeze_retry, source_campaign)
from run_arc_paper_case import digest


CONTROLS = ('peer_4gpu', 'host_4gpu', 'peer_2gpu', 'host_2gpu')


def control_flags(reference, transport):
    if transport not in ('peer', 'host'):
        raise ValueError('Unknown partition transport')
    flags = dict(reference)
    flags.update(GEGE_TRAINING_REPLAY_SEED='17', GEGE_TRAINING_INPUT_AUDIT='1',
                 GEGE_STATEFLOW_PEER_RELAY_FORCE_HOST_FALLBACK='1' if transport == 'host' else '0')
    return flags


def input_trace(path):
    rows = sorted(re.findall(r'\[training-input\] (.*)', path.read_text()))
    if not rows:
        raise ValueError('No replay input evidence')
    return dict(batches=len(rows), sha256=hashlib.sha256('\n'.join(rows).encode()).hexdigest())


def compare_transports(base, states, count):
    peer, host = f'peer_{count}gpu', f'host_{count}gpu'
    if any(states.get(name, {}).get('status') != 'control_complete_not_timing' for name in (peer, host)):
        return None
    spec = f'pipege_fb_complex_{count}gpu'
    left = base/peer/'results'/spec/'control'
    right = base/host/'results'/spec/'control'
    traces = [input_trace(path/'train.log') for path in (left, right)]
    a, b = [json.loads((path/'exact_eval.json').read_text()) for path in (left, right)]
    import numpy as np
    ranks = [np.load(path/'exact_eval.ranks.npz')['tail_ranks'] for path in (left, right)]
    if not np.array_equal(np.load(left/'exact_eval.ranks.npz')['triples'],
                          np.load(right/'exact_eval.ranks.npz')['triples']):
        raise ValueError('Transport comparisons used different queries')
    return dict(gpus=count, input_traces=traces, identical_inputs=traces[0] == traces[1],
                identical_entity_weights=a['entity_bin_sha256'] == b['entity_bin_sha256'],
                identical_forward_relations=a['src_relation_bin_sha256'] == b['src_relation_bin_sha256'],
                identical_inverse_relations=a['dst_relation_bin_sha256'] == b['dst_relation_bin_sha256'],
                identical_tail_ranks=bool(np.array_equal(*ranks)),
                peer_mrr=float(np.mean(1.0/ranks[0])), host_mrr=float(np.mean(1.0/ranks[1])),
                peer_hits10=float(np.mean(ranks[0] <= 10)), host_hits10=float(np.mean(ranks[1] <= 10)),
                interpretation='A transport attribution requires identical audited input traces; '
                               'quality agreement alone is not proof of tensor equivalence')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--payload', type=Path, required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--cases', nargs='+', choices=CONTROLS, default=list(CONTROLS))
    args = parser.parse_args()
    if Path(__file__).resolve().parent != args.payload.resolve():
        raise ValueError('Execute the frozen payload launcher')
    metadata = verify_payload(args.payload)
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGCHLD})

    def stop(sig, frame):
        raise KeyboardInterrupt('Allocation interrupted: '+str(sig))

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    os.environ['SLURM_JOB_ID'] = args.job
    allocation = subprocess.check_output(['scontrol', 'show', 'job', args.job, '-o'], text=True)
    if ('JobState=RUNNING ' not in allocation or 'NodeList=c30 ' not in allocation
            or os.uname().nodename.split('.')[0] != 'c30'
            or 'UserId='+os.environ['USER']+'(' not in allocation):
        raise RuntimeError('Requires owned running c30 allocation')
    deadline = datetime.datetime.fromisoformat(re.search(r'\bEndTime=(\S+)', allocation)[1]).timestamp()-180
    base = args.base.resolve()
    if base.parent != Path('/mnt/local/smansou2'):
        raise ValueError('Dedicated node-local work directory required')
    base.mkdir(exist_ok=False)
    archive = Path('/mnt/beegfs/smansou2')/base.name
    archive.mkdir(exist_ok=False)
    summary = archive/'status'
    summary.mkdir()
    state = dict(job=args.job, launcher_commit=metadata['commit'], control_only=True,
                 paper_ready=False, timing_eligible=False, status='preparing', cases={}, comparisons={},
                 power_w=metadata['power_w'], payload_sha256=digest(args.payload/'payload_manifest.json'),
                 protocol='FB ComplEx only: p32/q4, 50K per GPU, mass1, relabel17, '
                          'replay17, optimized kernels; coordinated host vs real peer transport',
                 evidence_root=str(archive))

    def update(**changes):
        state.update(changes, updated=datetime.datetime.now().isoformat())
        for target in (base, summary):
            write_json(target/'campaign_status.json', state)
        print(json.dumps(dict(status=state['status'], active_case=state.get('active_case'),
                              stage=state.get('stage'), updated=state['updated'])), flush=True)

    try:
        update()
        with Path('/mnt/local/smansou2/paper_multigpu.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            source = source_campaign(metadata)
            for name in args.cases:
                if deadline-time.time() < 3600:
                    update(status='waiting_next_allocation', active_case=None)
                    break
                transport, suffix = name.split('_')
                count = int(suffix[0])
                idle_devices(count)
                case = base/name
                case.mkdir()
                case_summary = summary/name
                case_summary.mkdir()
                spec_name = f'pipege_fb_complex_{count}gpu'
                manifest = freeze_retry(source, case, args.payload, spec_name, metadata['commit'])
                apply_engine_override(case, manifest, metadata)
                apply_final_cohort(manifest, metadata)
                manifest.update(control_only=True, cohort='fb_transport_accuracy_'+args.job,
                                diagnostic_protocol=state['protocol'])
                spec = manifest['cases'][spec_name]
                if (spec['model'], spec['p'], spec['q'], spec['width']) != ('complex', 32, 4, 100):
                    raise ValueError('Unexpected frozen FB model or partition capacity')
                flag_path = Path(spec['flags'])
                flags = control_flags(json.loads(flag_path.read_text()), transport)
                write_json(flag_path, flags)
                manifest['files'][str(flag_path.relative_to(case))] = digest(flag_path)
                spec.update(transport=transport, control_name=name)
                write_json(case/'manifest.json', manifest)
                shutil.copy2(case/'manifest.json', case_summary/'manifest.json')
                sys.path.insert(0, str(case/'harness/tools'))
                try:
                    for phase in ('gate', 'control'):
                        update(status='running', active_case=name, stage=phase)
                        run_case(case, manifest, spec_name, phase, deadline, archive/name,
                                 case_summary, diagnostic_gate=(phase == 'gate'),
                                 physical_devices=list(range(count)))
                    result = case/'results'/spec_name/'control'
                    state['cases'][name] = json.loads((result/'status.json').read_text())
                    state['cases'][name]['input_trace'] = input_trace(result/'train.log')
                    comparison = compare_transports(base, state['cases'], count)
                    if comparison:
                        state['comparisons'][str(count)] = comparison
                except Exception as error:
                    state['cases'][name] = dict(status='failed', error=repr(error))
                    save_failure_evidence(case/'results', case_summary/'failure_evidence', total=16 << 20)
                update()
            else:
                update(status='completed' if all(row['status'] == 'control_complete_not_timing'
                       for row in state['cases'].values()) else 'incomplete_requires_review', active_case=None)
    except BaseException as error:
        update(status='interrupted_or_failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
