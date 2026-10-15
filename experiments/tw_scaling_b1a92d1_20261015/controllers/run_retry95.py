#!/usr/bin/env python3
"""One isolated two-GPU replacement; original attempt and one-GPU evidence stay intact."""
import copy
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import traceback
import yaml

ROOT = Path(__file__).resolve().parent
RETRY = ROOT / 'retry95'
spec = importlib.util.spec_from_file_location('frozen_validation', ROOT / 'run_validation.py')
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)

def main():
    lock = (ROOT / 'validation.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM, v.interrupted)
    signal.signal(signal.SIGINT, v.interrupted)
    original = v.read(ROOT / 'execution_plan.json')
    v.require(v.sha(ROOT / 'run_validation.py') == original['runner_sha256'], 'Frozen runner changed')
    v.build_gate()
    v.require(v.data_gate() == original['data_gate'], 'Data gate changed')
    plan_hash = v.sha(ROOT / 'execution_plan.json')
    gate = v.read(ROOT / 'fixture_review.json')
    v.require(gate['status'] == 'passed' and gate['plan_sha256'] == plan_hash, 'Fixture gate mismatch')
    v.require(v.sha(ROOT / 'cpu_gate.json') == gate['cpu_gate_sha256'], 'CPU gate changed')
    for path, digest in gate['result_hashes'].items():
        v.require(v.sha(path) == digest, 'Fixture result changed')
    transfer = v.read(ROOT / 'transfer_gate.json')
    v.require(transfer['status'] == 'passed', 'Transfer gate failed')
    for key in ['build_receipt_sha256', 'source_manifest_sha256', 'patch_manifest_sha256']:
        v.require(transfer[key] == original[key], 'Transfer build binding changed')
    one_path = ROOT / 'cases/1gpu/result.json'
    one = v.read(one_path)
    v.require(one['status'] == 'completed' and all(one['checks'].values()), 'One-GPU case incomplete')
    v.require(one['plan_sha256'] == plan_hash and one['train_log_sha256'] == v.sha(one_path.parent / 'train.log'), 'One-GPU receipt mismatch')
    failed_path = ROOT / 'cases/2gpu/result.json'
    failed = v.read(failed_path)
    v.require(failed['status'] == 'failed' and failed['controller_error'] == "RuntimeError('Anon+shmem memory limit reached')", 'Unexpected failed attempt')
    v.require(failed['train_log_sha256'] == v.sha(failed_path.parent / 'train.log'), 'Failed-attempt log changed')
    v.require(not RETRY.exists(), 'Retry directory exists; inspect it, never launch a duplicate')
    v.require(shutil.disk_usage(ROOT).free >= 40 * 1024**3, 'Need40GiB free for temporary model store')
    v.H.require_idle(v.UUIDS)
    hard_limit = int(Path('/sys/fs/cgroup/memory.max').read_text())
    v.require(hard_limit >= 99900000000, 'Cgroup hard limit changed')
    RETRY.mkdir()
    case = RETRY / 'case'
    case.mkdir()
    descriptor = copy.deepcopy(original['cases'][3])
    v.require(v.sha(descriptor['config_source']) == descriptor['config_sha256'] and v.sha(descriptor['flags_source']) == descriptor['flags_sha256'], 'Original case inputs changed')
    config = yaml.safe_load(Path(descriptor['config_source']).read_text())
    config['storage']['model_dir'] = str(case / 'model') + '/'
    config['evaluation']['checkpoint_dir'] = str(case / 'model') + '/'
    (case / 'config.yaml').write_text(yaml.safe_dump(config))
    shutil.copyfile(descriptor['flags_source'], case / 'flags.json')
    descriptor.update(name='2gpu_retry95', directory=str(case), config_source=str(case / 'config.yaml'), flags_source=str(case / 'flags.json'), config_sha256=v.sha(case / 'config.yaml'), flags_sha256=v.sha(case / 'flags.json'))
    plan = copy.deepcopy(original)
    plan.update(run_root=str(RETRY), cases=[descriptor], memory_anon_plus_shmem_limit_bytes=95000000000, runner_sha256=v.sha(__file__), parent_plan_path=str(ROOT / 'execution_plan.json'), parent_plan_sha256=plan_hash, frozen_runner_sha256=original['runner_sha256'], reused_one_gpu_result=str(one_path), reused_one_gpu_result_sha256=v.sha(one_path), failed_attempt_result=str(failed_path), failed_attempt_result_sha256=v.sha(failed_path), cgroup_memory_max_bytes=hard_limit, amendment='Controller-only reset headroom:90GB->95GB anon+shmem guard; full memory.stat logged each poll. Same binary/source/data/flags/thread count/50k per GPU/two epochs, fresh temporary model directory. No training algorithm change.')
    v.write(RETRY / 'execution_plan.json', plan)
    shutil.copyfile(ROOT / 'status.json', RETRY / 'original_status.json')
    v.ROOT = RETRY
    v.MEMORY_LIMIT = plan['memory_anon_plus_shmem_limit_bytes']
    state = dict(status='preflight', stage='two_gpu_replacement', controller_pid=os.getpid(), started_utc=v.now(), active=None, completed=[], reused_one_gpu_result=str(one_path), retry_plan_sha256=v.sha(RETRY / 'execution_plan.json'))
    original_write = v.write
    def mirrored_write(path, value):
        original_write(path, value)
        if Path(path) == RETRY / 'status.json':
            original_write(ROOT / 'status.json', dict(value, authoritative_status=str(path)))
    v.write = mirrored_write
    base_memory = v.memory
    initial_events = base_memory()['events']
    with (RETRY / 'memory_detail.jsonl').open('x', buffering=1) as detailed:
        def sampled_memory():
            mem = base_memory()
            stat = dict((k, int(val)) for k, val in (line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines()))
            detailed.write(json.dumps(dict(captured_utc=v.now(), memory=mem, full_memory_stat=stat)) + '\n')
            v.require(mem['events'].get('oom', 0) == initial_events.get('oom', 0), 'Cgroup OOM event occurred')
            return mem
        v.memory = sampled_memory
        try:
            v.write(RETRY / 'status.json', state)
            two = v.run_case(plan, descriptor, state)
            one_times = one['timing']['epoch_seconds']
            two_times = two['timing']['epoch_seconds']
            review = dict(status='completed', completed_utc=v.now(), parent_commit=v.COMMIT, source_changed=True, patch_manifest_sha256=original['patch_manifest_sha256'], cpu_threads=8, original_plan_sha256=plan_hash, retry_plan_sha256=v.sha(RETRY / 'execution_plan.json'), result_paths={'1gpu':str(one_path),'2gpu':str(case / 'result.json')}, result_sha256={'1gpu':v.sha(one_path),'2gpu':v.sha(case / 'result.json')}, epoch_times_seconds={'1gpu':one_times,'2gpu':two_times}, epoch2_speedup=one_times[1]/two_times[1], epoch2_parallel_efficiency=one_times[1]/(2*two_times[1]), baseline_epoch_times_seconds={'1gpu':[195.367,193.255],'2gpu':[125.299,119.385]}, one_gpu_epoch2_change_vs_baseline=one_times[1]/193.255-1, two_gpu_epoch2_change_vs_baseline=two_times[1]/119.385-1, timing_eligible=one['timing_eligible'] and two['timing_eligible'], limitations=two['limitations'] + ['TwoGPU required a fresh full two-epoch retry after90GB controller guard stopped the first attempt during reset; replacement used95GB guard with unchanged training implementation.'], accuracy_evaluated=False, checkpoint_saving=False, automatic_followup='none; stop for review before any ten-epoch run')
            v.write(ROOT / 'comparison_review.json', review)
            state.update(status='completed', active=None, completed=[descriptor['name']], updated_utc=v.now())
        except BaseException as exc:
            state.update(status='failed', error=repr(exc), traceback=traceback.format_exc(), updated_utc=v.now())
            raise
        finally:
            v.write(RETRY / 'status.json', state)

if __name__ == '__main__':
    main()
