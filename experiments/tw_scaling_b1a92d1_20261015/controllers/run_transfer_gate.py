#!/usr/bin/env python3
"""Run only the small native model-transfer tests after the pinned build finishes."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'transfer_gate'
COMMIT = 'b1a92d1131e6dba37b6ce8b994caf72c033e6ad9'
UUIDS = ['GPU-0761238c-da94-7c93-1fce-739e46be8784', 'GPU-d2bca02c-5b52-54c6-3fd9-9ae95284bb02']
LIVE = None


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def command(argv, env=None):
    return subprocess.check_output(argv, env=env, text=True, stderr=subprocess.STDOUT, timeout=30)


def build_gate():
    receipt = json.loads((ROOT / 'build_receipt.json').read_text())
    require(receipt['status'] == 'completed' and receipt['commit'] == COMMIT, 'Expected completed patched b1a92d1 build')
    require(sha(ROOT / 'build_source_manifest.json') == receipt['source_manifest_sha256'], 'Source manifest hash changed')
    require(sha(ROOT / 'patch_manifest.json') == receipt['patch_manifest_sha256'], 'Patch manifest hash changed')
    require(command(['git', '-C', str(ROOT / 'source'), 'rev-parse', 'HEAD']).strip() == COMMIT, 'Source HEAD changed')
    expected_source = json.loads((ROOT / 'build_source_manifest.json').read_text())['files']
    current_source = {str(p.relative_to(ROOT / 'source')): sha(p)
                      for p in sorted((ROOT / 'source').rglob('*'))
                      if p.is_file() and '.git' not in p.relative_to(ROOT / 'source').parts and '__pycache__' not in p.parts}
    require(current_source == expected_source, 'Source differs from completed build')
    for name, artifact in receipt['artifacts'].items():
        require(Path(artifact['path']).resolve() == (ROOT / 'build' / name).resolve(), 'Unexpected build artifact path: ' + name)
        require(sha(artifact['path']) == artifact['sha256'], 'Build artifact changed: ' + name)
    require('gege_model_transfer_test' in receipt['artifacts'] and 'libge2.so' in receipt['artifacts'], 'Missing transfer test or library')
    return receipt


def environment(receipt):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('GEGE_', 'NCCL_', 'TORCH_', 'PYTORCH_', 'CUDA_'))
           and k not in ['LD_PRELOAD', 'LD_LIBRARY_PATH', 'PYTHONPATH', 'PYTHONHOME',
                         'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']}
    env.update(CUDA_VISIBLE_DEVICES=','.join(UUIDS), OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8',
               LD_LIBRARY_PATH=':'.join(receipt['runtime']['library_paths']), GEGE_NO_BINDINGS='1')
    return env


def gpu_snapshot():
    devices = command(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'])
    rows = []
    for line in devices.splitlines():
        values = [item.strip() for item in line.split(',')]
        if values[1] in UUIDS:
            rows.append({'index': int(values[0]), 'uuid': values[1], 'memory_mib': int(values[2]), 'utilization_percent': int(values[3])})
    require({row['uuid'] for row in rows} == set(UUIDS), 'Expected two physical GPU UUIDs unavailable')
    process_text = command(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory', '--format=csv,noheader,nounits'])
    processes = [line.strip() for line in process_text.splitlines() if any(uuid in line for uuid in UUIDS)]
    return {'captured_utc': now(), 'devices': rows, 'processes': processes}


def require_idle():
    # A just-exited CUDA process may need a moment to release its context. Process
    # absence and zero allocations establish idleness; utilization can be stale.
    for _ in range(20):
        snapshot = gpu_snapshot()
        if not snapshot['processes'] and all(row['memory_mib'] == 0 for row in snapshot['devices']):
            return snapshot
        time.sleep(.5)
    raise RuntimeError('GPU context or allocation still present: ' + json.dumps(snapshot))


def stop_owned():
    global LIVE
    if LIVE is not None and LIVE.poll() is None:
        os.killpg(LIVE.pid, signal.SIGTERM)
        try:
            LIVE.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(LIVE.pid, signal.SIGKILL)
            LIVE.wait(timeout=5)


def handle_signal(signum, frame):
    stop_owned()
    raise RuntimeError('Transfer gate interrupted by signal ' + str(signum))


def runtime_snapshot(pid, env, library):
    try:
        maps = (Path('/proc') / str(pid) / 'maps').read_text()
        if str(library) not in maps:
            return None
        actual = dict(part.split('=', 1) for part in (Path('/proc') / str(pid) / 'environ').read_bytes().decode().split('\0') if '=' in part)
    except (FileNotFoundError, ProcessLookupError):
        return None
    for key in ['CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'LD_LIBRARY_PATH']:
        require(actual.get(key) == env[key], 'Runtime environment differs for ' + key)
    libraries = sorted(set(line.split()[-1] for line in maps.splitlines() if 'libge2.so' in line))
    require(libraries == [str(library)], 'Process mapped unexpected libge2: ' + repr(libraries))
    return {'mapped_library': str(library), 'environment_verified': True,
            'selected_environment': {key: actual[key] for key in ['CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'LD_LIBRARY_PATH']}}


def result_checks(mode, text, code):
    checks = {'exit_zero': code == 0}
    if mode == 'host':
        for name in ['distmult_encoder_bias', 'decoder_buffer']:
            checks[name] = f'case={name} exact_broadcast=1 source_unchanged=1 collective_rejections=2' in text
        checks['four_host_payload_checks'] = len(re.findall(r'\[model-transfer-check\].*method=cpu_staging .*exact_match=1', text)) == 4
        checks['two_exact_broadcast_checks'] = len(re.findall(r'\[model-broadcast-check\].*transport=host_staged .*exact_match=1', text)) == 2
        checks['both_transfer_directions'] = all(len(re.findall(rf'\[model-transfer-check\] source=cuda:{src} destination=cuda:{1-src} method=cpu_staging', text)) == 2 for src in [0, 1])
        checks['no_direct_transport'] = 'method=cudaMemcpyPeerAsync' not in text and 'method=torch_copy' not in text
    else:
        # The native test accepts actionable direct-gate errors broadly. This
        # machine-specific controller accepts only the observed payload mismatch,
        # never missing devices, API failures, unavailable P2P, or arbitrary errors.
        checks['specific_payload_mismatch_rejected'] = bool(re.search(
            r'expected_direct_payload_rejection=1 reason=Multi-GPU direct_peer payload validation failed from cuda:[01] to cuda:[01]: '
            r'(?:cudaMemcpyPeerAsync|torch_copy) payload mismatch \(262144 values\)', text))
        checks['actionable_host_fallback_flag'] = 'GEGE_STATEFLOW_PEER_RELAY_FORCE_HOST_FALLBACK=1' in text
        checks['failed_before_replica_broadcast'] = '[model-broadcast-check]' not in text and 'Broadcast to GPU' not in text
    return checks


def run_case(mode, binary, library, env, state):
    global LIVE
    case = OUT / mode
    case.mkdir()
    write(case / 'gpu_before.json', require_idle())
    invocation = {'command': [str(binary), mode], 'started_utc': now(), 'timeout_seconds': 180,
                  'environment': {k: env[k] for k in ['CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'LD_LIBRARY_PATH', 'GEGE_NO_BINDINGS']}}
    write(case / 'execution.json', invocation)
    state.update(status='running', active_case=mode, updated_utc=now())
    write(OUT / 'status.json', state)
    runtime = None
    started = time.monotonic()
    with (case / 'test.log').open('x') as log:
        LIVE = subprocess.Popen(invocation['command'], env=env, cwd=case, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        invocation['pid'] = LIVE.pid
        write(case / 'execution.json', invocation)
        try:
            while LIVE.poll() is None:
                require(time.monotonic() - started < 180, 'Native test timeout: ' + mode)
                if runtime is None:
                    runtime = runtime_snapshot(LIVE.pid, env, library)
                time.sleep(.05 if runtime is None else .2)
            code = LIVE.wait()
        finally:
            stop_owned()
    LIVE = None
    text = (case / 'test.log').read_text(errors='replace')
    checks = result_checks(mode, text, code)
    checks['runtime_library_and_environment_verified'] = runtime is not None
    after = require_idle()
    write(case / 'gpu_after.json', after)
    result = {'mode': mode, 'status': 'passed' if all(checks.values()) else 'failed', 'exit_code': code,
              'checks': checks, 'runtime': runtime, 'ldd_verified_before_execution': True,
              'elapsed_seconds': time.monotonic() - started, 'finished_utc': now(),
              'test_log_sha256': sha(case / 'test.log'), 'execution_sha256': sha(case / 'execution.json')}
    write(case / 'result.json', result)
    require(result['status'] == 'passed', 'Native transfer gate failed: ' + mode + ' ' + json.dumps(checks))
    return result


def main():
    lock = (ROOT / 'transfer_gate.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    require(not OUT.exists() and not (ROOT / 'transfer_gate.json').exists(), 'Prior transfer gate exists; preserve completed or partial evidence and inspect it')
    OUT.mkdir()
    state = {'status': 'preflight', 'started_utc': now(), 'pid': os.getpid(), 'runner_sha256': sha(__file__), 'cases': []}
    write(OUT / 'status.json', state)
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    try:
        receipt = build_gate()
        binary = Path(receipt['artifacts']['gege_model_transfer_test']['path']).resolve()
        library = Path(receipt['artifacts']['libge2.so']['path']).resolve()
        env = environment(receipt)
        ldd = command(['ldd', str(binary)], env=env)
        (OUT / 'test_ldd.txt').write_text(ldd)
        require('not found' not in ldd, 'Unresolved native test dependency')
        require(bool(re.search(r'libge2\.so\s+=>\s+' + re.escape(str(library)) + r'\s', ldd)), 'Native test resolved another libge2.so')
        state.update(build_receipt_sha256=sha(ROOT / 'build_receipt.json'), artifacts=receipt['artifacts'],
                     source_manifest_sha256=receipt['source_manifest_sha256'], patch_manifest_sha256=receipt['patch_manifest_sha256'],
                     ldd_sha256=sha(OUT / 'test_ldd.txt'), gpu_uuids=UUIDS, cpu_threads=8)
        for mode in ['host', 'expect-direct-failure']:
            state['cases'].append(run_case(mode, binary, library, env, state))
            write(OUT / 'status.json', state)
        require(sha(ROOT / 'build_receipt.json') == state['build_receipt_sha256'], 'Build receipt changed during test')
        for name in ['gege_model_transfer_test', 'libge2.so']:
            entry = receipt['artifacts'][name]
            require(sha(entry['path']) == entry['sha256'], 'Test artifact changed during execution')
        state.update(status='passed', active_case=None, finished_utc=now(),
                     scope='Small native model transport tests only; no training or full dataset reads.',
                     limitations=['Direct mode is expected to fail specifically on this machine. A passing gate here means that unsafe peer payloads were rejected, not that P2P works.',
                                  'Host mode validates broadcast and rejects dense collectives; it does not validate FB/ComplEx collectives.'])
        write(ROOT / 'transfer_gate.json', state)
    except BaseException as error:
        stop_owned()
        state.update(status='failed', error=repr(error), traceback=traceback.format_exc(), finished_utc=now())
        raise
    finally:
        write(OUT / 'status.json', state)


if __name__ == '__main__':
    main()
