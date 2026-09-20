#!/usr/bin/env python3
"""Freeze existing fast presets and audited evaluation contracts for ARC replay."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil

import yaml

ROOT = Path('/home/smansou2/ge2_patch_work/tacc_baselines')
HOME = Path('/home/smansou2')
BASE = Path('/mnt/local/smansou2')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shell_flags(path):
    flags = {}
    for line in path.read_text().splitlines():
        words = shlex.split(line, comments=True)
        if words and words[0] == 'export':
            words = words[1:]
        if len(words) != 1 or not words[0].startswith('GEGE_') or '=' not in words[0]:
            continue
        name, value = words[0].split('=', 1)
        match = re.fullmatch(r'\$\{[A-Z0-9_]+:-([^{}]*)\}', value)
        if match:
            value = match[1]
        if '$' in value:
            raise ValueError('Unresolved shell expansion: ' + line)
        flags[name] = value
    return flags


def stage(base, cache_root=None):
    refs = base / 'references'
    exp = HOME / 'codex_runs/exp_logs'
    wk = HOME / 'wk_p30_local'
    fb_dm = next((exp / 'fb86m_p32_q4_distmult_1gpu_samecfg_10e_eval_local_20260613_142854').glob('*_train.yaml'))
    fb_cx = next((exp / 'fb86m_p32_q4_complex_1gpu_samecfg_10e_eval_local_20260612_154735').glob('*_train.yaml'))
    wk_cfg = wk / 'configs/wk_p30_q4_75state_distmult_ge2style_hf6_stale3_pinmapped_10e_eval_local_20260614_114100.yaml'
    wk_driver = wk / 'logs/wk_p30_q4_75state_complex_hf6_stale3_pinmapped_10e_eval_local_20260613_183404_driver.log'
    fb_flags = ROOT / 'runs/local_fb_fast_replay_20260920/historical_flags.json'
    configs = dict(lj_dot=refs/'lj/ours_LJ_Dot_1gpu.yaml', tw_dot=refs/'tw/archived_config.yaml',
                   fb_distmult=fb_dm, fb_complex=fb_cx, wk_distmult=wk_cfg, wk_complex=wk_cfg)
    common = dict(GEGE_BASELINE_TRAINING_SEMANTICS='0', GEGE_FRAME_CACHE_AUTO_PIPELINE_FRAMES='0',
                  GEGE_FRAME_CACHE_FIXED_PRELOAD_FRAMES='-1', GEGE_FRAME_CACHE_STRICT_FRAME_BUDGET='1',
                  GEGE_STATE_NEGATIVE_POOL_REFRESH_BATCHES='0', GEGE_GLOBAL_DEGREE_SAMPLING='0',
                  GEGE_MULTI_GPU_ASYNC_ADMIT_PRELOAD='0', GEGE_PARTITION_BUFFER_PEER_RELAY='0',
                  GEGE_STATEFLOW_ALLOW_PEER_RELAY='0', GEGE_STATEFLOW_ENABLE_UNVERIFIED_PEER_RELAY_RUNTIME='0',
                  GEGE_PARTITION_BUFFER_PIPELINE_TIMING='0', GEGE_PARTITION_BUFFER_SWAP_TIMING='0',
                  GEGE_PARTITION_BUFFER_REMAP_BREAKDOWN_TIMING='0', GEGE_STARTUP_TIMING='1',
                  GEGE_FIXED_BUFFER_MANUAL_DOT_RNS_VERIFY='0', GEGE_FIXED_BUFFER_MANUAL_COMPLEX_RNS_VERIFY='0')
    specs = {
        'lj': dict(p=2,q=2,hidden=0,states=1,nodes=4847571,relations=1,width=100,edges=62094395,epochs=30,columns=2,
                   source=str(BASE/'pipege_lj_281586/data/lj_p2'), query=str(BASE/'ge2-paper-data/LJ/exact10000/edges/test_edges.bin'),
                   eval_sha='1d1669b2fa920c8492341989a0c0ff2c741d97e74ace67ef76735ed29fb5691e', scope='held_out_test'),
        'tw': dict(p=16,q=4,hidden=3,states=20,nodes=41652230,relations=1,width=100,edges=1321528663,epochs=10,columns=2,
                   source=str(BASE/'ge2-paper-data/TW/ge2_table4_controlled_v1'),
                   query=str(BASE/'ge2-paper-data/TW/ge2_table4_controlled_v1/exact10000_uniform_v2/edges/test_edges.bin'),
                   eval_sha='93bf1dd7104a2a225800229abb62fbf7477ef006807d0ce523fe6094f3e3ded6',scope='held_out_test'),
        'fb': dict(p=32,q=4,hidden=6,states=88,nodes=86054151,relations=14824,width=100,edges=304727650,epochs=10,columns=3,
                   source=str(BASE/'ge2_kge_final_20260919/cache/FB'),
                   query=str(BASE/'ge2_kge_final_20260919/cache/FB/exact10000_uniform_v1/edges/test_edges.bin'),
                   eval_sha='192108226f4824df15466683d1001bd0ab692e926dfa5c26ad04cab31ad544fb',scope='held_out_test'),
        'wk': dict(p=30,q=4,hidden=6,states=75,nodes=91230610,relations=1387,width=80,edges=601062811,epochs=10,columns=3,
                   source=str(BASE/'ge2_kge_final_20260919/wk_distmult_284527/data'),
                   query=str(BASE/'ge2_kge_final_20260919/wk_distmult_284527/data/exact10000_uniform_v1/edges/test_edges.bin'),
                   eval_sha='9ecb8565232a4a20beb61e2475fcf4222dfee3937448671b6d313107ed82f64e',scope='public_validation_not_hidden_test'),
    }
    schedules = dict(tw=refs/'tw/state_order.txt', wk=HOME/'codex_runs/schedule_dump/wk_p30_q4_cyclic75_overlap2x14_order_20260608.txt')
    if cache_root:
        for graph, spec in specs.items():
            source, query = Path(spec['source']), Path(spec['query'])
            spec.update(source_origin=str(source), query_origin=str(query))
            target = cache_root/graph
            spec['source'] = str(target)
            spec['query'] = str(target/query.relative_to(source)) if source in query.parents else str(target/'exact10000/edges/test_edges.bin')
    manifest = dict(cases={}, references={}, interpretation='Archived fast policy replay on audited splits; repaired gradients; quality pending')
    for name, path in configs.items():
        graph, model = name.split('_')
        spec = dict(specs[graph], model=model, graph=graph, case=name)
        case = refs / name
        case.mkdir(exist_ok=False)
        cfg = yaml.safe_load(path.read_text())
        if graph == 'lj':
            flags = shell_flags(refs/'lj/env_flags.sh')
            flag_source = refs/'lj/env_flags.sh'
            spec['notes'] = 'Archived p2/q2 resident-domain, state-shuffle manual profile; repaired gradients'
        elif graph == 'tw':
            flags = {k:str(v) for k,v in json.loads((refs/'tw/archived_invocation.json').read_text()).items() if k.startswith('GEGE_')}
            flags['PYTORCH_CUDA_ALLOC_CONF'] = json.loads((refs/'tw/archived_invocation.json').read_text())['PYTORCH_CUDA_ALLOC_CONF']
            flag_source = refs/'tw/archived_invocation.json'
            spec['notes'] = 'Fast shared-3, 8-batch negative reuse; controlled 90% training split, not full-edge timing workload'
        elif graph == 'fb':
            flags = json.loads(fb_flags.read_text())
            flag_source = fb_flags
            spec['notes'] = 'Historical mass-8, graph-prefetch-off shared-6 profile on audited FB split'
        else:
            flags = shell_flags(wk_driver)
            flag_source = wk_driver
            cfg['model']['decoder']['type'] = 'COMPLEX' if model == 'complex' else 'DISTMULT'
            spec['notes'] = 'Bias-free GE2-style learning recipe plus archived p30 fast runtime; not exact replay of biased Adam-0.01 YAML'
        flags.update(common)
        flags.pop('GEGE_TRAINING_REPLAY_SEED', None)
        flags.pop('GEGE_TRAINING_INPUT_AUDIT', None)
        flags.pop('GEGE_BOUNDED_STATE_ORDER_FILE', None)
        flags['GEGE_FRAME_CACHE_HIDDEN_FRAMES'] = str(spec['hidden'])
        flags['GEGE_BATCHED_NEGATIVE_PLAN_BATCHES'] = '8' if graph in ('lj','tw') else '0'
        for decoder in ('dot','distmult','complex'):
            flags['GEGE_FIXED_BUFFER_MANUAL_'+decoder.upper()+'_RNS'] = str(int(model == decoder))
        for key in ('GEGE_FIXED_BUFFER_MASKED_UPDATE_VERIFY',):
            flags[key] = '0'
        if graph in schedules:
            shutil.copyfile(schedules[graph], case/'schedule.txt')
            spec['schedule'] = str((case/'schedule.txt').relative_to(base))
        spec['reference_config_sha256'] = digest(path)
        spec['reference_path'] = str(path)
        spec['reference_flags_sha256'] = digest(flag_source)
        spec['reference_flags_path'] = str(flag_source)
        (case/'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
        (case/'flags.json').write_text(json.dumps(flags, indent=2, sort_keys=True)+'\n')
        spec['config'] = str((case/'config.yaml').relative_to(base))
        spec['flags'] = str((case/'flags.json').relative_to(base))
        manifest['cases'][name] = spec
    for path in sorted(refs.rglob('*')):
        if path.is_file():
            manifest['references'][str(path.relative_to(base))] = digest(path)
    tools = base/'harness/tools'
    tools.mkdir(parents=True, exist_ok=True)
    for path in (ROOT/'tools').glob('*.py'):
        shutil.copyfile(path, tools/path.name)
    manifest['helpers'] = {str(p.relative_to(base)):digest(p) for p in tools.glob('*.py')}
    (base/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    print(json.dumps({k:{x:v[x] for x in ('p','q','hidden','states','epochs','notes')} for k,v in manifest['cases'].items()},indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--cache-root', type=Path)
    args = p.parse_args()
    stage(args.base.resolve(), args.cache_root)
