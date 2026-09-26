#!/usr/bin/env python3
"""Freeze a matched c30 campaign using existing, independently audited datasets."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

import yaml

from run_arc_paper_case import digest
from run_arc_pipege_best import freeze_baseline_sampling


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'old-base', 'old-work', 'engine', 'archive', 'summary', 'ge2-zip'):
        p.add_argument('--'+name, required=True, type=Path)
    p.add_argument('--commit', required=True)
    args = p.parse_args()
    base = args.base.resolve()
    if (base/'campaign.json').exists() or (base/'manifest.json').exists():
        raise ValueError('Refusing to overwrite a frozen campaign')
    old = json.loads((args.old_base/'manifest.json').read_text())
    for rel, expected in old['references'].items():
        if digest(args.old_base/rel) != expected:
            raise ValueError('Historical reference changed: '+rel)
    shutil.copytree(args.old_base/'references', base/'references')
    prepared_path = args.old_work/'prepared.json'
    prepared = json.loads(prepared_path.read_text())
    if prepared['status'] != 'ready' or set(prepared['data']) != {'lj','tw','fb','wk'}:
        raise ValueError('Missing audited input views')
    engine = dict(old['engine'], root=str(args.engine),
                  gradient_gate=str(base/'work/paired_gradient_gate/result.json'))
    if (args.engine/'build_git_completed_commit.txt').read_text().strip() != engine['commit']:
        raise ValueError('Unexpected native build')
    for name, key in [('libge2.so','libge2_so'), ('gege_train','gege_train')]:
        if digest(args.engine/'build_git'/name) != engine['hashes'][key]:
            raise ValueError('Native binary hash mismatch')
    if not Path(engine['gate_template']).is_dir():
        raise ValueError('Missing gradient gate fixture')
    subprocess.run(['git','-C',str(args.engine/'repo'),'bundle','verify',str(base/'source.bundle')], check=True)
    for case, spec in old['cases'].items():
        spec.update(negative_mass_scale=1, report_directions='tail',
                    notes='Matched 300 W cohort; mass 1; independent head/tail draws; tail reporting')
        flags_path = base/spec['flags']
        flags = json.loads(flags_path.read_text())
        flags.update(GEGE_BASELINE_TRAINING_SEMANTICS='1', GEGE_SOFTMAX_NEGATIVE_MASS_SCALE='1')
        flags.pop('GEGE_SOFTMAX_NEGATIVE_LOG_MASS_BIAS', None)
        config_path = base/spec['config']
        config, flags = freeze_baseline_sampling(yaml.safe_load(config_path.read_text()), flags)
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        spec['sampling_policy'] = 'independent head/tail draws; no negative-plan or state-pool reuse'
        flags_path.write_text(json.dumps(flags, indent=2, sort_keys=True)+'\n')
        if digest(Path(spec['query'])) != spec['eval_sha']:
            raise ValueError('Query hash mismatch: '+case)
        if not Path(prepared['data'][spec['graph']]['view']).is_dir():
            raise ValueError('Missing partitioned view: '+case)
    shutil.copyfile(args.ge2_zip, base/'ge2.zip')
    refs = {str(f.relative_to(base)):digest(f) for f in (base/'references').rglob('*') if f.is_file()}
    helpers = {str(f.relative_to(base)):digest(f) for directory in ('scripts','harness')
               for f in (base/directory).rglob('*') if f.is_file() and '__pycache__' not in f.parts}
    manifest = dict(old, engine=engine, references=refs, helpers=helpers,
        reuse_prepared=dict(path=str(prepared_path), commit=prepared['commit'],
                            manifest_sha256=prepared['manifest_sha256']),
        interpretation='Fresh isolated single-GPU GE2/PipeGE; c30 300 W; tail filtered exact 10K',
        supersedes_recipe_not_results=str(args.old_base/'manifest.json'))
    (base/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    # Check persistent storage now, and again inside every allocation.
    args.archive.mkdir(parents=True, exist_ok=True)
    probe = args.archive/'campaign_probe.json'
    probe.write_text(json.dumps(dict(base=str(base), commit=args.commit))+'\n')
    if json.loads(probe.read_text())['commit'] != args.commit:
        raise ValueError('Persistent archive readback failed')
    args.summary.mkdir(parents=True, exist_ok=True)
    frozen = dict(refs, **helpers)
    for rel in ('manifest.json', 'source.bundle', 'ge2.zip'):
        frozen[rel] = digest(base/rel)
    campaign = dict(node='c30', power_w=300, commit=args.commit,
                    env='/mnt/local/smansou2/ge2-a6000-cuda121', summary=str(args.summary),
                    archive=str(args.archive), files=frozen)
    (base/'campaign.json').write_text(json.dumps(campaign, indent=2, sort_keys=True)+'\n')
    for name in ('campaign.json','manifest.json'):
        shutil.copyfile(base/name, args.summary/name)
    print(json.dumps(dict(status='staged', base=str(base), commit=args.commit,
                         campaign_sha256=digest(base/'campaign.json'), archive=str(args.archive))))


if __name__ == '__main__':
    main()
