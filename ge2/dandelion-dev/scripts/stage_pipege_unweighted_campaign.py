#!/usr/bin/env python3
"""Make a new mass-1 campaign from frozen inputs, preserving historical evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from run_arc_pipege_best import loss_contract, report_directions


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('old-base', 'base', 'old-work', 'work', 'engine'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    base = args.base
    if (base/'manifest.json').exists():
        raise ValueError('Refusing to overwrite a frozen campaign')
    old = json.loads((args.old_base/'manifest.json').read_text())
    for rel, expected in {**old['references'], **old['helpers']}.items():
        if digest(args.old_base/rel) != expected:
            raise ValueError('Historical frozen input changed: '+rel)
    for directory in ('references', 'harness'):
        shutil.copytree(args.old_base/directory, base/directory)
    build = json.loads((base/'engine_build.json').read_text())
    if build['source_commit'] != args.commit or digest(base/'engine.tar.gz') != build['archive_sha256']:
        raise ValueError('Engine archive provenance mismatch')
    manifest = dict(old)
    manifest['engine'] = dict(commit=args.commit, root=str(args.engine),
                             hashes=dict(libge2_so=build['files']['libge2.so'], gege_train=build['files']['gege_train']),
                             gradient_gate=str(args.work/'paired_gradient_gate/result.json'),
                             gate_template=old['engine']['gate_template'])
    manifest['reuse_prepared'] = dict(path=str(args.old_work/'prepared.json'),
                                     commit=json.loads((args.old_work/'prepared.json').read_text())['commit'],
                                     manifest_sha256=digest(args.old_base/'manifest.json'))
    manifest['interpretation'] = 'Unweighted loss; otherwise unchanged fast runtime; TW tail-only reporting'
    manifest['supersedes_recipe_not_results'] = str(args.old_base/'manifest.json')
    for case, spec in manifest['cases'].items():
        flags_path = base/spec['flags']
        flags = json.loads(flags_path.read_text())
        flags['GEGE_SOFTMAX_NEGATIVE_MASS_SCALE'] = '1'
        flags.pop('GEGE_SOFTMAX_NEGATIVE_LOG_MASS_BIAS', None)
        loss_contract(flags)
        spec.update(negative_mass_scale=1, report_directions='tail' if spec['graph'] == 'tw' else 'both')
        report_directions(spec)
        spec['notes'] = 'Mass 1; same runtime/data/seed as prior campaign; '+spec['report_directions']+' evaluation'
        flags_path.write_text(json.dumps(flags, indent=2, sort_keys=True)+'\n')
    manifest['references'] = {str(p.relative_to(base)):digest(p) for p in (base/'references').rglob('*') if p.is_file()}
    manifest['helpers'] = {str(p.relative_to(base)):digest(p) for directory in ('scripts', 'harness')
                           for p in (base/directory).rglob('*') if p.is_file() and '__pycache__' not in str(p)}
    for name in ('engine_build.json', 'engine.tar.gz'):
        manifest['helpers'][name] = digest(base/name)
    (base/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    print(json.dumps(dict(status='staged', manifest_sha256=digest(base/'manifest.json'), engine=manifest['engine']), indent=2))


if __name__ == '__main__':
    main()
