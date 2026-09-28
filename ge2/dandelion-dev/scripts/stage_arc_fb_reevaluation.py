#!/usr/bin/env python3
"""Freeze the existing evaluator and evidence for a read-only ARC FB replay."""
import argparse
import json
from pathlib import Path
import shutil

from run_arc_fb_reevaluation import digest, validate_protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--harness', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    scripts = Path(__file__).resolve().parent
    tools = args.out/'tools'
    tools.mkdir()
    for name in ('eval_marius_kge_exact10k.py', 'eval_dglke_kge_exact10k.py',
                 'exact_eval_ranking.py', 'extract_ge2_relation_embeddings.py'):
        shutil.copy2(args.harness/'tools'/name, tools/name)
    for name in ('run_arc_fb_reevaluation.py', 'arc_accuracy_gpu_guard.py', 'arc_job_support.py'):
        shutil.copy2(scripts/name, args.out/name)
    previous = args.harness/'results/arc_matched_300w_20260926/summary'
    archive = Path('/mnt/beegfs/smansou2/paper_matched_300w_20260925')
    data = Path('/mnt/beegfs/smansou2/fb_eval_recheck_20260928/data')
    cases = []
    for name in ('290698_pipege_fb_complex', '290702_pipege_fb_distmult'):
        source = previous/name
        destination = args.out/'previous'/name
        destination.mkdir(parents=True)
        for item in ('raw_exact_eval.json', 'raw_exact_eval.ranks.npz', 'raw_checkpoint_manifest.json'):
            shutil.copy2(source/item, destination/item)
        old = json.loads((destination/'raw_exact_eval.json').read_text())
        validate_protocol(old)
        files = json.loads((destination/'raw_checkpoint_manifest.json').read_text())['files']
        hashes = {item['path']: item['sha256'] for item in files}
        if hashes['embeddings.bin'] != old['entity_bin_sha256']:
            raise ValueError('Checkpoint and evaluation identity mismatch: '+name)
        paths = dict(entity_bin=str(archive/name/'checkpoint/embeddings.bin'),
                     ge2_data_dir=str(data), eval_edges=str(data/'exact10000_uniform_v1/edges/test_edges.bin'))
        case = dict(name=name, paths=paths,
            model=str(archive/name/'checkpoint/model.pt_0'), model_sha256=hashes['model.pt_0'],
            previous_eval=str(Path('previous')/name/'raw_exact_eval.json'),
            previous_ranks=str(Path('previous')/name/'raw_exact_eval.ranks.npz'),
            filter_hashes=dict(
                train='fcbf8d7fc859221e1e95d3ed8c3917c2a68738d1ffbdeec3a0d92517474173ab',
                validation='a9922866ed3c1995bab4e3eff0d98060f01ee1f3424badc3801c73334d3eaedf',
                test='d8a318ec2aa4e3bd088123161c349bf7ac1d31c8f2ef1624415c80509c382aee'))
        if 'pipege' in name:
            shutil.copy2(source/'raw_flags.json', destination/'raw_flags.json')
            case.update(flags=str(Path('previous')/name/'raw_flags.json'), required_flags={
                'GEGE_BASELINE_TRAINING_SEMANTICS': '1',
                'GEGE_BATCHED_NEGATIVE_PLAN_BATCHES': '0',
                'GEGE_SOFTMAX_NEGATIVE_MASS_SCALE': '1'})
        cases.append(case)
    payload = {str(path.relative_to(args.out)): digest(path)
               for path in sorted(args.out.rglob('*')) if path.is_file()}
    manifest = dict(cases=cases, payload_sha256=payload, training_modified=False,
                    report_directions='tail', filtering_data='hash-verified original ARC split')
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(status='staged', manifest=str(args.out/'manifest.json'), cases=len(cases))))


if __name__ == '__main__':
    main()
