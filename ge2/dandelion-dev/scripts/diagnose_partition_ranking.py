#!/usr/bin/env python3
"""Decompose frozen tail ranks by the true tail's logical partition.

This is an accuracy diagnostic, not a new paper evaluation protocol. It uses
the existing filtered evaluator and ranking helper without changing either.
"""
import argparse
import csv
import fcntl
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import numpy as np
import torch
import exact_eval_ranking

from eval_marius_kge_exact10k import (
    DotRanker, apply_row_filters, build_filters, configure_scoring,
    exclude_targets, load_triples, make_queries, metric_values, raw_matrix, require_finite,
    sha256_file,
)


def same_partition_mask(tails, candidate_start, candidate_stop, nodes, partitions):
    if nodes < 1 or not 1 <= partitions <= nodes:
        raise ValueError('Invalid partition count or entity population')
    if not 0 <= candidate_start < candidate_stop <= nodes:
        raise ValueError('Invalid candidate interval')
    size = (nodes + partitions - 1) // partitions
    candidates = torch.arange(candidate_start, candidate_stop, device=tails.device)
    return tails[:, None] // size == candidates[None, :] // size


def partition_counts(scores, candidates, ranker, tails, candidate_start, nodes,
                     partitions, query_start, candidate_absmax):
    mask = same_partition_mask(tails, candidate_start,
                               candidate_start + candidates.shape[0], nodes, partitions)
    local_scores = scores.masked_fill(~mask, -torch.inf)
    return ranker.count(local_scores, candidates, query_start, candidate_absmax)


def validate_reference(reference, triples, ranks):
    if (reference['report_directions'] != 'tail' or not reference['filtered']
            or reference['tf32'] is not False or reference['mapping_contract'] != 'identity'
            or reference['ranking_contract'] != 'fp32_query_fp64_dot_refinement_v1'):
        raise ValueError('Need a frozen identity-mapped, filtered, exact tail report')
    if (triples.shape != (reference['num_eval_edges'], 3)
            or ranks.shape != (reference['num_eval_edges'],)
            or not np.issubdtype(ranks.dtype, np.integer)
            or np.any(ranks < 1) or np.any(ranks > reference['num_nodes'])):
        raise ValueError('Invalid saved query or rank array')
    if (not np.isclose(np.mean(1.0 / ranks), reference['mrr'], rtol=0, atol=1e-12)
            or not np.isclose(np.mean(ranks <= 10), reference['hits_at_10'], rtol=0, atol=1e-12)):
        raise ValueError('Frozen report disagrees with its saved ranks')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--partitions', type=int, nargs='+', default=[16, 32])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--workstation-host', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--after', type=Path, help='Wait for this diagnostic/control JSON to finish')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        execute(args)
    except BaseException as error:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if not args.out.exists():
            with args.out.open('x') as output:
                json.dump(dict(status='failed', paper_ready=False, error=repr(error)), output)
        raise


def execute(args):
    hosts = {socket.gethostname().split('.')[0], socket.getfqdn().split('.')[0]}
    if args.workstation_host.split('.')[0] not in hosts or os.environ.get('SLURM_JOB_ID'):
        raise ValueError('This diagnostic is restricted to the authorized workstation')
    deadline = time.monotonic() + 12 * 3600
    while args.after:
        if time.monotonic() >= deadline:
            raise TimeoutError('Parent did not finish before the diagnostic deadline')
        if args.after.exists():
            try:
                parent = json.loads(args.after.read_text())
            except json.JSONDecodeError:
                time.sleep(15)
                continue
            if parent.get('status') == 'failed':
                raise RuntimeError('Parent diagnostic failed')
            if parent.get('status') == 'done':
                break
        time.sleep(15)
    uuid = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-gpu=uuid',
                                   '--format=csv,noheader'], text=True, timeout=20).strip()
    lock = Path.home() / ('.fb_accuracy_gpu_' + uuid + '.lock')
    lock_file = lock.open('a')
    fcntl.flock(lock_file, fcntl.LOCK_EX)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                   '--format=csv,noheader'], text=True, timeout=20)
    if any(row and row[0] == uuid for row in csv.reader(io.StringIO(apps), skipinitialspace=True)):
        raise RuntimeError('The selected GPU is still busy; no process will be displaced')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    configure_scoring()
    torch.set_num_threads(8)
    reference = json.loads(args.reference.read_text())
    nodes, relations, width = reference['num_nodes'], reference['num_relations'], reference['entity_shape'][1]
    partitions = sorted(set(args.partitions))
    if not partitions or any(not 1 <= p <= nodes for p in partitions):
        raise ValueError('Partition counts must lie between one and the entity population')
    if sha256_file(Path(exact_eval_ranking.__file__)) != reference['ranking_helper_sha256']:
        raise ValueError('Frozen ranking helper changed')
    rank_path = args.reference.with_suffix('.ranks.npz')
    if sha256_file(rank_path) != reference['ranks_sha256']:
        raise ValueError('Frozen rank-file hash mismatch')
    with np.load(rank_path, allow_pickle=False) as saved:
        old_triples, old_ranks = saved['triples'], saved['tail_ranks']
    validate_reference(reference, old_triples, old_ranks)
    for path_key, hash_key in [('eval_edges', 'eval_edges_sha256'),
                               ('entity_bin', 'entity_bin_sha256'),
                               ('src_relation_bin', 'src_relation_bin_sha256'),
                               ('dst_relation_bin', 'dst_relation_bin_sha256')]:
        if sha256_file(Path(reference[path_key])) != reference[hash_key]:
            raise ValueError('Changed frozen artifact: ' + path_key)
    count = len(old_ranks)
    triples = load_triples(Path(reference['eval_edges']), None, None, nodes, relations, count,
                           reference.get('sample_seed'))
    if not np.array_equal(triples, old_triples):
        raise ValueError('Frozen queries disagree with saved ranks')
    started = time.monotonic()
    tail_filters, _ = build_filters(Path(reference['ge2_data_dir']), None, None, triples,
                                    nodes, relations, reference['filter_chunk'])
    print('Filters ready; rescoring frozen tail queries', flush=True)
    entity = raw_matrix(Path(reference['entity_bin']), nodes, width)
    rel = raw_matrix(Path(reference['src_relation_bin']), relations, width)
    if args.device != 'cuda:0':
        raise ValueError('Use cuda:0 after selecting the physical GPU')
    device = torch.device(args.device)
    heads_np, rels_np, tails_np = triples.T
    heads = torch.from_numpy(np.array(entity[heads_np], copy=True)).to(device)
    src_relations = torch.from_numpy(np.array(rel[rels_np], copy=True)).to(device)
    targets = torch.from_numpy(np.array(entity[tails_np], copy=True)).to(device)
    query, _, _, _ = make_queries(reference['score'], heads, src_relations, src_relations, targets)
    tails = torch.from_numpy(tails_np.copy()).to(device)
    keys = heads_np * np.int64(relations) + rels_np
    global_ranker = DotRanker(query, targets, reference['tie_policy'])
    local_rankers = {p: DotRanker(query, targets, reference['tie_policy']) for p in partitions}
    global_counts = torch.zeros(count, dtype=torch.int64, device=device)
    local_counts = {p: torch.zeros_like(global_counts) for p in partitions}
    chunk = reference['candidate_chunk']
    batch = reference['batch_size']
    with torch.no_grad():
        for lo in range(0, nodes, chunk):
            hi = min(lo + chunk, nodes)
            candidates = torch.from_numpy(np.array(entity[lo:hi], copy=True)).to(device)
            require_finite(candidates, 'candidate embeddings')
            absmax = candidates.abs().max().double()
            for start in range(0, count, batch):
                stop = min(start + batch, count)
                scores = query[start:stop] @ candidates.T
                require_finite(scores, 'unfiltered scores')
                apply_row_filters(scores, keys[start:stop], tails_np[start:stop], tail_filters, lo, hi)
                exclude_targets(scores, tails_np[start:stop], lo, hi)
                global_counts[start:stop] += global_ranker.count(scores, candidates, start, absmax)
                for p in partitions:
                    local_counts[p][start:stop] += partition_counts(
                        scores, candidates, local_rankers[p], tails[start:stop], lo, nodes, p, start, absmax)
            if lo // chunk % 20 == 0:
                print('Candidate rows:', hi, '/', nodes, flush=True)
    actual_ranks = global_counts.cpu().numpy() + 1
    if not np.array_equal(actual_ranks, old_ranks):
        raise ValueError('Rescored global ranks changed; refusing a partition diagnosis')
    arrays = {'triples': triples, 'global_tail_ranks': actual_ranks}
    summaries = {}
    for p, value in local_counts.items():
        local = value.cpu().numpy() + 1
        outside = actual_ranks - local
        if np.any(outside < 0):
            raise ValueError('Local competitor count exceeds the global count')
        arrays['within_p' + str(p) + '_tail_ranks'] = local
        summaries[str(p)] = dict(
            **metric_values(local.astype(np.float64)),
            mean_outside_partition_outrankers=float(outside.mean()),
            locally_top10_but_not_globally_top10=float(np.mean((local <= 10) & (actual_ranks > 10))),
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists() or args.out.with_suffix('.ranks.npz').exists():
        raise ValueError('Refusing to overwrite diagnostic evidence')
    ranks_out = args.out.with_suffix('.ranks.npz')
    np.savez_compressed(ranks_out, **arrays)
    result = dict(status='done', paper_ready=False, purpose='Local versus cross-partition ranking diagnosis',
                  caveat='Restricted-catalog ranks are not comparable to the main paper metrics',
                  reference=str(args.reference), reference_sha256=sha256_file(args.reference),
                  query_sha256=reference['eval_edges_sha256'], global_ranks_reproduced_exactly=True,
                  mrr=reference['mrr'], hits_at_10=reference['hits_at_10'], within_partition=summaries,
                  ranks_sha256=sha256_file(ranks_out), elapsed_s=time.monotonic()-started,
                  script_sha256=sha256_file(Path(__file__)))
    with args.out.open('x') as output:
        json.dump(result, output, indent=2)
        output.write('\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
