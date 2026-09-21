#!/usr/bin/env python3
"""Derive a direction-specific report without rescoring or modifying old evidence."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive(source, ranks_path, direction):
    if direction not in ('head', 'tail', 'both'):
        raise ValueError('Unknown reporting direction')
    original = json.loads(source.read_text())
    if digest(ranks_path) != original['ranks_sha256']:
        raise ValueError('Saved-rank hash mismatch')
    with np.load(ranks_path, allow_pickle=False) as archive:
        head, tail = archive['head_ranks'], archive['tail_ranks']
        for ranks in (head, tail):
            if (ranks.shape != (original['num_eval_edges'],)
                    or not np.issubdtype(ranks.dtype, np.integer)
                    or np.any(ranks < 1) or np.any(ranks > original['num_nodes'])):
                raise ValueError('Invalid saved ranks')
        both = np.concatenate((head, tail))
        used = {'head': head, 'tail': tail, 'both': both}[direction].astype(np.float64)
        previous = {'head': head, 'tail': tail, 'both': both}[original.get('report_directions', 'both')]
        if (not np.isclose(np.mean(1.0/previous), original['mrr'], rtol=0, atol=1e-12)
                or not np.isclose(np.mean(previous <= 10), original['hits_at_10'], rtol=0, atol=1e-12)):
            raise ValueError('Saved ranks disagree with original reported metrics')
        result = dict(original, report_directions=direction, num_ranks=int(used.size),
                      mrr=float(np.mean(1.0/used)), avg_rank=float(np.mean(used)))
        for k in (1, 5, 10, 20, 50, 100):
            result['hits_at_'+str(k)] = float(np.mean(used <= k))
    result['derived_report'] = dict(source=str(source.resolve()), source_sha256=digest(source),
                                   local_ranks=str(ranks_path.resolve()), rescored=False,
                                   created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--ranks', type=Path, required=True)
    parser.add_argument('--direction', choices=('head', 'tail', 'both'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = derive(args.source, args.ranks, args.direction)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in ('report_directions', 'num_ranks', 'mrr', 'hits_at_10')}))
