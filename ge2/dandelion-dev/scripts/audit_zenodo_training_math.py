#!/usr/bin/env python3
"""Compare installed GE2 math with independent equations, without changing training.

The Python release cannot configure the RNS training dispatcher. This audit uses
its differentiable forward_lp(train=False) scoring path, then the native loss,
Batch.accumulateGradients and native dense optimizer. It does not certify the
complete executable, storage transitions, or the paper's experimental protocol.
"""

import argparse
import faulthandler
import hashlib
import json
import math
from pathlib import Path
import socket
import time


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_case(gege, torch, device, kind, width, rows, chunks, reduction, seed):
    torch.manual_seed(seed)
    n, r, negatives = 37, 3, 12
    make = lambda *shape: torch.randn(*shape, device=device) * .2
    embeddings = make(n, width).requires_grad_()
    reference = embeddings.detach().clone().requires_grad_()
    ctor = gege.nn.decoders.edge.ComplEx if kind == 'complex' else gege.nn.decoders.edge.DistMult
    decoder = ctor(r, width, kind != 'dot', device, torch.float32, 'CORRUPT_NODE')
    decoder.relations = make(r, width).requires_grad_()
    relation = decoder.relations.detach().clone().requires_grad_()
    inv_relation = None
    if kind != 'dot':
        decoder.inverse_relations = make(r, width).requires_grad_()
        inv_relation = decoder.inverse_relations.detach().clone().requires_grad_()
    edges = torch.stack((torch.randint(n, (rows,), device=device),
                         torch.randint(r, (rows,), device=device),
                         torch.randint(n, (rows,), device=device)), dim=1)
    edges[-1] = edges[0]
    if kind == 'dot':
        edges = edges[:, [0, 2]]
    dst = torch.randint(n, (chunks, negatives), device=device)
    src = torch.randint(n, (chunks, negatives), device=device)
    dst[:, 1] = dst[:, 0]
    src[:, 1] = src[:, 0]
    filt = torch.tensor([[0, 2], [rows - 1, 3]], device=device)
    batch = gege.data.Batch(True)
    batch.batch_size = rows
    batch.edges = edges
    batch.node_embeddings = embeddings
    batch.node_embeddings_state = torch.rand_like(embeddings) + .1
    old_state = batch.node_embeddings_state.clone()
    state_alias = batch.node_embeddings_state
    batch.dst_neg_indices_mapping = dst
    batch.src_neg_indices_mapping = src
    batch.dst_neg_filter = filt
    batch.src_neg_filter = filt
    model = gege.nn.Model(gege.nn.encoders.GeneralEncoder([[gege.nn.layers.EmbeddingLayer(width, device)]]), decoder)
    actual = model.forward_lp(batch, False)
    loss_fn = gege.nn.SoftmaxCrossEntropy(reduction.upper())
    per_chunk = math.ceil(rows / chunks)
    chunk_ids = torch.arange(rows, device=device) // per_chunk

    def score(h, rel, t):
        if kind == 'dot':
            return (h * t).sum(-1)
        if kind == 'distmult':
            return (h * rel * t).sum(-1)
        hr, hi = h.chunk(2, -1)
        rr, ri = rel.chunk(2, -1)
        tr, ti = t.chunk(2, -1)
        return (hr * rr * tr + hi * rr * ti + hr * ri * ti - hi * ri * tr).sum(-1)

    expected = []
    ref_losses, native_losses = [], []
    for inverse in range(1 if kind == 'dot' else 2):
        h = reference[edges[:, -1 if inverse else 0]]
        t = reference[edges[:, 0 if inverse else -1]]
        rels = (inv_relation if inverse else relation)[edges[:, 1]] if kind != 'dot' else None
        neg = reference[(src if inverse else dst)[chunk_ids]]
        pos = score(h, rels, t)
        neg_scores = score(h[:, None], rels[:, None] if rels is not None else None, neg)
        neg_scores = neg_scores.index_put((filt[:, 0], filt[:, 1]), torch.tensor(-1e9, device=device))
        expected.extend((pos, neg_scores))
        ref_losses.append(torch.nn.functional.cross_entropy(torch.cat((pos[:, None], neg_scores), 1),
                           torch.zeros(rows, device=device, dtype=torch.long), reduction=reduction))
        native_losses.append(loss_fn(actual[inverse * 2], actual[inverse * 2 + 1], True))
    ref_loss, native_loss = sum(ref_losses), sum(native_losses)
    ref_loss.backward()
    native_loss.backward()
    checks = {}

    def check(name, a, b, atol=2e-5, rtol=2e-4):
        checks[name] = dict(passed=bool(torch.allclose(a, b, atol=atol, rtol=rtol)),
                            max_abs=float((a - b).abs().max()))

    for i, exp in enumerate(expected):
        check('scores_' + str(i), actual[i][:rows], exp)
    padding = chunks * per_chunk - rows
    # Zero-padded queries add a constant to SUM loss; MEAN also rescales gradients.
    scale = rows / (rows + padding) if reduction == 'mean' else 1.
    check('entity_gradient', embeddings.grad, reference.grad * scale)
    if kind != 'dot':
        check('relation_gradient', decoder.relations.grad, relation.grad * scale)
        check('inverse_relation_gradient', decoder.inverse_relations.grad, inv_relation.grad * scale)
    grad = embeddings.grad.clone()
    batch.accumulateGradients(.1)
    check('sparse_state_increment', batch.node_state_update, grad.square())
    check('sparse_state', state_alias, old_state + grad.square())
    check('sparse_update', batch.node_gradients, -.1 * grad / ((old_state + grad.square()).sqrt() + 1e-10))
    raw_loss_delta = float(native_loss - ref_loss)
    expected_loss = ref_loss * scale
    constant = padding * math.log(negatives + 1) * len(ref_losses)
    expected_loss = expected_loss + constant / (rows + padding) if reduction == 'mean' else expected_loss + constant
    check('loss_with_padding_accounted', native_loss, expected_loss, atol=2e-4)
    return dict(kind=kind, width=width, rows=rows, chunks=chunks, seed=seed, reduction=reduction,
                padded_rows=padding, native_loss=float(native_loss), real_rows_loss=float(ref_loss),
                raw_loss_delta=raw_loss_delta, gradient_scale_vs_real_rows=scale, checks=checks,
                passed=all(item['passed'] for item in checks.values()))


def sampler_case(gege, torch, device):
    torch.manual_seed(712)
    nodes, rows, chunks, negatives = 401, 101, 7, 20
    # Unique endpoints make every sampled self-positive independently identifiable.
    edges = torch.stack((torch.arange(rows, device=device), torch.zeros(rows, dtype=torch.long, device=device),
                         torch.arange(rows, device=device) + rows), 1)
    graph = gege.data.GegeGraph(edges, edges, nodes)
    sampler = gege.data.samplers.CorruptNodeNegativeSampler(chunks, negatives, .5, False, 'DEG')
    result = []
    for inverse in (False, True):
        ids, filt = sampler.getNegatives(graph, edges, inverse)
        size = math.ceil(rows / chunks)
        actual = sorted(map(tuple, filt.cpu().tolist()))
        expected = []
        for row in range(rows):
            target = int(edges[row, 0 if inverse else -1])
            for col in range(negatives // 2):
                if int(ids[row // size, col]) == target:
                    expected.append((row, col))
        result.append(dict(inverse=inverse, passed=actual == expected, expected=len(expected), actual=len(actual)))
    return result


def dense_case(gege, torch, device):
    decoder = gege.nn.decoders.edge.DistMult(3, 10, True, device, torch.float32, 'CORRUPT_NODE')
    model = gege.nn.Model(gege.nn.encoders.GeneralEncoder([[gege.nn.layers.EmbeddingLayer(10, device)]]), decoder)
    native = gege.nn.AdagradOptimizer(model.named_parameters(), .1, 1e-10, 0., 0., 0.)
    pairs = [(decoder.relations, decoder.relations.detach().clone().requires_grad_()),
             (decoder.inverse_relations, decoder.inverse_relations.detach().clone().requires_grad_())]
    ref = torch.optim.Adagrad([b for a, b in pairs], lr=.1, eps=1e-10)
    for step in range(10):
        for a, b in pairs:
            grad = torch.randn_like(a)
            grad[0] = 0
            a.grad, b.grad = grad.clone(), grad.clone()
        native.step()
        ref.step()
        # Python owns these gradient tensors. Clearing them in GE2's OpenMP
        # binding while Python holds the GIL can deadlock; train_batch releases it.
        for a, b in pairs:
            a.grad = None
        ref.zero_grad()
    error = max(float((a - b).abs().max()) for a, b in pairs)
    return dict(steps=10, max_abs=error, passed=error < 1e-6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    import gege
    import torch
    faulthandler.dump_traceback_later(60, repeat=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    began = time.time()
    device = torch.device(args.device)
    report = dict(scope=__doc__, host=socket.gethostname(), torch=torch.__version__,
                  gege_module=gege.__file__, library_sha256=sha256(Path(gege.__file__).parent / 'libge2.so'),
                  script_sha256=sha256(__file__), device=str(device), cases=[])
    for kind in ('dot', 'distmult', 'complex'):
        for width in (10, 80, 100):
            for rows, chunks in ((12, 3), (13, 3), (3, 5)):
                for reduction in ('sum', 'mean'):
                    for seed in (71, 193):
                        row = run_case(gege, torch, device, kind, width, rows, chunks, reduction, seed)
                        report['cases'].append(row)
    print('Completed forward/loss/sparse tests', len(report['cases']), flush=True)
    report['sampler'] = sampler_case(gege, torch, device)
    report['dense_adagrad'] = dense_case(gege, torch, device)
    report['passed'] = all(r['passed'] for r in report['cases'] + report['sampler']) and report['dense_adagrad']['passed']
    report['elapsed_s'] = time.time() - began
    faulthandler.cancel_dump_traceback_later()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(passed=report['passed'], cases=len(report['cases']),
                         failures=[r for r in report['cases'] if not r['passed']],
                         sampler=report['sampler'], dense_adagrad=report['dense_adagrad']), indent=2))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
