import copy
from pathlib import Path
import unittest

from run_arc_pipege_best import configure, evaluation_check, schedule_check, training_check


class Contracts(unittest.TestCase):
    def setUp(self):
        self.spec = dict(graph='tw', model='dot', p=16, q=4, hidden=3, nodes=41652230,
                         states=20, edges=1321528663, relations=1, width=100, epochs=10, eval_sha='query')
        self.config = dict(model=dict(encoder=dict(layers=[[dict(output_dim=100, bias=False)]]),
                                     decoder=dict(type='DISTMULT'),
                                     sparse_optimizer=dict(type='ADAGRAD', options=dict(learning_rate=.1))),
                           storage=dict(embeddings=dict(options=dict(num_partitions=16, buffer_capacity=4)), prefetch=True),
                           training=dict(batch_size=50000, negative_sampling_method='RNS',
                                         negative_sampling=dict(num_chunks=50, negatives_per_positive=1000, degree_fraction=.5)),
                           evaluation={})
        self.data = dict(num_nodes=41652230, num_relations=1, num_train=1321528663)

    def test_gate_is_distinct_from_final(self):
        gate = configure(self.config, self.data, Path('/gate'), self.spec, True)
        final = configure(self.config, self.data, Path('/final'), self.spec, False)
        self.assertEqual(gate['training']['num_epochs'], 2)
        self.assertFalse(gate['training']['save_model'])
        self.assertEqual(final['training']['num_epochs'], 10)
        self.assertTrue(final['training']['save_model'])
        self.assertEqual(final['evaluation']['epochs_per_eval'], 1000)
        self.assertNotIn('model_dir', self.config['storage'])

    def test_reject_bias_and_wrong_workload(self):
        changed = copy.deepcopy(self.config)
        changed['model']['encoder']['layers'][0][0]['bias'] = True
        with self.assertRaises(ValueError):
            configure(changed, self.data, Path('/x'), self.spec, False)
        with self.assertRaises(ValueError):
            configure(self.config, dict(self.data, num_train=1468365182), Path('/x'), self.spec, False)

    def test_cover(self):
        spec = dict(p=4, q=3, states=3)
        text = 'state=[0,1,2]\nstate=[0,1,3]\nstate=[1,2,3]\n'
        schedule_check(text, spec)
        with self.assertRaises(ValueError):
            schedule_check(text.replace('[1,2,3]', '[0,1,2]'), spec)
        with self.assertRaises(ValueError):
            schedule_check(text.replace('[1,2,3]', '[1,2,4]'), spec)

    def training_log(self):
        rows = (self.spec['nodes']+15)//16
        log = f'deferred backing allocation device=cuda:0 visible_rows={4*rows} physical_rows={7*rows} dim=100 pinned=true hidden_frames=3\n'*2
        log += 'Generating bounded GREEDY_COVER ordering states=20\n'
        log += '[manual_dot_rns] enabled=1\nUsing bucket-streaming LP path\n'
        for i in (1, 2):
            log += f'Edges processed: [1321528663/1321528663], 100.00%\nFinished training epoch {i}\nEpoch Runtime: 183000ms\n'
        return log

    def test_training_requires_full_edges_and_correct_frames(self):
        text = self.training_log()
        self.assertEqual(training_check(text, self.spec, 2), [183., 183.])
        for changed in (text.replace('100.00%', '99.99%'), text.replace('hidden_frames=3', 'hidden_frames=6'),
                        text.replace('Finished training epoch 2', 'Finished training epoch 3'), text+'CUDA error'):
            with self.assertRaises(ValueError):
                training_check(changed, self.spec, 2)

    def test_evaluation_requires_both_directions_and_pinned_subset(self):
        value = dict(num_ranks=20000, filtered=True, eval_edges_sha256='query', tie_policy='pessimistic',
                     tf32=False, mrr=.14, hits_at_10=.33)
        evaluation_check(value, self.spec)
        for key, wrong in [('num_ranks',10000),('tf32',True),('mrr',float('nan')),
                           ('eval_edges_sha256','other'),('filtered',False)]:
            with self.assertRaises(ValueError):
                evaluation_check(dict(value, **{key:wrong}), self.spec)


if __name__ == '__main__':
    unittest.main()
