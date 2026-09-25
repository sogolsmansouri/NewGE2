import copy
from pathlib import Path
import tempfile
import unittest
import yaml

from run_arc_pipege_best import (configure, engine_contract, evaluation_check, loss_contract,
                                normalize_dataset_metadata, preparation_cases, reused_data_contract,
                                schedule_check, training_check)


class Contracts(unittest.TestCase):
    def test_prepare_only_selected_graphs_and_reject_conflicting_views(self):
        spec = dict(graph='fb', source='/fb', query='/query', eval_sha='sha', p=32,
                    columns=3, nodes=10, relations=2, edges=100, model='complex')
        self.assertEqual(preparation_cases({'fb_complex': spec}), [spec])
        self.assertEqual(preparation_cases({'fb_complex': spec,
                         'fb_distmult': dict(spec, model='distmult')}), [spec])
        with self.assertRaisesRegex(ValueError, 'disagree'):
            preparation_cases({'fb_complex': spec, 'fb_distmult': dict(spec, p=16)})
        with self.assertRaisesRegex(ValueError, 'No cases'):
            preparation_cases({})

    def test_new_engine_requires_complete_pins_and_gate(self):
        pinned = dict(commit='a'*40, root='/engine', gradient_gate='/gate/result.json',
                      gate_template='/fixture', hashes=dict(libge2_so='b'*64, gege_train='c'*64))
        self.assertEqual(engine_contract(dict(engine=pinned)), pinned)
        for key, value in [('commit', 'latest'), ('root', 'relative'), ('hashes', {}),
                           ('gradient_gate', ''), ('gate_template', '')]:
            with self.assertRaises(ValueError):
                engine_contract(dict(engine=dict(pinned, **{key:value})))
        self.assertEqual(len(engine_contract({})['commit']), 40)

    def test_dataset_reuse_requires_prior_preparation_identity(self):
        reference = dict(commit='old', manifest_sha256='manifest')
        prepared = dict(reference, status='ready', data={x:{} for x in ('lj', 'tw', 'fb', 'wk')})
        self.assertEqual(reused_data_contract(prepared, reference), prepared['data'])
        for key, value in [('status', 'failed'), ('commit', 'other'), ('manifest_sha256', 'other'), ('data', {})]:
            with self.assertRaises(ValueError):
                reused_data_contract(dict(prepared, **{key:value}), reference)

    def test_relocated_metadata_persisted_and_idempotent(self):
        with tempfile.TemporaryDirectory() as temp:
            view = Path(temp)
            (view/'edges').mkdir()
            path = view/'dataset.yaml'
            old = dict(dataset_dir='/old/node/data/', num_nodes=17, num_train=50, split_identity='fixed')
            path.write_text(yaml.safe_dump(old))
            expected = dict(old, dataset_dir=str(view.resolve())+'/')
            self.assertEqual(normalize_dataset_metadata(view, expected), expected)
            self.assertEqual(yaml.safe_load(path.read_text()), expected)
            first = path.read_bytes()
            normalize_dataset_metadata(view, expected)
            self.assertEqual(path.read_bytes(), first)
            with self.assertRaisesRegex(ValueError, 'beyond its location'):
                normalize_dataset_metadata(view, dict(expected, num_train=51))
            self.assertEqual(path.read_bytes(), first)

    def test_relocation_does_not_mutate_symlink_target(self):
        with tempfile.TemporaryDirectory() as temp:
            view = Path(temp)/'copy'
            (view/'edges').mkdir(parents=True)
            original = Path(temp)/'original.yaml'
            original.write_text('dataset_dir: /old/\nnum_nodes: 17\n')
            (view/'dataset.yaml').symlink_to(original)
            with self.assertRaisesRegex(ValueError, 'shared dataset metadata'):
                normalize_dataset_metadata(view)
            self.assertEqual(original.read_text(), 'dataset_dir: /old/\nnum_nodes: 17\n')

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

    def test_unweighted_loss_only(self):
        loss_contract({})
        loss_contract(dict(GEGE_SOFTMAX_NEGATIVE_MASS_SCALE='1', GEGE_SOFTMAX_NEGATIVE_LOG_MASS_BIAS='0'))
        for flags in [dict(GEGE_SOFTMAX_NEGATIVE_MASS_SCALE=x) for x in ('8', 'nan', 'invalid', '0')] + [
                dict(GEGE_SOFTMAX_NEGATIVE_LOG_MASS_BIAS='2.0794415416798357')]:
            with self.assertRaises(ValueError):
                loss_contract(flags)

    def test_tw_evaluation_requires_tail_and_pinned_subset(self):
        value = dict(num_ranks=10000, report_directions='tail', filtered=True, eval_edges_sha256='query', tie_policy='pessimistic',
                     tf32=False, mrr=.14, hits_at_10=.33)
        evaluation_check(value, self.spec)
        for key, wrong in [('num_ranks',20000),('report_directions','both'),('tf32',True),('mrr',float('nan')),
                           ('eval_edges_sha256','other'),('filtered',False)]:
            with self.assertRaises(ValueError):
                evaluation_check(dict(value, **{key:wrong}), self.spec)
        for graph in ('lj', 'fb', 'wk'):
            spec = dict(self.spec, graph=graph)
            evaluation_check(dict(value, report_directions='both', num_ranks=20000), spec)
            with self.assertRaises(ValueError):
                evaluation_check(value, spec)


if __name__ == '__main__':
    unittest.main()
