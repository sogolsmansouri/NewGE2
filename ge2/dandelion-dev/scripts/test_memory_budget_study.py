import unittest
from pathlib import Path
import tempfile
import os
import json
from types import SimpleNamespace
from unittest.mock import patch
from memory_budget_study import limits, estimate, make_plan, policy, configuration, check_log, selection_points, schedule_evidence, certified_schedule


class MemoryBudgetStudyTest(unittest.TestCase):
    def test_cap_reserves_external_allocations(self):
        cap = limits(24576)
        self.assertEqual(cap['total_limit_mib'], 22118)
        self.assertEqual(cap['allocator_limit_mib'], 21606)
        for args in [(0,), (1,), (16384, 1.1), (16384, .9, -1)]:
            with self.assertRaises(ValueError):
                limits(*args)

    def test_plan_budget_order_and_scope(self):
        plan = make_plan(49140)
        self.assertEqual(plan['budgets_mib'], [16384,24576,32768,40960,49140])
        self.assertEqual(len(plan['rows']), 10)
        self.assertEqual(plan['measurement_epochs'], 10)
        self.assertEqual(plan['hidden_counts'], list(range(7)))
        self.assertEqual(make_plan(24576)['budgets_mib'], [16384,24576])
        for row in plan['rows']:
            self.assertEqual(len({c['case'] for c in row['candidates']}), len(row['candidates']))
            self.assertTrue(all(c['k']==4+c['hidden'] for c in row['candidates']))

    def test_real_frame_rounding_not_rounded_gib(self):
        point = estimate('tw', 16, 3, 24576)
        self.assertEqual(point['frame_bytes'], 800*((41652230+15)//16))
        self.assertGreater(estimate('tw',16,6,24576)['estimated_bytes'], point['estimated_bytes'])

    def test_old_training_flags_cannot_leak(self):
        old = {'GEGE_BASELINE_TRAINING_SEMANTICS':'0',
               'GEGE_BATCHED_NEGATIVE_PLAN_BATCHES':'8',
               'GEGE_SOFTMAX_NEGATIVE_MASS_SCALE':'8',
               'GEGE_SOFTMAX_NEGATIVE_LOG_MASS_BIAS':'2',
               'GEGE_BOUNDED_STATE_ORDER_FILE':'old', 'OTHER':'bad'}
        flags = policy(old,'fb',32,0,86054151,limits(24576))
        self.assertEqual(flags['GEGE_BASELINE_TRAINING_SEMANTICS'],'1')
        self.assertEqual(flags['GEGE_BATCHED_NEGATIVE_PLAN_BATCHES'],'0')
        self.assertEqual(flags['GEGE_SOFTMAX_NEGATIVE_MASS_SCALE'],'1')
        self.assertEqual(flags['GEGE_FRAME_CACHE_DELAYED_STALE_WRITEBACK'],'0')
        self.assertEqual(flags['GEGE_SINGLE_GPU_ASYNC_ADMIT_PRELOAD'],'0')
        self.assertEqual(flags['GEGE_BOUNDED_COVER_EPOCH_RELABEL'],'1')
        self.assertNotIn('OTHER',flags)
        self.assertNotIn('GEGE_BOUNDED_STATE_ORDER_FILE',flags)
        self.assertNotIn('GEGE_SOFTMAX_NEGATIVE_LOG_MASS_BIAS',flags)

    def test_graph_policy_fixed_across_arms(self):
        template={'storage':{'embeddings':{'options':{}}},
                  'training':{'negative_sampling':{}},'evaluation':{}}
        for workload, expected in [('tw', True),('fb',False)]:
            cfg = configuration(template,{},'model',workload,16,5)
            self.assertEqual(cfg['storage']['prefetch'],expected)
            self.assertEqual(cfg['training']['batch_size'],50000)
            self.assertEqual(cfg['evaluation']['epochs_per_eval'],0)
            self.assertFalse(cfg['training']['save_model'])
        self.assertNotIn('prefetch',template['storage'])

    def test_incomplete_epoch_cannot_pass(self):
        with self.assertRaises(ValueError):
            check_log('Epoch Runtime: 100ms',2,100,0,16,2048)

    def test_partition_candidates_come_from_budget(self):
        points=selection_points(49140)
        self.assertEqual(len(points),len(set(points)))
        self.assertIn(('tw',49140,8,0),points)
        self.assertIn(('tw',49140,7,0),points)
        self.assertIn(('tw',49140,9,0),points)
        for w in ('tw','fb'):
            for b in (16384,24576,32768,40960,49140):
                for h in range(7):
                    self.assertTrue(any((a,c,d)==(w,b,h) for a,c,p,d in points))

    def test_peak_and_edge_contract(self):
        text='''[memory-budget] device=0 allocator_limit_bytes=10000000
deferred backing allocation device=cuda:0 visible_rows=512 physical_rows=896 dim=100 pinned=true hidden_frames=3
deferred backing allocation device=cuda:0 visible_rows=512 physical_rows=896 dim=100 pinned=true hidden_frames=3
Epoch Runtime: 100ms
Edges processed: [100/100], 100.00%
[memory-budget-peak] epoch=1 device=0 allocated_peak_bytes=2000000 reserved_peak_bytes=4000000
Epoch Runtime: 90ms
Edges processed: [100/100], 100.00%
[memory-budget-peak] epoch=2 device=0 allocated_peak_bytes=3000000 reserved_peak_bytes=6000000
'''
        result=check_log(text,2,100,3,16,2048)
        self.assertAlmostEqual(result['average_epoch_s'],.095)
        self.assertAlmostEqual(result['steady_epoch_s'],.09)
        for invalid in (text.replace('epoch=2','epoch=1'),
                        text.replace('reserved_peak_bytes=6000000','reserved_peak_bytes=11000000'),
                        text.replace('[100/100]','[99/100]'),
                        text.replace('hidden_frames=3','hidden_frames=2')):
            with self.assertRaises(ValueError):
                check_log(invalid,2,100,3,16,2048)

    def test_physical_repartition_preserves_training_rows(self):
        import numpy as np
        from memory_budget_study import prepare_view
        from run_arc_paper_case import digest
        # CI may supply the frozen helpers in a separate directory.
        import prepare_ge2_partitioned_view
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=root/'source'
            (source/'edges').mkdir(parents=True)
            rows=np.array([[0,7],[7,0],[1,3],[4,6],[4,6]],dtype='<i4')
            rows.tofile(source/'edges/train_edges.bin')
            expected=digest(source/'edges/train_edges.bin')
            spec=dict(nodes=8,edges=5,columns=2,relations=1)
            with patch('memory_budget_study.shutil.disk_usage') as disk:
                disk.return_value.free=100*2**30
                data=prepare_view(source,expected,spec,4,root/'view')
            output=np.fromfile(root/'view/edges/train_edges.bin',dtype='<i4').reshape(-1,2)
            self.assertEqual(sorted(map(tuple,output)),sorted(map(tuple,rows)))
            self.assertEqual(data['num_train'],5)
            self.assertEqual(data['num_test'],-1)
            self.assertEqual(prepare_view(source,expected,spec,4,root/'view'),data)
            (root/'view/edges/train_edges.bin').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                prepare_view(source,expected,spec,4,root/'view')

    def test_schedule_evidence(self):
        row='Generating bounded GREEDY_COVER ordering states=20 transitions=19 total_buckets=256 max_admits=3 transition_admits=57 edge_total=8192\n'
        info=schedule_evidence(row*2,2,8192,16)
        self.assertTrue(info['state_count_certified_minimum'])
        self.assertEqual(info['admissions'],57)
        for bad in (row,row+row.replace('states=20','states=21'),row.replace('total_buckets=256','total_buckets=255')*2):
            with self.assertRaises(ValueError):
                schedule_evidence(bad,2,8192,16)

    def test_schedule_certificate_required_and_hash_checked(self):
        from dataclasses import asdict
        from plan_pipege_cover import plan_cover,write_schedule
        states,summary,_=plan_cover(5,4,solver='optimal',restarts=1)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            folder=root/'p5'
            folder.mkdir()
            write_schedule(folder/'states.txt',states)
            evidence=asdict(summary)
            (folder/'cover.json').write_text(json.dumps(evidence))
            self.assertEqual(certified_schedule(root,5)[1].state_count,3)
            for field,value in [('state_count_optimal',False),('overlap_optimality','not_proven'),
                                ('maximum_overlap',999),('schedule_sha256','changed')]:
                changed=json.loads(json.dumps(evidence))
                changed['optimality'][field]=value
                (folder/'cover.json').write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    certified_schedule(root,5)
            (folder/'cover.json').write_text(json.dumps(evidence))
            write_schedule(folder/'states.txt',list(reversed(states)))
            with self.assertRaises(ValueError):
                certified_schedule(root,5)

    def test_all_requested_partitions_are_in_certificate_plan(self):
        from certify_memory_schedules import required_geometries
        plan=make_plan(49140)
        groups=required_geometries(plan)
        self.assertEqual(set(groups),{point[2] for point in plan['selection_points']})
        self.assertEqual(sum(map(len,groups.values())),len(plan['selection_points']))
        self.assertIn(67,groups)
        self.assertTrue(all(r['k']==4+r['h'] for rows in groups.values() for r in rows))

    @unittest.skipUnless(os.environ.get('PIPEGE_GPU_INTEGRATION'),'opt-in local GPU integration')
    def test_ten_epoch_sweep_and_cleanup(self):
        import yaml
        import memory_budget_study as study
        from arc_job_support import write_json
        from run_arc_paper_case import digest
        source=Path(os.environ['PIPEGE_TEST_DATA'])
        metadata=yaml.safe_load((source/'dataset.yaml').read_text())
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            sources=root/'sources.json'
            write_json(sources,{'tw':dict(path=str(source),train_sha256=digest(source/'edges/train_edges.bin'))})
            args=SimpleNamespace(output=root/'sweep',sources=sources,
                references=Path(os.environ['PIPEGE_TEST_REFERENCES']),
                gate=Path(os.environ['PIPEGE_TEST_GATE']),
                gege=Path(os.environ['PIPEGE_TEST_GEGE']),
                build=Path(os.environ['PIPEGE_TEST_BUILD']),env=Path(os.environ['PIPEGE_TEST_ENV']),
                gpu='0',seconds=6000,case=json.dumps(['tw',16384,8,2]),commit='synthetic-integration',
                epochs=10,power_w=350,schedule_root=Path(os.environ['PIPEGE_TEST_SCHEDULES']))
            spec=dict(study.WORKLOADS['tw'],nodes=metadata['num_nodes'],edges=metadata['num_train'])
            with patch.dict(study.WORKLOADS,tw=spec):
                study.sweep(args)
            case=root/'sweep/tw_m16384_p8_q4_h2'
            result=json.loads((case/'result.json').read_text())
            self.assertEqual(result['status'],'pass')
            self.assertEqual(len(result['epoch_times_s']),10)
            self.assertFalse(result['paper_ready'])
            self.assertTrue((case/'data_manifest.json').is_file())
            self.assertFalse((case/'model').exists())
            self.assertFalse((root/'sweep/data/tw_p8').exists())


if __name__=='__main__':
    unittest.main()
