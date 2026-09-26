import sqlite3
from pathlib import Path
import tempfile
import unittest
from analyze_ge2_motivation import device_accounting,phase_segments,counter_phases,trace_summary,PHASES


class AccountingTests(unittest.TestCase):
    def test_counter_closure_and_missing_fields(self):
        metrics={name:dict(elapsed_ms=1) for names in PHASES.values() for name in names}
        value=counter_phases(dict(official_epoch_ms=100,metrics=metrics))
        self.assertAlmostEqual(sum(x['seconds'] for x in value['phases']),.1)
        self.assertIsNone(value['idle_seconds'])
        with self.assertRaises(KeyError):
            counter_phases(dict(official_epoch_ms=100,metrics={}))
        with self.assertRaises(ValueError):
            counter_phases(dict(official_epoch_ms=1,metrics=metrics))
        with self.assertRaises(ValueError):
            counter_phases(dict(official_epoch_ms=0,metrics=metrics))

    def test_disjoint_host_and_overlap_rejection(self):
        rows=[(0,2,'dataloader.negative_sample'),(2,5,'trainer.model_train')]
        self.assertEqual(phase_segments(rows,(0,6))[-1],(5,6,'Graph construction/control/other'))
        with self.assertRaises(ValueError):
            phase_segments(rows+[(1,3,'active_edges.shuffle')],(0,6))

    def test_cuda_union_includes_memset_and_clips(self):
        c=sqlite3.connect(':memory:')
        c.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start,end)')
        c.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start,end,copyKind,bytes)')
        c.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET(start,end)')
        c.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?)',[(0,4),(2,5)])
        c.execute('INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(3,7,1,100)')
        c.execute('INSERT INTO CUPTI_ACTIVITY_KIND_MEMSET VALUES(8,9)')
        out=device_accounting(c,(1,10),[(1,6,'Sampling'),(6,10,'Other')])
        self.assertAlmostEqual(sum(out['activity_seconds'].values()),9e-9)
        self.assertAlmostEqual(out['activity_seconds']['no recorded CUDA work'],2e-9)
        self.assertAlmostEqual(sum(out['gpu_inactive_by_host_phase'].values()),2e-9)
        self.assertEqual(out['copy_service_nonadditive']['H2D']['bytes'],100)
        c.close()

    def test_whole_epoch_requires_complete_markers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'trace.sqlite'
            with sqlite3.connect(path) as c:
                c.execute('CREATE TABLE NVTX_EVENTS(start,end,globalTid,textId,text)')
                c.execute('CREATE TABLE StringIds(id,value)')
                c.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start,end)')
                c.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(10,50)')
                c.execute("INSERT INTO NVTX_EVENTS VALUES(0,100,1,NULL,'ge2.profile.native_epoch')")
                c.execute("INSERT INTO NVTX_EVENTS VALUES(0,60,1,NULL,'dataloader.negative_sample')")
            with self.assertRaises(ValueError):
                trace_summary(path)
            with sqlite3.connect(path) as c:
                c.execute("INSERT INTO NVTX_EVENTS VALUES(100,120,1,NULL,'ge2.profile.epoch_finalize')")
            value=trace_summary(path)
            self.assertEqual(value['scope'],'whole_native_epoch_2')
            self.assertAlmostEqual(sum(value['host_phase_seconds'].values()),100e-9)
            self.assertAlmostEqual(sum(value['activity_seconds'].values()),100e-9)
            self.assertAlmostEqual(value['post_timer_finalize_seconds'],20e-9)
            with sqlite3.connect(path) as c:
                c.execute("INSERT INTO NVTX_EVENTS VALUES(200,300,1,NULL,'ge2.profile.native_epoch')")
            with self.assertRaises(ValueError):
                trace_summary(path)


if __name__=='__main__':
    unittest.main()
