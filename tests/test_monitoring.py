import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from store import Store
from monitoring import psi,summarize

def row(i,label=0,prediction=0,source="replay"):
    return {"id":str(i),"event_time":f"2020-01-01T00:{i%60:02d}:00","amount":10.0,"hour":0,"category":"misc_net",
            "probability":.9 if prediction else .01,"prediction":prediction,"label":label,"source":source,
            "latency_ms":2.0,"features":json.dumps({"amt":10,"hour":0,"category":"misc_net"})}

class MonitoringTests(unittest.TestCase):
    def test_psi_has_zero_for_identical_and_detects_large_shift(self):
        self.assertAlmostEqual(psi([10,20],[10,20]),0)
        self.assertGreater(psi([1000,1],[1,1000]),.2)
        self.assertTrue(psi([100,0],[0,100])>0)
    def test_metrics_confusion_and_unlabeled_exclusion(self):
        rows=[row(0,1,1),row(1,0,1),row(2,1,0),row(3,0,0),row(4,None,1)]
        for r in rows:r["features"]=json.loads(r["features"])
        d=summarize(rows,{},.3482)
        self.assertEqual(d["labeled"],4)
        self.assertEqual(d["confusion"],{"tp":1,"fp":1,"fn":1,"tn":1})
        self.assertEqual(d["precision"],.5)
        self.assertEqual(d["recall"],.5)
        self.assertEqual(d["flagged"],3)
        self.assertIsNone(summarize([dict(rows[0],label=None)],{},.3)["pr_auc"])
    def test_sqlite_persistence_idempotence_and_separate_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"scores.db"
            a=Store(path); a.add([row(1),row(1),row(2,source="live")])
            self.assertEqual(len(Store(path).recent("replay")),1)
            self.assertEqual(len(a.recent("live")),1)
            self.assertEqual(a.reserve_demo(70,50),(0,50))
            self.assertEqual(Store(path).reserve_demo(70,50),(50,70))
            self.assertEqual(a.reserve_demo(70,50),(70,70))
    def test_journal_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            a=Store(Path(directory)/"scores.db")
            a.add([row(i) for i in range(5005)])
            retained=a.recent("replay")
            self.assertEqual(len(retained),5000)
            self.assertEqual([int(r["id"]) for r in retained],list(range(5,5005)))
if __name__=="__main__":unittest.main()
