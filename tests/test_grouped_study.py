"""Grouped development selection, immutable evidence and real training boundaries."""

import contextlib
from copy import deepcopy
import io
import json
import fcntl
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from grace_rul import grouped_study as study
from grace_rul.config import COLUMNS, SUBSETS
from grace_rul.pooled import load_pooled_training, pooled_config, prepare_pooled_run, sha, write_json
from grace_rul.training import fit_pipeline


def fixture_data(root):
    (root/"data").mkdir()
    for subset in SUBSETS:
        rows = [[u,t,0.,0.,100.,*[j+.01*t+.03*u for j in range(21)]]
                for u in range(1,21) for t in range(1,61)]
        pd.DataFrame(rows,columns=COLUMNS).to_csv(root/f"data/train_{subset}.txt",sep=" ",header=False,index=False)


def protocol():
    return json.loads((study.ROOT/"configs/grouped_study.json").read_text())


def fake_records(plan, *, regularized=1.):
    """Synthetic statistics exercise selection math; never used as model evidence."""
    rows=[]
    for candidate in plan["candidates"]:
        value=2. if candidate["name"]==plan["protocol"]["reference"] else regularized
        for fold in plan["folds"]:
            for seed in plan["protocol"]["training_seeds"]:
                for family,severity,replicate in study.assessment_cases(plan["protocol"]):
                    for unit in fold["assessment"]:
                        rows.append({"candidate":candidate["name"],"fold":fold["fold"],"seed":seed,
                            "family":family,"severity":severity,"replicate":replicate,"unit":unit,
                            "source_subset":plan["engine_sources"][str(unit)],"mse":value**2,
                            "mae":value,"nasa":value,"prefixes":2})
    return pd.DataFrame(rows)


class GroupedStudyContracts(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        fixture_data(self.root)
        self.config={**pooled_config(),"device":"cpu"}
        self.boundary=self.root/"artifacts/old/pooled_plan.json"
        prepare_pooled_run(self.root,self.config,self.boundary.parent)
        self.output=self.root/"artifacts/study"
        self.protocol=protocol()
        # Small synthetic fault matrix keeps contract tests fast. The actual
        # checked-in scientific protocol retains every family/level/replicate.
        self.protocol.update(fault_families=["mixed","stuck"],fault_severities=[.2],corruption_replicates=1)
        self.plan=study.create_study(self.root,self.output,self.boundary,self.protocol,config=self.config,device="cpu")

    def tearDown(self):
        self.temp.cleanup()

    def test_outer_inner_and_historical_groups_are_disjoint_and_target_independent(self):
        plan,frame=study.read_study(self.root,self.output)
        forbidden=set(plan["partitions"]["calibration"])|set(plan["partitions"]["test"])
        assessment=[]
        for fold in plan["folds"]:
            a,b,c=[set(fold[k]) for k in ["fit","inner_validation","assessment"]]
            self.assertFalse(a&b or a&c or b&c)
            self.assertFalse((a|b|c)&forbidden)
            self.assertEqual(a|b|c,set(plan["study_units"]))
            assessment+=fold["assessment"]
        self.assertEqual(sorted(assessment),plan["study_units"])
        changed=frame.copy();changed["cycle"]*=19
        for column in COLUMNS[2:]:changed[column]=0.
        self.assertEqual(study.grouped_folds(changed,plan["study_units"],self.protocol),plan["folds"])

    def test_plan_cannot_change_or_reuse_previous_run(self):
        with self.assertRaises(FileExistsError):
            study.create_study(self.root,self.output,self.boundary,self.protocol)
        changed=deepcopy(self.plan);changed["protocol"]["clean_weight"]=.7
        write_json(self.output/"study_plan.json",changed)
        with self.assertRaisesRegex(ValueError,"preregistration"):
            study.read_study(self.root,self.output)

    def test_changed_code_training_data_or_historical_boundaries_reject_resume(self):
        with patch("grace_rul.grouped_study.implementation_hashes",return_value={}):
            with self.assertRaisesRegex(ValueError,"Implementation changed"):
                study.read_study(self.root,self.output)
        original=self.boundary.read_bytes();self.boundary.write_bytes(original+b"\n")
        with self.assertRaisesRegex(ValueError,"Historical boundary"):
            study.read_study(self.root,self.output)
        self.boundary.write_bytes(original)
        with (self.root/"data/train_FD001.txt").open("a") as stream:stream.write("\n")
        with self.assertRaisesRegex(ValueError,"training data changed"):
            study.read_study(self.root,self.output)

    def test_candidate_cannot_override_splits_ids_or_test_controls(self):
        for key in ["split_seed","seeds","pooled_evaluate_official","unit","engine_id","calibration_policy"]:
            invalid=deepcopy(self.protocol);invalid["candidates"][0]["overrides"][key]=1
            with self.assertRaisesRegex(ValueError,"global training/model"):
                study.validate_protocol(invalid)

    def test_scientific_study_cannot_reduce_seeds_or_reassign_historical_engines(self):
        invalid=deepcopy(self.protocol);invalid["training_seeds"]=[42]
        with self.assertRaisesRegex(ValueError,"predeclared training seeds"):
            study.validate_protocol(invalid)
        changed={**self.config,"split_seed":43}
        with self.assertRaisesRegex(ValueError,"historical calibration/test"):
            study.create_study(self.root,self.root/"other",self.boundary,self.protocol,config=changed,device="cpu")

    def test_one_active_writer_per_study(self):
        with (self.output/"study.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError,"active writer"):
                study.run_study(self.root,self.output)

    def test_read_boundary_blocks_original_tests_results_and_old_weights(self):
        old=self.root/"artifacts/old/grace_rul_ALL.pt";old.write_text("fixture")
        for path in [self.root/"data/test_FD001.txt",self.root/"data/RUL_FD001.txt",old]:
            with study.selection_read_boundary(self.root,self.output,self.boundary) as access:
                with self.assertRaises(PermissionError):path.read_bytes()
            self.assertEqual(len(access["blocked"]),1)
        with study.selection_read_boundary(self.root,self.output,self.boundary) as access:
            (self.root/"data/train_FD001.txt").read_bytes()
            self.boundary.read_text()
        self.assertEqual(access["training_files"],{"train_FD001.txt":1})

    def test_population_score_uses_all_engines_contexts_sources_and_seeds(self):
        records=fake_records(self.plan)
        summary,_=study.summarize(self.plan,records)
        self.assertEqual(summary["selected_candidate"],"shared_prefix_regularized")
        for result in summary["candidates"]:
            self.assertEqual(result["mean_score"],2. if result["candidate"]=="shared_reference" else 1.)
            self.assertEqual(result["between_seed_std"],0.)
        partial=records.drop(records.index[0])
        with self.assertRaisesRegex(ValueError,"Incomplete"):
            study.summarize(self.plan,partial)
        duplicate=pd.concat([records,records.iloc[:1]])
        with self.assertRaisesRegex(ValueError,"Repeated"):
            study.summarize(self.plan,duplicate)
        wrong=records.copy();wrong.loc[0,"seed"]=987
        with self.assertRaisesRegex(ValueError,"undeclared"):
            study.summarize(self.plan,wrong)

    def test_clean_guard_rejects_one_seed_source_regression_despite_good_fault_score(self):
        records=fake_records(self.plan,regularized=.1)
        bad=(records.candidate.eq("shared_prefix_regularized")&records.family.eq("clean")&
             records.seed.eq(self.protocol["training_seeds"][0])&records.source_subset.eq("FD004"))
        records.loc[bad,["mse","nasa"]]=[25.,5.]
        summary,_=study.summarize(self.plan,records)
        self.assertEqual(summary["selected_candidate"],"shared_reference")
        self.assertFalse(summary["candidates"][1]["eligible"])

    def test_engine_statistics_do_not_weight_long_engines_more_heavily(self):
        from types import SimpleNamespace
        dataset=SimpleNamespace(engines=[{"unit":1,"cycles":[1,2,3]},{"unit":2,"cycles":[1]}],
            samples=[(0,0),(0,1),(0,2),(1,0)],source_by_unit={1:"FD001",2:"FD001"})
        records=pd.DataFrame({"unit":[1,1,1,2],"cycle":[1,2,3,1],"prediction":[0.,0.,0.,4.],"target":[0.]*4})
        errors=study.engine_errors(records,dataset)
        self.assertEqual(errors.mse.mean(),8.)
        self.assertEqual(float(((records.prediction-records.target)**2).mean()),4.)
        with self.assertRaisesRegex(ValueError,"omit or duplicate"):
            study.engine_errors(records.iloc[:-1],dataset)

    def complete_fake_cells(self):
        all_records=fake_records(self.plan)
        for candidate in self.plan["candidates"]:
            for fold in self.plan["folds"]:
                for seed in self.plan["protocol"]["training_seeds"]:
                    d=study._cell_path(self.output,candidate["name"],fold["fold"],seed);d.mkdir(parents=True)
                    for name in study.CELL_FILES:(d/name).write_text("synthetic contract fixture\n")
                    rows=all_records[(all_records.candidate==candidate["name"])&all_records.fold.eq(fold["fold"])&all_records.seed.eq(seed)]
                    rows.to_csv(d/"engine_errors.csv",index=False)
                    write_json(d/"result.json",{"status":"complete","deployable":False,
                        "study_identity_sha256":self.plan["identity_sha256"],"candidate":candidate["name"],"fold":fold["fold"],"seed":seed,
                        "fit_units":fold["fit"],"inner_selection_units":fold["inner_validation"],"assessment_units":fold["assessment"],
                        "output_sha256":{name:sha(d/name) for name in study.CELL_FILES}})
        write_json(self.output/"study_status.json",{"status":"complete","identity_sha256":self.plan["identity_sha256"]})
        write_json(self.output/"file_access_audit.json",{"blocked":[],"training_files":{k:1 for k in self.plan["source_hashes"]}})

    def test_freeze_requires_complete_evidence_and_preserves_predeclared_final_seed(self):
        with self.assertRaises((FileNotFoundError,ValueError)):
            study.freeze_recipe(self.root,self.output)
        self.complete_fake_cells()
        frozen=study.freeze_recipe(self.root,self.output)
        self.assertEqual(frozen["config"]["seeds"],[self.protocol["final_seed"]])
        self.assertFalse(frozen["selection_uses_test_metrics"])
        self.assertFalse(frozen["independent_confirmatory_evidence"])
        self.assertEqual(study.freeze_recipe(self.root,self.output),frozen)
        d=next((self.output/"cells").glob("*/*/engine_errors.csv"))
        with d.open("a") as stream:stream.write("\n")
        with self.assertRaisesRegex(ValueError,"evidence changed"):
            study.freeze_recipe(self.root,self.output)

    def test_final_refit_requires_frozen_recipe_and_protects_completed_outputs(self):
        self.complete_fake_cells();study.freeze_recipe(self.root,self.output)
        final=self.root/"artifacts/final"
        def fake_final(root,config,output):
            self.assertEqual(config["seeds"],[self.protocol["final_seed"]])
            self.assertEqual(json.loads((output/"pooled_plan.json").read_text())["partitions"],self.plan["partitions"])
            self.assertTrue((output/"frozen_study_reference.json").is_file())
            return "synthetic final contract",{}
        with patch("grace_rul.grouped_study.run_pooled_experiment",side_effect=fake_final):
            study.final_experiment(self.root,self.output,final)
        (final/"grace_rul_ALL.pt").write_text("fixture")
        with self.assertRaises(FileExistsError):study.final_experiment(self.root,self.output,final)


class RealGroupedExecution(unittest.TestCase):
    def test_real_training_never_receives_outer_or_historical_held_engines(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);fixture_data(root)
            config={**pooled_config(quick=True),"device":"cpu","maximum_batches":1}
            boundary=root/"artifacts/old/pooled_plan.json"
            prepare_pooled_run(root,config,boundary.parent)
            output=root/"artifacts/cv"
            plan=study.create_study(root,output,boundary,protocol(),config=config,quick=True,device="cpu")
            calls=[];current={}
            original=study.run_cell
            def cell(*args):
                current["fold"]=args[4]
                return original(*args)
            def train(fit,inner,*args,**kwargs):
                fold=current["fold"]
                forbidden=set(fold["assessment"])|set(plan["partitions"]["calibration"])|set(plan["partitions"]["test"])
                self.assertFalse((set(fit.unit)|set(inner.unit))&forbidden)
                self.assertFalse(set(fit.unit)&set(inner.unit))
                calls.append((set(fit.unit),set(inner.unit)))
                model,scaler,history,score=fit_pipeline(fit,inner,*args,**kwargs)
                self.assertEqual(set(scaler.state["fit_units"]),set(fit.unit))
                return model,scaler,history,score
            with contextlib.redirect_stdout(io.StringIO()),patch("grace_rul.grouped_study.run_cell",side_effect=cell), \
                    patch("grace_rul.grouped_study.fit_pipeline",side_effect=train), \
                    patch("grace_rul.training.fit_pipeline",side_effect=train):
                summary=study.run_study(root,output)
            self.assertIn(summary["selected_candidate"],[c["name"] for c in plan["candidates"]])
            self.assertEqual(len(calls),len(plan["candidates"])*len(plan["folds"])*len(plan["protocol"]["training_seeds"])*(1+config["oof_folds"]))
            access=json.loads((output/"file_access_audit.json").read_text())
            self.assertEqual(access["blocked"],[])
            self.assertEqual(set(access["training_files"]),{f"train_{s}.txt" for s in SUBSETS})
            with contextlib.redirect_stdout(io.StringIO()),patch("grace_rul.grouped_study.fit_pipeline",side_effect=AssertionError("must reuse completed cells")):
                self.assertEqual(study.run_study(root,output),summary)
            with self.assertRaisesRegex(ValueError,"execution check"):
                study.freeze_recipe(root,output)
            self.assertFalse(list(output.rglob("grace_rul_ALL.pt")))


if __name__=="__main__":
    unittest.main()
