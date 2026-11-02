"""Loss and selection checks for severe errors on held engine prefixes."""

from pathlib import Path
import tempfile
import unittest

import pandas as pd
import torch

from grace_rul.predictor_safety import (engine_tail_score, fault_accuracy_guards,
                                       restore_refinement_checkpoint, tail_rul_loss)
from grace_rul.checkpoints import random_state, save_checkpoint
from scripts.finetune_predictor_continuity import finetune_predictor


class PredictorSafety(unittest.TestCase):
    def test_tail_loss_focuses_gradients_on_large_errors_without_altering_predictions(self):
        prediction = torch.tensor([1., 2., 3., 40.], requires_grad=True)
        original = prediction.detach().clone()
        loss = tail_rul_loss(prediction, torch.zeros(4), scale=10., fraction=.25)
        loss.backward()
        self.assertEqual(float(loss.detach()), 16.)
        torch.testing.assert_close(prediction.detach(), original)
        torch.testing.assert_close(prediction.grad[:3], torch.zeros(3))
        self.assertGreater(float(prediction.grad[3]), 0.)

    def test_engine_tail_is_engine_balanced_and_rejects_duplicate_prefixes(self):
        frame = pd.DataFrame([{"unit": 1, "cycle": i, "target": 10., "prediction": 11.}
                              for i in range(1, 21)] +
                             [{"unit": 2, "cycle": 1, "target": 100., "prediction": 0.}])
        self.assertEqual(engine_tail_score(frame, .5), 100.)
        self.assertEqual(engine_tail_score(frame, 1.), 50.5)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            engine_tail_score(pd.concat([frame, frame.iloc[:1]]))

    def test_fault_guard_cannot_hide_a_bad_source_in_good_pooled_accuracy(self):
        reference = {"stuck": {"RMSE": 10., "source_metrics": {"FD002": {"RMSE": 8.}}}}
        measured = {"stuck": {"RMSE": 9., "source_metrics": {"FD002": {"RMSE": 9.}}}}
        guards = fault_accuracy_guards(measured, reference, .03)
        self.assertTrue(guards["stuck_rul"])
        self.assertFalse(guards["stuck_rul_FD002"])

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source.pt"
            source.write_bytes(b"preserve")
            output = root / "completed"
            output.mkdir()
            (output / "sentinel").write_text("original")
            with self.assertRaises(FileExistsError):
                finetune_predictor(root, source, output, device="cpu")
            self.assertEqual(source.read_bytes(), b"preserve")
            self.assertEqual((output / "sentinel").read_text(), "original")

    def test_resume_restores_parameters_optimizer_scheduler_and_sampling_rng(self):
        torch.manual_seed(42)
        model = torch.nn.Linear(2,1)
        optimizer = torch.optim.AdamW(model.parameters(),lr=.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,1,.9)
        generator = torch.Generator().manual_seed(17)
        def step(net,opt,sch,gen):
            x=torch.randn((4,2),generator=gen)
            opt.zero_grad();net(x).square().mean().backward();opt.step();sch.step()
        step(model,optimizer,scheduler,generator)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"resume.pt"
            save_checkpoint(path,{"source_bundle_sha256":"source", "implementation_sha256":{"code":"same"},
                "config":{"device":"cpu","recipe":1}, "state_dict":model.state_dict(),
                "optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),
                "loader_rng":generator.get_state(),"rng":random_state()})
            step(model,optimizer,scheduler,generator)
            expected={key:value.clone() for key,value in model.state_dict().items()}
            other=torch.nn.Linear(2,1);opt=torch.optim.AdamW(other.parameters(),lr=.5)
            sch=torch.optim.lr_scheduler.StepLR(opt,1,.8);gen=torch.Generator().manual_seed(99)
            restore_refinement_checkpoint(path,other,opt,sch,gen,config={"device":"cpu","recipe":1},
                source_hash="source",implementation={"code":"same"})
            step(other,opt,sch,gen)
            for key,value in other.state_dict().items():
                torch.testing.assert_close(value,expected[key],rtol=0,atol=0)
            with self.assertRaisesRegex(ValueError,"recipe or implementation"):
                restore_refinement_checkpoint(path,other,opt,sch,gen,config={"device":"cpu","recipe":2},
                    source_hash="source",implementation={"code":"same"})


if __name__ == "__main__":
    unittest.main()
