"""The exported detector must use calibrated full scores and remain label-free."""
import numpy as np
import torch
from torch.nn import functional as F

from models.region_detector import load_region_detector
from models.region_temporal import RegionMemory, RegionTemporalModel


def test_export_uses_calibrated_spatial_score_with_causal_inference(tmp_path):
    torch.manual_seed(13)
    model = RegionTemporalModel(8, 18, hidden_dim=16, top_regions=1)
    # The auxiliary classifier is constant; its predictions cannot explain novelty.
    with torch.no_grad():
        for parameter in model.classifier.parameters():
            parameter.zero_()
    bank = F.normalize(torch.randn(2, 3, 8), dim=-1)
    memory = RegionMemory(bank, torch.zeros(2, 3, 18), k=1)
    text = F.normalize(torch.randn(4, 8), dim=-1)
    calibration = {"component_order": ["appearance", "motion", "prediction", "semantic", "classifier"],
                   "center": [0.] * 5, "scale": [1.] * 5, "thresholds": {"full": .1}}
    config = {"model": {"hidden_dim": 16, "fusion": "concat", "num_heads": 4, "top_regions": 1},
              "memory": {"k": 1}, "eval": {"primary_variant": "full", "smooth_window": 2,
                                           "variants": {"full": {"appearance": 1.}}}}
    state = {"version": 1, "config": config, "prompts": ["n1", "n2", "a1", "a2"],
             "calibration": calibration, "text": text, "polarity": torch.tensor([-1, -1, 1, 1]),
             "model": model.state_dict(), "memory": memory.state_dict()}
    path = tmp_path / "detector.pt"
    torch.save(state, path)
    detector = load_region_detector(path)
    appearance = bank[:, 0].unsqueeze(0).unsqueeze(0).expand(1, 4, -1, -1).clone()
    appearance[:, 2:] *= -1
    motion = torch.zeros(1, 4, 2, 18)
    scores = detector(appearance, motion)
    assert scores["anomaly_score"].shape == (1,)
    assert scores["frame_score"].shape == (1, 4)
    assert scores["embedding"].shape == (1, 8)
    assert "explanation" in scores
    assert scores["frame_score"][0, 0] < .0001
    assert scores["frame_score"][0, -1] > .1
    first = detector(appearance[:, :2], motion[:, :2])
    np.testing.assert_allclose(first["frame_score"].numpy(), scores["frame_score"][:, :2].numpy(), atol=1e-6)
    assert not any("label" in name for name in detector.state_dict())
