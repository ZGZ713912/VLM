"""Behavioral checks for causal prediction, provenance and reproducible updates."""
import copy
import io
import json
from pathlib import Path
import tarfile

import cv2
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from datasets.region_features import (MOTION_DIM, REGION_VERSION, RegionClips,
                                      load_region_cache, motion_descriptor, sha256)
from eval.region_evaluation import calibrate, causal_smooth, fuse
from models.region_temporal import RegionMemory, RegionTemporalModel
from prompts.region import region_prompts
from scripts.train_region_model import split_videos
from tools.prepare_ped2 import parse_matlab_intervals, safe_extract_ped2, verify_prepared
from train.region_trainer import region_loss, restore_rng, rng_state
from utils.reproducibility import set_seed


def tensors():
    generator = torch.Generator().manual_seed(9)
    appearance = F.normalize(torch.randn(2, 6, 3, 16, generator=generator), dim=-1)
    motion = torch.randn(2, 6, 3, MOTION_DIM, generator=generator)
    text = F.normalize(torch.randn(4, 16, generator=generator), dim=-1)
    return appearance, motion, text, torch.tensor([-1, -1, 1, 1])


@pytest.mark.parametrize("fusion", ["concat", "crossattn"])
def test_region_causal_and_required_output(fusion):
    a, m, text, _ = tensors()
    model = RegionTemporalModel(16, MOTION_DIM, 16, fusion).eval()
    # Make prediction trainable/nontrivial to expose accidental future dependence.
    torch.nn.init.normal_(model.appearance_predictor.weight, std=.01)
    output = model(a, m, text)
    changed_a, changed_m = a.clone(), m.clone()
    changed_a[:, 4:] = -changed_a[:, 4:]
    changed_m[:, 4:] += 100
    changed = model(changed_a, changed_m, text)
    for key in ("predicted_appearance", "predicted_motion"):
        torch.testing.assert_close(output[key][:, :4], changed[key][:, :4], atol=1e-6, rtol=1e-5)
    assert output["anomaly_score"].shape == (2,)
    assert output["frame_score"].shape == (2, 6)
    assert output["embedding"].shape == (2, 16)
    assert "explanation" in output


def test_synthetic_training_has_real_gradients_and_exact_rng_resume():
    set_seed(71)
    a, m, text, polarity = tensors()
    model = RegionTemporalModel(16, MOTION_DIM, 16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    generator = torch.Generator().manual_seed(71)
    weights = dict(appearance=1., motion=.25, bce=.2, contrastive=.05)

    def update():
        # Sampler order and augmentation randomness must both resume.
        order = torch.randperm(len(a), generator=generator)
        optimizer.zero_grad()
        losses = region_loss(model, a[order], m[order], text, polarity, weights)
        losses["loss"].backward()
        for name in ("gru.weight_ih_l0", "classifier.2.weight", "alignment.weight"):
            gradient = dict(model.named_parameters())[name].grad
            assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        optimizer.step()
        return float(losses["loss"])

    before = model.classifier[2].weight.detach().clone()
    update()
    assert not torch.equal(before, model.classifier[2].weight)
    saved_model = copy.deepcopy(model.state_dict())
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    saved_rng = rng_state(generator)
    expected_loss = update()
    expected_model = copy.deepcopy(model.state_dict())
    model.load_state_dict(saved_model)
    optimizer.load_state_dict(saved_optimizer)
    restore_rng(saved_rng, generator)
    assert update() == expected_loss
    for name, value in expected_model.items():
        torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0)


def test_memory_geometry_is_spatial_and_checkpoint_roundtrip(tmp_path):
    bank = torch.tensor([[[1., 0.], [1., 0.]], [[0., 1.], [0., 1.]]])
    memory = RegionMemory(bank, torch.zeros(2, 2, MOTION_DIM), k=1)
    a = torch.tensor([[[[1., 0.], [0., 1.]], [[0., 1.], [1., 0.]]]])
    m = torch.zeros(1, 2, 2, MOTION_DIM)
    scores = memory(a, m)
    torch.testing.assert_close(scores["frame_score"], torch.tensor([[0., 1.]]))
    torch.save(memory.state_dict(), tmp_path / "state.pt")
    restored = RegionMemory(bank, torch.zeros(2, 2, MOTION_DIM), k=1)
    restored.load_state_dict(torch.load(tmp_path / "state.pt", weights_only=True))
    torch.testing.assert_close(scores["frame_score"], restored(a, m)["frame_score"])


def test_motion_descriptor_not_labels_and_continuous_boundary():
    gray = np.zeros((64, 64), np.uint8)
    gray[12:32, 12:32] = 255
    shifted = np.roll(gray, 3, axis=1)
    first = motion_descriptor(None, gray, (2, 2))
    motion = motion_descriptor(gray, shifted, (2, 2))
    assert first.shape == motion.shape == (4, MOTION_DIM)
    assert np.isfinite(motion).all() and np.all(first == 0)
    assert motion[:, 0].max() > .1  # rightward translation


def test_train_split_and_calibration_independent_of_test():
    videos = {str(i): {"appearance": torch.ones(20, 3, 4), "motion": torch.zeros(20, 3, 18)} for i in range(8)}
    fit, held = split_videos(videos, .25, 42)
    assert set(fit).isdisjoint(held) and len(held) == 2
    clips = RegionClips(fit, 8, 8)
    assert set(k for k, _ in clips.indices) == set(fit)
    cfg = {"variants": {"primary": {"appearance": .5, "motion": .5}},
           "smooth_window": 3, "calibration_quantile": .99}
    raw = {"train_holdout": np.random.default_rng(42).normal(size=(50, 5))}
    calibration = calibrate(raw, cfg)
    assert calibration["source_video_ids"] == ["train_holdout"]
    query = raw["train_holdout"][:10].copy()
    score = fuse(query, calibration, cfg["variants"]["primary"], 3)
    # Adding extreme later test data cannot change earlier predictions or calibration.
    more = fuse(np.concatenate([query, np.ones((10, 5)) * 999]), calibration, cfg["variants"]["primary"], 3)
    np.testing.assert_allclose(score, more[:10])
    np.testing.assert_allclose(causal_smooth(np.array([1., 3., 5.]), 2), [1, 2, 4])


def test_region_manifest_rejects_corruption_and_backbone(tmp_path):
    videos, cache = tmp_path / "videos", tmp_path / "cache"
    videos.mkdir()
    cache.mkdir()
    video = videos / "a.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (16, 16))
    for _ in range(3):
        writer.write(np.zeros((16, 16, 3), np.uint8))
    writer.release()
    feature = cache / "a.pt"
    torch.save({"appearance": torch.ones(3, 2, 4), "motion": torch.zeros(3, 2, 18),
                "global": torch.ones(3, 4), "frame_indices": torch.arange(3)}, feature)
    backbone = {"provider": "test", "name": "x", "pretrained": "a"}
    manifest = {"version": REGION_VERSION, "resize": "full_frame_squash", "backbone": backbone,
                "num_regions": 2, "appearance_dim": 4,
                "videos": {"a": {"num_frames": 3, "source_sha256": sha256(video), "feature_sha256": sha256(feature)}}}
    (cache / "manifest.json").write_text(json.dumps(manifest))
    assert len(load_region_cache(cache, videos, backbone)[0]) == 1
    with pytest.raises(ValueError, match="backbone"):
        load_region_cache(cache, videos, {})
    feature.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_region_cache(cache, videos, backbone)


def test_annotation_parser_and_archive_path_guard(tmp_path):
    assert parse_matlab_intervals("Test{end+1}.gt_frame=[1:3, 6:8];") == [[(1, 3), (6, 8)]]
    with pytest.raises(ValueError):
        parse_matlab_intervals("x.gt_frame=[system('evil')];")
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        entry = tarfile.TarInfo("UCSD_Anomaly_Dataset.v1p2/UCSDped2/../../evil")
        entry.size = 1
        stream.addfile(entry, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="Unsafe"):
        safe_extract_ped2(archive, tmp_path / "out")


def test_prepared_dataset_reuse_checks_archive_and_labels(tmp_path):
    archive = tmp_path / "official.tar.gz"
    archive.write_bytes(b"recorded source archive")
    out, labels = tmp_path / "ped2", tmp_path / "labels"
    (out / "training_videos").mkdir(parents=True)
    (out / "testing_videos").mkdir()
    (labels / "testing").mkdir(parents=True)
    (out / "training_videos/t.avi").write_bytes(b"training video")
    (out / "testing_videos/v.avi").write_bytes(b"testing video")
    label_file = labels / "testing/v.pt"
    torch.save(torch.tensor([0, 1]), label_file)
    source = {"url": "official", "sha256": sha256(archive)}
    test_record = {"num_frames": 2, "num_anomaly_frames": 1,
                   "label_sha256": sha256(label_file), "video_sha256": sha256(out / "testing_videos/v.avi")}
    preparation = {"source": source, "training": {"t": {"video_sha256": sha256(out / "training_videos/t.avi")}},
                   "testing": {"v": test_record}}
    manifest = {"version": 1, "label_mode": "official", "split": "testing", "frame_index_base": 0,
                "reduction": "any_nonzero_pixel", "annotation_type": "pixel_mask_derived_frame_labels",
                "source": source, "videos": {"v": test_record}, "num_videos": 1, "num_frames": 2, "num_anomaly_frames": 1}
    (out / "preparation.json").write_text(json.dumps(preparation))
    (labels / "testing/manifest.json").write_text(json.dumps(manifest))
    assert verify_prepared(archive, out, labels)["num_frames"] == 2
    torch.save(torch.tensor([1, 1]), label_file)
    with pytest.raises(ValueError, match="labels changed"):
        verify_prepared(archive, out, labels)


@pytest.mark.parametrize("dataset", ["avenue", "ped2"])
@pytest.mark.parametrize("kind", ["label", "scene", "contrast"])
def test_explicit_prompts(dataset, kind):
    prompts, polarity = region_prompts("configs/region_prompts.yaml", dataset, kind)
    assert len(prompts) == len(polarity) and set(polarity.tolist()) == {-1, 1}
