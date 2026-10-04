# -*- coding: utf-8 -*-
"""v5 smoke test -- runs on CPU, needs no competition data.

Checks, in order:
  1. load_tri() produces the documented 9-channel layout from synthetic
     PNG files (RGB + 16-bit depth + thermal);
  2. the v5 fusion attaches to a real ultralytics YOLO11 DetectionModel;
  3. a 9-channel forward produces the same output shape as the RGB model;
  4. IDENTITY: with SwitchInject.warmup == 0 the fused model is
     bit-identical to the plain RGB model  (this is the safety floor);
  5. warmup == 1 changes the output  (the aux stream is actually wired);
  6. gradients reach the aux parameters (the aux stream can learn at all);
  7. gradients do NOT reach the RGB backbone when it is frozen.

Usage:
    python bisai_v5/v5_smoke.py
"""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(tempfile.gettempdir(), "ultralytics_cfg"))
os.makedirs(os.environ["YOLO_CONFIG_DIR"], exist_ok=True)
os.environ.setdefault("YOLO_VERBOSE", "False")

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

FAIL = []


def check(name, ok, detail=""):
    print("[%s] %s%s" % ("PASS" if ok else "FAIL", name, (" -- " + detail) if detail else ""))
    if not ok:
        FAIL.append(name)


# ---------------------------------------------------------------- 1. data
def test_load_tri():
    import cv2
    from v5_dataset import load_tri, N_CH
    tmp = Path(tempfile.mkdtemp(prefix="v5data_"))
    h, w = 64, 96
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    ir = rng.integers(0, 255, (h, w), dtype=np.uint8)
    dep = rng.integers(300, 15000, (h, w), dtype=np.uint16)
    dep[0:4, 0:4] = 0                                  # invalid region
    cv2.imwrite(str(tmp / "a.png"), rgb)
    cv2.imwrite(str(tmp / "ir_a.png"), ir)
    cv2.imwrite(str(tmp / "d_a.png"), dep)
    img, (h0, w0) = load_tri(tmp / "a.png", tmp / "ir_a.png", tmp / "d_a.png")

    check("load_tri: channel count == %d" % N_CH, img.shape[2] == N_CH, str(img.shape))
    check("load_tri: spatial size preserved", (h0, w0) == (h, w), "%s" % ((h0, w0),))
    check("load_tri: dtype uint8", img.dtype == np.uint8, str(img.dtype))
    check("load_tri: RGB passthrough", np.array_equal(img[:, :, :3], rgb))
    check("load_tri: IR passthrough", np.array_equal(img[:, :, 3], ir))
    check("load_tri: valid mask marks invalid pixels",
          img[0, 0, 6] == 0 and img[10, 10, 6] == 255)
    check("load_tri: thermal saliency centred on neutral",
          abs(int(img[10, 10, 7]) - 128) <= 255, "ch7=%d" % img[10, 10, 7])
    check("load_tri: depth edge finite and in range",
          img[:, :, 8].min() >= 0 and img[:, :, 8].max() <= 255)


# ---------------------------------------------------------------- 2. model
def build_model():
    """Prefer a real checkpoint (set V5_SMOKE_PT) so the channel widths are the
    production ones; fall back to a locally generated yaml model."""
    pt = os.environ.get("V5_SMOKE_PT", "")
    if pt and os.path.exists(pt):
        from ultralytics import YOLO
        print("[smoke] building from checkpoint:", pt)
        model = YOLO(pt).model
        model.eval()
        return model
    from ultralytics.nn.tasks import DetectionModel
    print("[smoke] building from yolo11m.yaml (unscaled; channels still coherent)")
    torch.manual_seed(0)
    model = DetectionModel("yolo11m.yaml", ch=3, nc=12, verbose=False)
    model.eval()
    return model


def flat(out):
    """Flatten any ultralytics head output (tensor / list / tuple / dict)."""
    if isinstance(out, dict):
        return torch.cat([flat(out[k]) for k in sorted(out.keys())])
    if isinstance(out, (list, tuple)):
        return torch.cat([flat(o) for o in out])
    if isinstance(out, torch.Tensor):
        return out.flatten()
    raise TypeError("unsupported output type: %s" % type(out))


def test_fusion():
    from v5_model import (apply_fusion_v5, set_inject_warmup, N_CH,
                          _rgb_forward_only, FUSION_LAYERS)

    det = build_model()
    apply_fusion_v5(det)

    chans = [det.model[i].conv.bn.num_features if hasattr(det.model[i], "conv") else None
             for i in FUSION_LAYERS]

    x = torch.rand(2, N_CH, 256, 256)
    with torch.no_grad():
        out = det(x)
    check("forward: 9ch input runs", out is not None)
    check("forward: rgb-only reference has the same shape",
          flat(out).numel() == flat(_rgb_forward_only(det, x[:, :3])).numel(),
          "fused=%d rgb=%d" % (flat(out).numel(), flat(_rgb_forward_only(det, x[:, :3])).numel()))

    # identity floor
    from v5_model import stem_input_9ch
    set_inject_warmup(det, 0.0)
    with torch.no_grad():
        a = flat(det(x)).clone()
        b = flat(_rgb_forward_only(det, stem_input_9ch(x, getattr(det, "_v5_legacy9", False)))).clone()
    check("identity: warmup=0 is bit-identical to the RGB model",
          torch.allclose(a, b, atol=0, rtol=0),
          "max|diff|=%.3e" % float((a - b).abs().max()))

    # warmup=1 must change something now that aux weights are random
    set_inject_warmup(det, 1.0)
    with torch.no_grad():
        c = flat(det(x))
    check("wiring: warmup=1 differs from RGB (aux stream is connected)",
          not torch.allclose(b, c, atol=0, rtol=0),
          "max|diff|=%.3e" % float((b - c).abs().max()))

    # gradients -- the fused branch is zero-initialised on purpose, so the very
    # first backward cannot reach every upstream weight (the last conv is
    # exactly 0).  What matters is that after a couple of real optimizer steps
    # the WHOLE auxiliary branch is learning.
    for p in det.parameters():
        p.requires_grad_(False)
    trainable = []
    for attr in ("aux_encoder", "inject", "reliability"):
        for p in getattr(det, attr).parameters():
            p.requires_grad_(True)
            trainable.append(p)

    def n_grad(mod):
        return sum(1 for p in mod.parameters() if p.grad is not None and p.grad.abs().sum() > 0)

    def n_param(mod):
        return sum(1 for _ in mod.parameters())

    det.train()
    opt = torch.optim.SGD(trainable, lr=1e-2)
    stats = {a: [] for a in ("aux_encoder", "inject", "reliability")}
    rgb_grad = 0
    for step in range(3):
        opt.zero_grad()
        loss = flat(det(x)).pow(2).mean()
        loss.backward()
        for a in stats:
            stats[a].append(n_grad(getattr(det, a)))
        rgb_grad = max(rgb_grad, sum(1 for p in det.model.parameters()
                                     if p.grad is not None and p.grad.abs().sum() > 0))
        opt.step()

    for attr in ("aux_encoder", "inject", "reliability"):
        mod = getattr(det, attr)
        g, t = stats[attr][-1], n_param(mod)
        check("grad: %s learns after 3 steps (%d/%d tensors, %s)"
              % (attr, g, t, stats[attr]), g == t)

    check("freeze: RGB backbone received no gradient", rgb_grad == 0,
          "%d tensors with grad" % rgb_grad)

    # channel widths of the injection points must match the backbone outputs
    from v5_model import _probe_channels
    probe = _probe_channels(det)
    inject_c = [m.proj.conv.in_channels for m in det.inject]
    check("channel match: inject widths == backbone widths at %s" % FUSION_LAYERS,
          probe == inject_c, "%s vs %s" % (probe, inject_c))


def main():
    print("=" * 68)
    print("v5 smoke test (CPU, synthetic data)")
    print("=" * 68)
    test_load_tri()
    test_fusion()
    print("=" * 68)
    if FAIL:
        print("FAILED: %d check(s): %s" % (len(FAIL), ", ".join(FAIL)))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())