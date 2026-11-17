import pytest
import torch

from src.medicalnet import load_medicalnet
from src.model import ResNet18, ShortcutA, build_model


def test_outputs_features_and_gradients():
    model = ResNet18(base_channels=8)
    x = torch.randn(2, 1, 16, 16, 16)
    features = model.forward_features(x)
    assert features.shape == (2, 64)
    logits = model.forward_head(features)
    assert logits.shape == (2, 4)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    assert model.conv1.weight.grad.abs().sum() > 0
    activation = model.get_layer4_activations(x)
    gradient = torch.autograd.grad(
        model.forward_from_activations(activation)[:, 0].sum(), activation
    )[0]
    assert gradient.shape == activation.shape
    assert torch.isfinite(gradient).all()


def test_shortcut_preserves_gradients():
    x = torch.randn(2, 4, 8, 8, 8, requires_grad=True)
    out = ShortcutA(8, 2)(x)
    out.sum().backward()
    assert x.grad.abs().sum() > 0
    assert torch.all(out[:, 4:] == 0)


def test_full_width_model_and_medicalnet_load(tmp_path):
    model = ResNet18()
    assert model.fc.in_features == 512
    assert sum(parameter.numel() for parameter in model.fc.parameters()) == 2052
    model.eval()
    with torch.no_grad():
        assert model.forward_features(torch.randn(1, 1, 16, 16, 16)).shape == (1, 512)
    state = {
        "module." + k: v.clone()
        for k, v in model.state_dict().items()
        if not k.startswith("fc.")
    }
    state["module.conv_seg.0.weight"] = torch.ones(1)
    path = tmp_path / "synthetic_medicalnet.pth"
    torch.save({"state_dict": state}, path)
    head = model.fc.weight.detach().clone()
    report = load_medicalnet(model, path)
    assert report["success"]
    assert report["backbone_tensor_fraction"] == 1.0
    assert report["sha256_match"] is None
    assert torch.equal(model.fc.weight, head)
    del state["module.layer4.1.conv2.weight"]
    torch.save({"state_dict": state}, path)
    with pytest.raises(ValueError, match="incompatible"):
        load_medicalnet(model, path)


def test_medicalnet_rejects_checkpoint_checksum_mismatch(tmp_path):
    model = ResNet18()
    path = tmp_path / "checkpoint.pth"
    torch.save({"state_dict": {}}, path)
    with pytest.raises(ValueError, match="SHA-256"):
        load_medicalnet(model, path, expected_sha256="0" * 64)


def test_pretrained_requires_correct_architecture():
    with pytest.raises(ValueError, match="shortcut A"):
        load_medicalnet(ResNet18(base_channels=8), "unused")


def test_dropout_pooling_and_hybrid_medicalnet_load(tmp_path):
    from src.model import HybridInceptionResNet3D, ResNet18, build_model

    for arch in ("ResNet18", "Inception3D", "InceptionResNet3D", "HybridInceptionResNet3D"):
        for pool in ("gap", "attention", "roi_attention"):
            model = build_model(
                {
                    "architecture": arch,
                    "num_classes": 4,
                    "base_channels": 8,
                    "shortcut": "A",
                    "dropout": 0.5,
                    "pooling": pool,
                }
            )
            x = torch.randn(2, 1, 16, 16, 16)
            features = model.forward_features(x)
            assert features.shape == (2, 64)
            logits = model(x)
            assert logits.shape == (2, 4)
            act = model.get_layer4_activations(x)
            grad = torch.autograd.grad(
                model.forward_from_activations(act)[:, 0].sum(), act
            )[0]
            assert grad.shape == act.shape
            assert torch.isfinite(grad).all()

    # Verify HybridInceptionResNet3D loads MedicalNet stem (conv1..layer3) cleanly
    ref_resnet = ResNet18()
    state = {
        "module." + k: v.clone()
        for k, v in ref_resnet.state_dict().items()
        if not k.startswith("fc.")
    }
    state["module.conv_seg.0.weight"] = torch.ones(1)
    ckpt_path = tmp_path / "synthetic_medicalnet_hybrid.pth"
    torch.save({"state_dict": state}, ckpt_path)

    hybrid = HybridInceptionResNet3D(dropout=0.5, pooling="roi_attention")
    report = load_medicalnet(hybrid, ckpt_path)
    assert report["success"]
    assert report["hybrid_stem_transfer"] is True
    assert report["backbone_tensor_fraction"] == 1.0


def test_yang_resnet_cbam_variants_and_tcav_contract():
    for arch, expected_dim in [
        ("YangResNet18_CBAM", 256),
        ("YangResNet34_CBAM", 256),
        ("YangResNet50_CBAM", 1024),
    ]:
        config = {
            "architecture": arch,
            "num_classes": 4,
            "base_channels": 32,
            "shortcut": "B",
            "use_cbam": True,
            "cbam_reduction": 16,
            "bottleneck_expansion": 4,
            "dropout": 0.0,
            "pooling": "gap",
        }
        model = build_model(config)
        x = torch.randn(2, 1, 32, 32, 32, requires_grad=True)
        act = model.get_layer4_activations(x)
        assert act.shape == (2, expected_dim, 1, 1, 1)
        logits = model.forward_from_activations(act)
        assert logits.shape == (2, 4)
        loss = logits.sum()
        loss.backward()
        assert x.grad is not None
