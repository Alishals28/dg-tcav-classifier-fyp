import pytest
import torch

from src.medicalnet import load_medicalnet
from src.model import ResNet18, ShortcutA


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
