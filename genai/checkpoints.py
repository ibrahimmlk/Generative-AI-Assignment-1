"""Rebuild trained models from their checkpoint files."""
import torch

from .models import CorruptionClassifier, DenoisingAE, SoftMoE, StyleUNetGenerator

EXPERTS = ["salt_pepper", "blur", "occlusion"]


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def load_ae(path):
    c = _load(path)
    m = DenoisingAE(**c["cfg"])
    m.load_state_dict(c["state"])
    return m.eval()


def load_classifier(path):
    c = _load(path)
    m = CorruptionClassifier(**c["cfg"])
    m.load_state_dict(c["state"])
    return m.eval()


def load_moe(path):
    c = _load(path)
    m = SoftMoE(CorruptionClassifier(**c["gate_cfg"]), [DenoisingAE(**c["expert_cfg"]) for _ in EXPERTS], c["tau"])
    m.load_state_dict(c["state"])
    return m.eval()


def load_generator(path):
    c = _load(path)
    m = StyleUNetGenerator(**c["cfg"])
    m.load_state_dict(c["state"])
    return m.eval()
