"""What was photographed: a small local AI (TinyCLIP, MIT) that names the subject.

The subject drives everything downstream: how the photos are reconstructed
(scene profile), how the model is rendered (studio, portrait, architecture or
aerial light) and which capture advice is given. TinyCLIP compares the photos
with short descriptions of each subject (zero-shot), so no training data is
needed and new subjects are a line of text.

Like the other AI models, it is allowlisted, pinned to an exact commit,
loaded from safetensors only and works offline after the first download.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from reverse3d.config import Settings
from reverse3d.errors import DependencyMissingError

logger = logging.getLogger(__name__)

ALLOWED_CLASSIFIERS: dict[str, tuple[str, str]] = {
    "wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M": ("clip", "MIT"),
}
DEFAULT_CLASSIFIER = "wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M"

#: subject -> descriptions (averaged). Order matters only for ties.
SUBJECTS: dict[str, tuple[str, ...]] = {
    "product": (
        "a photo of a product on a table",
        "a photo of a bottle, cup or household object",
        "a photo of an object on a desk",
    ),
    "part": ("a photo of a mechanical part", "a photo of a tool or machine component"),
    "figure": ("a photo of a statue or figurine", "a photo of a toy or sculpture"),
    "person": ("a photo of a person", "a photo of a man or a woman standing"),
    "animal": ("a photo of an animal", "a photo of a dog, cat or pet", "a photo of a wild animal"),
    "vehicle": ("a photo of a car", "a photo of a vehicle or motorcycle"),
    "building": ("a photo of a building exterior", "a photo of a house facade"),
    "interior": ("a photo of a room interior", "a photo of the inside of a house"),
    "forest": ("a photo of a forest or trees", "a photo of plants and vegetation"),
    "terrain": ("an aerial photo of land or terrain", "a photo of a landscape or field"),
}

#: How each subject is reconstructed (scene profile) ...
SUBJECT_SCENE: dict[str, str] = {
    "product": "object",
    "part": "object",
    "figure": "object",
    "vehicle": "object",
    "person": "character",
    "animal": "character",
    "building": "architecture",
    "interior": "architecture",
    "forest": "landscape",
    "terrain": "landscape",
}
#: ... and rendered.
SUBJECT_RENDER: dict[str, str] = {
    "product": "studio",
    "part": "studio",
    "figure": "studio",
    "vehicle": "showroom",
    "person": "portrait",
    "animal": "portrait",
    "building": "architecture",
    "interior": "architecture",
    "forest": "aerial",
    "terrain": "aerial",
}
SUBJECT_LABEL_ES: dict[str, str] = {
    "product": "Objeto o producto",
    "part": "Pieza mecánica",
    "figure": "Figura o escultura",
    "vehicle": "Vehículo",
    "person": "Persona",
    "animal": "Animal",
    "building": "Edificio",
    "interior": "Interior",
    "forest": "Bosque o vegetación",
    "terrain": "Terreno o paisaje",
}


def classifier_available() -> bool:
    """True when the AI stack (torch + transformers) is installed."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def _features(output: Any) -> Any:
    """Projected embeddings: a tensor in transformers 4, an output object in 5."""
    return output if hasattr(output, "norm") else output.pooler_output


@lru_cache(maxsize=1)
def _load(model_dir: str) -> tuple[Any, Any, Any]:
    import torch
    from transformers import CLIPModel, CLIPProcessor

    try:
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
        hf_logging.set_verbosity_error()
    except ImportError:  # pragma: no cover
        pass
    # model_dir is the local snapshot of a commit-pinned, allowlisted model.
    model: Any = CLIPModel.from_pretrained(  # nosec B615
        model_dir, local_files_only=True, use_safetensors=True, trust_remote_code=False
    )
    processor = CLIPProcessor.from_pretrained(model_dir, local_files_only=True)  # nosec B615
    model.eval()
    prompts = [p for texts in SUBJECTS.values() for p in texts]
    with torch.no_grad():
        tokens = processor(text=prompts, return_tensors="pt", padding=True)
        text = _features(model.get_text_features(**tokens))
        text = text / text.norm(dim=-1, keepdim=True)
    # One averaged, normalised embedding per subject.
    rows, start = [], 0
    for texts in SUBJECTS.values():
        emb = text[start : start + len(texts)].mean(0)
        rows.append(emb / emb.norm())
        start += len(texts)
    return model, processor, torch.stack(rows)


def classify_photos(paths: list[Path], settings: Settings) -> dict[str, float]:
    """Probability of each subject for a set of photos (averaged over them).

    Raises:
        DependencyMissingError: the AI extra or the model is not available.
    """
    if not classifier_available():
        raise DependencyMissingError(
            "The AI subject classifier is not installed.", hint='pip install "reverse3d[depth]"'
        )
    import torch
    from PIL import Image

    from reverse3d.offline import force_offline_env, is_offline, resolve_model

    model_dir = resolve_model(DEFAULT_CLASSIFIER, settings)
    if is_offline(settings):
        force_offline_env()
    model, processor, subjects = _load(str(model_dir))
    images = []
    for path in paths:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            rgb.thumbnail((448, 448))
            images.append(rgb)
    with torch.no_grad():
        pixels = processor(images=images, return_tensors="pt")
        feats = _features(model.get_image_features(**pixels))
        feats = feats / feats.norm(dim=-1, keepdim=True)
        logits = model.logit_scale.exp() * feats @ subjects.T
        probs = logits.softmax(dim=-1).mean(0).cpu().numpy()
    result = {name: float(p) for name, p in zip(SUBJECTS, np.asarray(probs), strict=True)}
    logger.info("Subject: %s", max(result, key=result.__getitem__))
    return result
