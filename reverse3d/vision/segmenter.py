"""AI object segmentation (Segment Anything) for silhouette carving.

GrabCut only compares colours, so a dark cap or handle in front of a dark
background is cut off. Segment Anything (Meta, Apache-2.0) understands what
"the object" is: given a box and a point on it, it returns the whole object,
cap and handle included.

The model is allowlisted, pinned to an exact commit and loaded from
safetensors only (see :mod:`reverse3d.offline`); after the first download it
works without internet. Everything degrades gracefully: without the AI extra
or the model, callers fall back to GrabCut.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import numpy as np

from reverse3d.config import Settings
from reverse3d.errors import DependencyMissingError

logger = logging.getLogger(__name__)

#: Allowlisted segmentation models: id -> (kind, license).
ALLOWED_SEGMENTERS: dict[str, tuple[str, str]] = {
    "facebook/sam-vit-base": ("sam", "Apache-2.0"),
}
DEFAULT_SEGMENTER = "facebook/sam-vit-base"


def segmenter_available() -> bool:
    """True when the AI stack (torch + transformers) is installed."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


@lru_cache(maxsize=2)
def _load(model_id: str, device: str, model_dir: str) -> tuple[Any, Any]:
    from transformers import SamModel, SamProcessor

    try:  # keep CLI/agent output clean: no loading progress bars
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
        hf_logging.set_verbosity_error()
    except ImportError:  # pragma: no cover - very old transformers
        pass
    # model_dir is the local snapshot of a commit-pinned, allowlisted model
    # (reverse3d.offline.resolve_model): nothing is fetched here.
    model: Any = SamModel.from_pretrained(  # nosec B615
        model_dir, local_files_only=True, use_safetensors=True, trust_remote_code=False
    )
    processor = SamProcessor.from_pretrained(model_dir, local_files_only=True)  # nosec B615
    model.to(device).eval()
    logger.info("Loaded %s on %s", model_id, device)
    return model, processor


class ObjectSegmenter:
    """Segments the object inside a box (plus a point on it) in RGB images.

    Raises:
        DependencyMissingError: the AI extra or the model is not available.
    """

    def __init__(self, settings: Settings, device: str = "cpu", model_id: str = DEFAULT_SEGMENTER):
        if not segmenter_available():
            raise DependencyMissingError(
                "The AI segmentation stack is not installed.",
                hint='pip install "reverse3d[depth]"',
            )
        from reverse3d.offline import force_offline_env, is_offline, resolve_model

        self.device = device
        model_dir = resolve_model(model_id, settings)
        if is_offline(settings):
            force_offline_env()
        self.model, self.processor = _load(model_id, device, str(model_dir))

    def embed(self, rgb: np.ndarray) -> dict[str, Any]:
        """Run the heavy image encoder once; reuse the result for several prompts."""
        import torch

        inputs = self.processor(rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embeddings = self.model.get_image_embeddings(inputs["pixel_values"])
        return {
            "rgb": rgb,
            "embeddings": embeddings,
            "original_sizes": inputs["original_sizes"],
            "reshaped_input_sizes": inputs["reshaped_input_sizes"],
        }

    def segment(
        self,
        image: np.ndarray | dict[str, Any],
        box: tuple[int, int, int, int],
        point: tuple[int, int],
    ) -> np.ndarray:
        """Object mask (uint8, 255 = object) for a box ``(x0, y0, x1, y1)`` and a point.

        ``image`` is an RGB array or the result of :meth:`embed` (much faster).
        """
        import torch

        state = image if isinstance(image, dict) else self.embed(image)
        prompts = self.processor(
            state["rgb"],
            input_boxes=[[[float(v) for v in box]]],
            input_points=[[[[float(point[0]), float(point[1])]]]],
            input_labels=[[[1]]],
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(
                image_embeddings=state["embeddings"],
                input_boxes=prompts["input_boxes"],
                input_points=prompts["input_points"],
                input_labels=prompts["input_labels"],
                multimask_output=True,
            )
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            state["original_sizes"].cpu(),
            state["reshaped_input_sizes"].cpu(),
        )[0][0]
        scores = outputs.iou_scores.cpu()[0, 0]
        best = int(torch.argmax(scores))
        return np.where(masks[best].numpy(), 255, 0).astype(np.uint8)
