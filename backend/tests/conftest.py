from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vision.synthetic.scene import make_scene, write_scene

REPO = Path(__file__).resolve().parents[2]
MODELS = REPO / "models" / "weights"


@pytest.fixture(scope="session")
def synthetic_clip(tmp_path_factory) -> dict:
    """A short, clearly synthetic two-player rally video with exact ground truth."""
    d = tmp_path_factory.mktemp("clip")
    scene = make_scene(duration=8.0, fps=30.0, width=640, height=360, seed=5)
    video, truth = write_scene(scene, d / "rally")
    return {"scene": scene, "video": video, "truth_path": truth, "truth": json.loads(truth.read_text())}


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


def models_available(*names: str) -> bool:
    return all((MODELS / n).exists() for n in names)
