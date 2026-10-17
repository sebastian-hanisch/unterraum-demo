"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mini_x():
    """x = (1, 0, -1, 0, 1, 0) (Kosinus bei f = 1/4), Matrixgröße 3: vier Schnappschüsse (1, 0, -1), (0, -1, 0), (-1, 0, 1), (0, 1, 0);
    R = [[0,5; 0; -0,5], [0; 0,5; 0], [-0,5; 0; 0,5]] (schon vorwärts-rückwärts-symmetrisch), Eigenwerte 1 / 0,5 / 0 mit den Eigenvektoren
    (1, 0, -1)/sqrt 2, (0, 1, 0), (1, 0, 1)/sqrt 2."""
    return np.array([1.0, 0.0, -1.0, 0.0, 1.0, 0.0])
