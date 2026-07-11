"""
Pathos AI — Shared pytest fixtures
"""
from __future__ import annotations

import os

os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("PATHOS_AI_JWT_SECRET_KEY", "test-secret-key-not-for-production")

import pytest


@pytest.fixture
def sample_pii_message() -> str:
    return (
        "Hi, my name is Jane Doe. My phone number is (415) 555-0132 and my "
        "email is jane.doe@example.com. My MRN is MRN-88213 and my DOB is 04/12/1990."
    )
