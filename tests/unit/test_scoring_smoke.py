"""Smoke tests: known GT/pred pairs must produce exact F1 values."""

from __future__ import annotations

from resume_bench.grading.strict import score_resume_strict
from resume_bench.schema.sections import SECTIONS


def _make_resume(basics=None, experience=None, education=None):
    """Build a minimal resume dict."""
    r = {}
    if basics is not None:
        r["basics"] = basics
    if experience is not None:
        r["experience"] = experience
    if education is not None:
        r["education"] = education
    return r


class TestScoringSmoke:
    def test_perfect_extraction(self):
        """Identical GT and pred must score 1.0."""
        gt = _make_resume(
            basics={"fname": "Alice", "lname": "Smith", "email": "alice@example.com"},
            experience=[
                {
                    "company": "Acme Corp",
                    "position": "Engineer",
                    "startYear": 2020,
                    "endYear": 2023,
                    "description": ["Built APIs", "Led team of 5"],
                }
            ],
        )
        result = score_resume_strict(gt, gt, SECTIONS, gate=1.0)
        assert result["f1"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_empty_pred_scores_zero(self):
        """Empty prediction against non-empty GT must score 0."""
        gt = _make_resume(
            basics={"fname": "Alice", "lname": "Smith"},
            experience=[
                {
                    "company": "Acme Corp",
                    "position": "Engineer",
                    "startYear": 2020,
                    "endYear": 2023,
                }
            ],
        )
        pred = _make_resume()
        result = score_resume_strict(gt, pred, SECTIONS, gate=1.0)
        assert result["f1"] == 0.0

    def test_partial_basics(self):
        """Missing fields in basics lower the score but don't crash."""
        gt = _make_resume(
            basics={"fname": "Alice", "lname": "Smith", "email": "a@b.com", "phone": "555-1234"},
        )
        pred = _make_resume(
            basics={"fname": "Alice", "lname": "Smith"},
        )
        result = score_resume_strict(gt, pred, SECTIONS, gate=1.0)
        assert 0.0 < result["f1"] < 1.0
        assert result["recall"] < 1.0

    def test_hallucinated_experience(self):
        """Extra predicted entries should lower precision."""
        gt = _make_resume(
            experience=[
                {"company": "Acme", "position": "Engineer", "startYear": 2020, "endYear": 2023},
            ],
        )
        pred = _make_resume(
            experience=[
                {"company": "Acme", "position": "Engineer", "startYear": 2020, "endYear": 2023},
                {"company": "Fake Inc", "position": "Wizard", "startYear": 2018, "endYear": 2019},
            ],
        )
        result = score_resume_strict(gt, pred, SECTIONS, gate=1.0)
        assert result["precision"] < 1.0
        assert result["recall"] == 1.0

    def test_deterministic(self):
        """Same inputs must produce identical scores across runs."""
        gt = _make_resume(
            basics={"fname": "Bob", "lname": "Jones", "email": "bob@test.com"},
            experience=[
                {"company": "BigCo", "position": "Manager", "startYear": 2015, "endYear": 2020},
            ],
            education=[
                {"institution": "State University", "studyType": "BS", "startYear": 2011, "endYear": 2015},
            ],
        )
        pred = _make_resume(
            basics={"fname": "Bob", "lname": "Jones", "email": "bob@test.com"},
            experience=[
                {"company": "BigCo", "position": "Manager", "startYear": 2015, "endYear": 2020},
            ],
            education=[
                {"institution": "State University", "studyType": "BS", "startYear": 2011, "endYear": 2015},
            ],
        )
        results = [score_resume_strict(gt, pred, SECTIONS, gate=1.0) for _ in range(10)]
        assert all(r["f1"] == results[0]["f1"] for r in results)
