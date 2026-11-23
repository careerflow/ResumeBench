"""Grades from different scorers never mix or overwrite each other."""

from __future__ import annotations

import json

from resume_bench.grading import grader
from resume_bench.settings import settings


def test_grades_are_kept_apart_by_scorer(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "output_dir", tmp_path)
    gt = {"basics": {"fname": "Wei", "lname": "DENG"}}

    class Case:
        resume_id, ground_truth = "r1", gt

    monkeypatch.setattr(grader, "load_split", lambda s: [Case()])
    res = tmp_path / "pipe" / "test"
    res.mkdir(parents=True)
    out = {"resume_id": "r1", "output": {"basics": {"fname": "DENG", "lname": "Wei"}}}
    (res / "r1.result.json").write_text(json.dumps(out))
    a = grader.grade_pipelines_strict(["pipe"], scorer="v5")
    b = grader.grade_pipelines_strict(["pipe"], scorer="strict")
    assert sorted(p.name for p in (res / "grades").iterdir()) == ["strict", "v5"]
    assert a["pipe"]["macro_f1"] == 1.0
    assert b["pipe"]["macro_f1"] < 1.0


def test_report_command_runs_after_grade(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from resume_bench.cli import app

    monkeypatch.setattr(settings, "output_dir", tmp_path)
    d = grader.grades_path("pipe", "test", "v5")
    d.mkdir(parents=True)
    (d / "r1.grade.json").write_text(json.dumps({"resume_id": "r1", "f1": 0.5}))
    out = CliRunner().invoke(app, ["report"])
    assert out.exit_code == 0
    assert "pipe" in out.output and "OptionInfo" not in out.output


def _cli():
    from typer.testing import CliRunner

    from resume_bench.cli import app

    return CliRunner(), app


def test_grade_output_names_the_scorer(tmp_path, monkeypatch):
    runner, app = _cli()
    monkeypatch.setattr(settings, "output_dir", tmp_path)
    monkeypatch.setattr(grader, "load_split", lambda s: [])
    (tmp_path / "pipe" / "test").mkdir(parents=True)
    out = runner.invoke(app, ["grade", "pipe"])
    assert out.exit_code == 0
    assert "scorer=v5" in out.output


def test_unknown_scorer_is_a_usage_error(tmp_path, monkeypatch):
    runner, app = _cli()
    monkeypatch.setattr(settings, "output_dir", tmp_path)
    for cmd in (["grade", "pipe"], ["grade-file", "p.jsonl"], ["leaderboard"], ["report"]):
        out = runner.invoke(app, [*cmd, "--scorer", "V5"])
        assert out.exit_code == 2, cmd


def test_leaderboard_reports_grades_from_before_v5(tmp_path, monkeypatch):
    runner, app = _cli()
    monkeypatch.setattr(settings, "output_dir", tmp_path)
    old = tmp_path / "pipe" / "test" / "grades"
    old.mkdir(parents=True)
    (old / "r1.grade.json").write_text(json.dumps({"resume_id": "r1", "f1": 0.5}))
    out = runner.invoke(app, ["leaderboard", "--scorer", "strict"])
    assert "grade --scorer strict" in out.output


def test_report_takes_scorer(tmp_path, monkeypatch):
    runner, app = _cli()
    monkeypatch.setattr(settings, "output_dir", tmp_path)
    d = grader.grades_path("pipe", "test", "strict")
    d.mkdir(parents=True)
    (d / "r1.grade.json").write_text(json.dumps({"resume_id": "r1", "f1": 0.5}))
    out = runner.invoke(app, ["report", "--scorer", "strict"])
    assert out.exit_code == 0 and "pipe" in out.output


def test_pre_v5_notice_stops_after_regrade_and_leaderboard_accepts_str(tmp_path, monkeypatch):
    from resume_bench.cli import leaderboard

    runner, app = _cli()
    monkeypatch.setattr(settings, "output_dir", tmp_path)
    old = tmp_path / "pipe" / "test" / "grades"
    old.mkdir(parents=True)
    (old / "r1.grade.json").write_text(json.dumps({"resume_id": "r1", "f1": 0.5}))
    new = grader.grades_path("pipe", "test", "strict")
    new.mkdir(parents=True)
    (new / "r1.grade.json").write_text(json.dumps({"resume_id": "r1", "f1": 0.5}))
    out = runner.invoke(app, ["leaderboard", "--scorer", "strict"])
    assert out.exit_code == 0 and "pipe" in out.output
    assert "before v5" not in out.output
    leaderboard(split="test", scorer="strict")
