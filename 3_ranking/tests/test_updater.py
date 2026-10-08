"""The dashboard's Update runner: step order, stop-then-finish, failure handling."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ranking import updater

PY = [sys.executable, "-u", "-c"]


def _steps(fetch, build="print('built')", rank="print('ranked')"):
    cwd = Path(__file__).parent
    return (("fetch", "Fetch", PY + [fetch], cwd),
            ("build", "Build", PY + [build], cwd),
            ("rank", "Rank", PY + [rank], cwd))


def _wait(job, timeout=20):
    end = time.time() + timeout
    while job.running and time.time() < end:
        time.sleep(0.05)


def test_runs_all_steps_in_order(monkeypatch):
    monkeypatch.setattr(updater, "STEPS", _steps("print('fetched')"))
    job = updater.UpdateJob()
    assert job.start() and not job.start()          # second click is ignored
    _wait(job)
    log = job.tail(50)
    assert job.state == "done" and job.generation == 1
    assert log.index("fetched") < log.index("built") < log.index("ranked")


def test_failed_fetch_still_builds_and_ranks(monkeypatch):
    monkeypatch.setattr(updater, "STEPS", _steps("raise SystemExit(3)"))
    job = updater.UpdateJob()
    job.start()
    _wait(job)
    assert job.state == "done" and "ranked" in job.tail(50)


def test_stop_kills_the_fetch_but_finishes_the_rest(monkeypatch):
    monkeypatch.setattr(updater, "STEPS", _steps("import time; print('go'); time.sleep(60)"))
    job = updater.UpdateJob()
    job.start()
    while "go" not in job.tail(50):
        time.sleep(0.05)
    job.stop_fetch()
    _wait(job)
    assert job.state == "done" and "ranked" in job.tail(50)


def test_failed_build_marks_the_job_failed(monkeypatch):
    monkeypatch.setattr(updater, "STEPS", _steps("pass", build="raise SystemExit(1)"))
    job = updater.UpdateJob()
    job.start()
    _wait(job)
    assert job.state == "failed" and "ranked" not in job.tail(50)
