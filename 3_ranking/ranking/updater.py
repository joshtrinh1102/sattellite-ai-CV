"""Background refresh for the dashboard's Update button.

    fetch   python -m factors.build --fetch-news   fill GDELT/weather gaps (hours;
                                                   stoppable - the cache keeps progress)
    build   python -m factors.build                rebuild handoff/factors_daily.csv
                                                   from whatever is cached
    rank    python -m ranking.rank                 re-run the study on the new CSV

Steps run in child processes so a GDELT stall can be killed without taking the
app down. `build` and `rank` always run, even after a stopped or failed fetch:
the fetch only writes cache files, and the factors CSV is only rewritten by
`build`, so skipping it would throw away the progress that was just made.
"""

import subprocess
import sys
import threading
import time
from collections import deque

from .config import FACTORS_DIR, ROOT

PY = [sys.executable, "-u"]
STEPS = (
    ("fetch", "Fetch news + weather", PY + ["-m", "factors.build", "--fetch-news"],
     FACTORS_DIR),
    ("build", "Rebuild daily factors from cache", PY + ["-m", "factors.build"],
     FACTORS_DIR),
    ("rank", "Re-rank factors", PY + ["-m", "ranking.rank"], ROOT),
)


class UpdateJob:
    """One refresh at a time, shared by every session of the app."""

    def __init__(self):
        self.state = "idle"          # idle | running | done | failed
        self.step = None
        self.finished_at = None
        self.generation = 0          # bumped on every finish, so pages know to reload
        self.log = deque(maxlen=400)
        self._proc = None
        self._skip_fetch = threading.Event()
        self._lock = threading.Lock()

    @property
    def running(self):
        return self.state == "running"

    def start(self):
        with self._lock:
            if self.running:
                return False
            self.state, self.step = "running", None
            self._skip_fetch.clear()
            self.log.clear()
        threading.Thread(target=self._run, daemon=True).start()
        return True

    def stop_fetch(self):
        """Cut the fetch short; the cache keeps what it got and the rest still runs."""
        self._skip_fetch.set()
        proc = self._proc
        if proc and self.step == "fetch" and proc.poll() is None:
            proc.terminate()

    def tail(self, n=25):
        return "\n".join(list(self.log)[-n:])

    def _run(self):
        ok = True
        for key, title, cmd, cwd in STEPS:
            if key == "fetch" and self._skip_fetch.is_set():
                continue
            self.step = key
            self.log.append(f"=== {title} ===")
            code = self._exec(cmd, cwd)
            if code != 0:
                self.log.append(f"[{title} exited with code {code}]")
                if key != "fetch":       # a failed fetch is expected; the rest can still run
                    ok = False
                    break
        self.step = None
        self.finished_at = time.time()
        self.state = "done" if ok else "failed"
        self.generation += 1

    def _exec(self, cmd, cwd):
        try:
            self._proc = subprocess.Popen(
                cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace")
            for line in self._proc.stdout:
                self.log.append(line.rstrip())
            return self._proc.wait()
        except OSError as exc:
            self.log.append(f"[could not start: {exc}]")
            return -1
