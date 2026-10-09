import sqlite3

from melomane.unattended import build_tail_command, llm_budget_cap, wait_for_file


def _cache(path, costs):
    db = sqlite3.connect(str(path))
    db.execute("CREATE TABLE cache (k TEXT PRIMARY KEY, resp TEXT, cost REAL, tin INT, tout INT)")
    db.executemany("INSERT INTO cache VALUES (?,?,?,?,?)", [(str(i), "{}", c, 0, 0) for i, c in enumerate(costs)])
    db.commit(); db.close()


def test_budget_cap_is_what_is_already_spent_plus_the_allowance(tmp_path):
    p = tmp_path / "llm.sqlite"
    _cache(p, [0.25, 0.5, 10.0])
    assert llm_budget_cap(p, 1.5) == 12.25
    assert llm_budget_cap(tmp_path / "missing.sqlite", 1.5) == 1.5            # no cache yet: just the allowance


def test_wait_for_file_returns_when_it_appears_and_times_out_otherwise(tmp_path):
    f = tmp_path / "S5.json"
    clock = {"t": 0.0, "polls": 0}
    def sleep(s):
        clock["t"] += s; clock["polls"] += 1
        if clock["polls"] == 3:
            f.write_text("{}")
    assert wait_for_file(f, timeout_s=1000, poll_s=60, sleep=sleep, now=lambda: clock["t"]) is True and clock["polls"] == 3
    g = tmp_path / "never.json"
    clock.update(t=0.0, polls=0)
    assert wait_for_file(g, timeout_s=300, poll_s=60, sleep=sleep, now=lambda: clock["t"]) is False and clock["polls"] == 5


def test_tail_command_runs_the_listed_stages_without_prompting():
    cmd = build_tail_command("onion", "standard", ["S6", "S7"], python="py")
    assert cmd == ["py", "-m", "melomane", "run-all", "--corpus", "onion", "--profile", "standard", "--stages", "S6,S7", "--no-prompt", "--no-awake"]
