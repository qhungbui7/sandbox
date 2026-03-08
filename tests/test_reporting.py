import json
from pathlib import Path

import torch

from src.reporting import start_run_report


def test_run_reporter_writes_metrics_logs_and_summary(tmp_path: Path):
    reporter = start_run_report(
        repo_root=Path(__file__).resolve().parents[1],
        report_dir=tmp_path / "reports",
        run_name="unit_report",
        args={"env_id": "CartPole-v1", "policy": "ff", "algo": "ppo"},
        device="cpu",
        obs_dim=4,
        act_dim=2,
        mask_indices=[],
        config_path=None,
        enabled=True,
    )
    reporter.log_metrics({"loop/frames": 16, "train/ret50": 10.0, "perf/eta_sec": 1.0})
    reporter.log_line("unit test line")
    ckpt = reporter.save_checkpoint({"x": torch.tensor(1)})
    reporter.finalize(checkpoint_path=ckpt)

    summary_path = tmp_path / "reports" / "unit_report" / "run_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["training_stats"]["metric_rows"] == 1
    assert summary["training_stats"]["best_ret50"] == 10.0
    assert summary["training_stats"]["second_best_ret50"] is None
    assert summary["training_stats"]["last_ret50"] == 10.0
    assert summary["checkpoint"] == str(ckpt)
    assert summary["active_args"]["policy"] == "ff"
    assert summary["active_args"]["algo"] == "ppo"
    assert summary["active_args"]["env_id"] == "CartPole-v1"
    assert "git_commit_hash" in summary
    if (summary.get("git_commit_hash") is not None) and (summary.get("git_sha") is not None):
        assert str(summary["git_commit_hash"]).startswith(str(summary["git_sha"]))

    metrics_path = Path(summary["artifacts"]["metrics_jsonl"])
    metrics_csv_path = Path(summary["artifacts"]["metrics_csv"])
    logs_path = Path(summary["artifacts"]["logs_txt"])
    assert metrics_path.exists()
    assert metrics_csv_path.exists()
    csv_text = metrics_csv_path.read_text()
    assert "loop/frames" in csv_text
    assert "train/ret50" in csv_text
    assert logs_path.exists()
    assert "unit test line" in logs_path.read_text()


def test_run_reporter_tracks_best_second_and_last_ret50(tmp_path: Path):
    reporter = start_run_report(
        repo_root=Path(__file__).resolve().parents[1],
        report_dir=tmp_path / "reports",
        run_name="unit_report_ranked",
        args={"env_id": "CartPole-v1", "policy": "ff", "algo": "ppo"},
        device="cpu",
        obs_dim=4,
        act_dim=2,
        mask_indices=[],
        config_path=None,
        enabled=True,
    )
    reporter.log_metrics({"loop/frames": 16, "train/ret50": 10.0})
    reporter.log_metrics({"loop/frames": 32, "train/ret50": 7.0})
    reporter.log_metrics({"loop/frames": 48, "train/ret50": 12.0})
    reporter.log_metrics({"loop/frames": 64, "train/ret50": 9.0})
    reporter.finalize()

    summary_path = tmp_path / "reports" / "unit_report_ranked" / "run_summary.json"
    summary = json.loads(summary_path.read_text())
    stats = summary["training_stats"]
    assert stats["best_ret50"] == 12.0
    assert stats["second_best_ret50"] == 10.0
    assert stats["last_ret50"] == 9.0
