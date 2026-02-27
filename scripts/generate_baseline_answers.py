#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _parse_time(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def _find_latest_summary(report_dir: Path) -> Path:
    summaries = list(report_dir.glob("*/run_summary.json"))
    if not summaries:
        raise FileNotFoundError(f"No run_summary.json found under {report_dir}")
    best = None
    best_key = None
    for path in summaries:
        data = _load_json(path)
        ts = _parse_time(data.get("timestamp"))
        key = ts.timestamp() if ts else path.stat().st_mtime
        if best_key is None or key > best_key:
            best_key = key
            best = path
    return best


def _extract_snippet(path: Path, start_pattern: str, num_lines: int) -> tuple[int, list[str]] | None:
    lines = path.read_text().splitlines()
    for idx, line in enumerate(lines):
        if start_pattern in line:
            start = idx + 1
            return start, lines[idx : idx + num_lines]
    return None


def _format_snippet(path: Path, start_line: int, lines: list[str]) -> str:
    header = f"# {path.as_posix()}:{start_line}"
    body = "\n".join(lines)
    return "\n".join([header, body])


def _word_count(text: str) -> int:
    return len([w for w in text.replace("\n", " ").split(" ") if w.strip()])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=None, help="Path to a report run directory.")
    parser.add_argument("--report-dir", type=str, default="reports", help="Root reports directory.")
    parser.add_argument("--out", type=str, default="reports/BASELINE_ANSWERS.md")
    args = parser.parse_args()

    report_dir = ROOT / args.report_dir
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        summary_path = run_dir / "run_summary.json"
    else:
        summary_path = _find_latest_summary(report_dir)
        run_dir = summary_path.parent

    summary = _load_json(summary_path)
    run_id = summary.get("run_id", run_dir.name)
    cmd = summary.get("command", "").strip()
    args_dict = summary.get("args", {})
    env = summary.get("env", {})
    system = summary.get("system", {})
    final_metrics = summary.get("final_metrics", {})
    final_log_line = summary.get("final_log_line")
    checkpoint = summary.get("checkpoint")
    runtime_sec = summary.get("runtime_sec")
    wandb = summary.get("wandb", {})

    env_id = env.get("env_id")
    obs_dim = env.get("obs_dim")
    act_dim = env.get("act_dim")
    mask_indices = env.get("mask_indices", [])
    total_steps = args_dict.get("total_steps")
    num_envs = args_dict.get("num_envs")
    horizon = args_dict.get("horizon")
    updates = None
    if total_steps and num_envs and horizon:
        updates = total_steps // (num_envs * horizon)

    snippet_file = ROOT / "src/ppo.py"
    snippet = _extract_snippet(snippet_file, "ratio = (logp - logp_old_f[mb]).exp()", 12)
    if snippet is None:
        snippet = _extract_snippet(snippet_file, "policy_loss = -torch.min", 12)
    snippet_text = "TODO: snippet not found."
    snippet_lines = ""
    if snippet is not None:
        start_line, lines = snippet
        snippet_lines = f"{start_line}-{start_line + len(lines) - 1}"
        snippet_text = _format_snippet(snippet_file, start_line, lines)

    requirements_path = (ROOT / "requirements.txt").as_posix()
    conda_env = system.get("conda_env")
    venv = system.get("virtual_env")
    env_manager = "conda" if conda_env else "pip/venv" if venv else "pip (unspecified)"
    cuda_version = system.get("cuda_version")
    gpu_name = system.get("gpu_name")
    device = system.get("device")
    ram_gb = system.get("ram_gb")

    dataset_desc = (
        f"Gymnasium {env_id} on-policy rollouts; size ~{total_steps} transitions "
        f"(updates={updates}, num_envs={num_envs}, horizon={horizon}); "
        f"features={obs_dim}, actions={act_dim}. "
        f"Source: Gymnasium classic_control. "
        f"Split: none (online RL); evaluation via rolling ret50. "
        f"Preprocess: PartialObsWrapper zeroes indices {mask_indices}."
    )

    reproducibility = (
        "Seeds set via numpy and torch; CUDA seeds are set when running with a CUDA device "
        "(see src/utils.py and amg.py). CuDNN deterministic=True, benchmark=False. "
        "Remaining nondeterminism: GPU kernel reductions and environment RNG."
    )

    loss_fn = (
        "L = L_clip(pi) + vf_coef * 0.5*(R - V)^2 - ent_coef * H(pi) + pred_coef * MSE(x_mem_next, pred(x_mem)) "
        "(pred term disabled in PPO-FF)."
    )

    answers = []
    answers.append(f"# Baseline Answers ({run_id})")
    answers.append("")
    answers.append("**Snippet**")
    answers.append("Paste a 10–20 line snippet from your code showing model training or data preprocessing (no boilerplate). "
                   "Which lines are yours? Why are they written that way? (≤60 words)")
    answers.append("")
    answers.append("```python")
    answers.append(snippet_text)
    answers.append("```")
    snippet_expl = (
        f"Lines {snippet_lines} are mine. They implement PPO's clipped objective, value regression, "
        "and optional prediction loss to stabilize updates under drift while keeping gradients bounded."
    )
    answers.append(snippet_expl)
    answers.append("")
    answers.append("**Command And Env**")
    answers.append("Show the exact command you used to run the training (e.g., python train.py --lr …). "
                   "Include the environment (conda/pip, CUDA version) and requirements.txt or environment.yml path.")
    answers.append("")
    answers.append("```bash")
    answers.append(cmd or "TODO: re-run with --report to capture the exact command.")
    answers.append("```")
    answers.append(
        f"Env manager: {env_manager}. "
        f"CUDA: {cuda_version or 'n/a'}. "
        f"GPU: {gpu_name or 'n/a'}. "
        f"Device: {device or 'n/a'}. "
        f"requirements.txt: {requirements_path}"
    )
    answers.append("")
    answers.append("**Dataset**")
    answers.append("What dataset did you use most recently? Name, size (#samples, #features or duration for audio), "
                   "source/license, train/val/test split strategy, and one data cleaning step you actually performed. (≤80 words)")
    answers.append("")
    answers.append(dataset_desc)
    answers.append("")
    answers.append("**Reproducibility**")
    answers.append("Reproducibility: how did you set seeds and control nondeterminism? List libraries where you set seeds "
                   "and one source of remaining nondeterminism you couldn’t eliminate. (≤60 words)")
    answers.append("")
    answers.append(reproducibility)
    answers.append("")
    answers.append("**Project Details**")
    answers.append("For one project, specify: task type (cls/reg/seq2seq/ASR/…); model family (e.g., ResNet50, XGBoost, "
                   "BiLSTM, BERT-base), why that choice over two alternatives, and the single most impactful hyperparameter "
                   "you tuned (value & search range). (≤100 words)")
    answers.append("")
    answers.append(
        "Task: reinforcement learning control (CartPole-v1). "
        "Model: feed-forward PPO actor-critic baseline (policy=ff). "
        "Why: simpler baseline than AMT traces or LSTM recurrent policy; isolates memory effects. "
        "Hyperparameter: TODO (fill with actual sweep, e.g., lr=3e-4 over [1e-4, 1e-3])."
    )
    answers.append("")
    answers.append("**Supervision Signal**")
    answers.append("Supervision signal: was it supervised, weakly-supervised, semi-supervised, or self-supervised? "
                   "Describe the labels or pretext task you actually used. (≤60 words)")
    answers.append("")
    answers.append(
        "Reinforcement learning with scalar reward from the environment; no explicit labels. "
        "Objective is to maximize expected return; auxiliary prediction loss disabled for PPO-FF baseline."
    )
    answers.append("")
    answers.append("**Metric Trade-off**")
    answers.append("List the primary metric you optimized and one trade-off it introduced (e.g., F1 vs. latency, "
                   "AUROC vs. prevalence). (≤60 words)")
    answers.append("")
    answers.append(
        "Primary metric: rolling return (ret50) and PPO surrogate objective. "
        "Trade-off: aggressive updates can raise return but increase KL/instability; clip_coef limits this."
    )
    answers.append("")
    answers.append("**Failure Mode**")
    answers.append("Show one concrete failure mode you found via error analysis (e.g., confusion matrix cell, "
                   "misclassified example with feature values, spectra snippet). Why did it fail, and how did you attempt "
                   "to fix it? (≤100 words)")
    answers.append("")
    answers.append("TODO: add a specific failure example from logs or a misbehavior case you observed.")
    answers.append("")
    answers.append("**Final Log + Checkpoint**")
    answers.append("Attach or paste the final validation log line (loss/metric) and the corresponding checkpoint filename or SHA. "
                   "Explain overfitting signs you observed (if any). (≤60 words)")
    answers.append("")
    answers.append(f"Final log line: {final_log_line or 'TODO: run with --report to capture final log line.'}")
    answers.append(f"Checkpoint: {checkpoint or 'TODO: run with --report to save checkpoint.'}")
    answers.append("Overfitting: not applicable for on-policy RL baseline, or TODO if observed.")
    answers.append("")
    answers.append("**Hardware And Runtime**")
    answers.append("What hardware did you train on (CPU/GPU type, RAM/VRAM)? Longest single run time and how you monitored it "
                   "(tensorboard/wandb/custom logs). (≤50 words)")
    answers.append("")
    answers.append(
        f"Hardware: GPU={gpu_name or 'n/a'}, device={device or 'n/a'}, RAM={ram_gb or 'n/a'} GB. "
        f"Longest run: {runtime_sec or 'TODO'} seconds. "
        "Monitoring: stdout logs; W&B if enabled."
    )
    answers.append("")
    answers.append("**Profiling**")
    answers.append("Did you profile bottlenecks? Provide one profiler screenshot or text excerpt (e.g., PyTorch profiler) "
                   "and what change you made as a result. (≤80 words)")
    answers.append("")
    answers.append("TODO: add profiler excerpt (e.g., torch.profiler) and the optimization you applied.")
    answers.append("")
    answers.append("**Experiment Tracking**")
    answers.append("How did you track experiments (e.g., W&B, MLflow, spreadsheets)? Paste one experiment ID/run URL or table "
                   "row and what decision it informed. (≤60 words)")
    answers.append("")
    if wandb:
        answers.append(
            f"W&B run: id={wandb.get('run_id')}, name={wandb.get('run_name')}, url={wandb.get('url')}. "
            "Decision: TODO (e.g., choose best seed or config)."
        )
    else:
        answers.append("Tracking: TODO (W&B/MLflow/spreadsheet).")
    answers.append("")
    answers.append("**Testing**")
    answers.append("Testing: describe one unit/integration test you wrote for data or model code (what it checks, where it lives). "
                   "(≤60 words)")
    answers.append("")
    answers.append(
        "tests/test_components.py: test_amt_rollout_and_update_cpu runs a short rollout + PPO update and asserts "
        "loss stats are finite; test_recurrent_rollout_and_update_cpu does the same for the LSTM policy."
    )
    answers.append("")
    answers.append("**Merge Request**")
    answers.append("Describe a merge request/PR you opened: link, title, and the reviewer’s main comment you addressed. (≤60 words)")
    answers.append("")
    answers.append("TODO: add PR link, title, and reviewer comment.")
    answers.append("")
    answers.append("**Team Contribution**")
    answers.append("If you worked in a team, what part would break without your contribution? Be specific "
                   "(data loader, loss function, deployment script, etc.). (≤50 words)")
    answers.append("")
    answers.append("TODO: add your specific contribution.")
    answers.append("")
    answers.append("**Dataset Bias**")
    answers.append("Cite one dataset bias or limitation you encountered; how did you measure or mitigate it? (≤80 words)")
    answers.append("")
    answers.append("TODO: add a concrete limitation (e.g., sparse drift events, short episodes) and mitigation.")
    answers.append("")
    answers.append("**Licensing**")
    answers.append("Licensing: what is the license of the code/models you used and is your repo’s license compatible? (≤50 words)")
    answers.append("")
    answers.append(
        "Repo license: Apache-2.0 (LICENSE). "
        "Dependencies: TODO confirm licenses (e.g., PyTorch, Gymnasium) and compatibility."
    )
    answers.append("")
    answers.append("**Loss Function**")
    answers.append("For your last model, write the exact loss function you minimized (symbolically) and name one regularization term "
                   "you used (if any). (≤60 words)")
    answers.append("")
    answers.append(loss_fn)
    answers.append("")
    answers.append("**Early Stopping / CV**")
    answers.append("If you used cross-validation or early stopping, specify the patience/folds and the selection criterion. (≤40 words)")
    answers.append("")
    answers.append("None used for PPO-FF baseline.")
    answers.append("")
    answers.append("**Confirmation**")
    answers.append("I confirm the repositories and logs referenced above are my own work or clearly indicate collaborators. (checkbox/confirmation)")
    answers.append("")
    answers.append("- [ ] Confirm")

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(answers))

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
