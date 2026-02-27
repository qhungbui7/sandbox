% Adaptive Memory Traces: Handling Non-Stationary Control with PPO

# Abstract
Adaptive Memory Traces (AMT) tackles reinforcement-learning tasks where the environment undergoes hidden, piecewise drifts. The single-file system (`amg.py`) couples partial observability, a drift-aware memory, and PPO with auxiliary prediction losses. This paper summarizes the problem formulation, the logic that glues AMT together, and the algorithms + benchmarks we will report.

# 1. Problem Definition
- **Setting:** We train an agent on environments such as `CartPole-v1` that have hidden state perturbations. Drift is injected by swapping observation/reward mappings without warning.
- **Challenge:** Standard PPO struggles because on-policy batches mix pre/post-drift transitions, causing stale advantages and uninformative gradients.
- **Goal:** Maintain a compact recurrent representation that (1) reacts quickly to abrupt changes, (2) preserves long-term context when the environment is stable, and (3) exposes drift events so PPO can adapt resets/learning rates.

# 2. System Logic
1. **Partial Observation Wrapper** hides a subset of state dimensions to force reliance on learned memory.
2. **Piecewise Drift Wrapper** triggers parameter jumps following a configurable scheduler (rho, tau, cooldown).
3. **FeatureEncoder → Traces:** Observations + previous actions pass through MLPs and two exponential trace banks (short/long horizons). The adaptive gate (`alpha`) softly interpolates between retaining history vs. refreshing.
4. **Predictor Head** performs next-embedding prediction so reconstruction error highlights regime shifts.
5. **DriftMonitor** tracks short/long error averages. When short-term deviation rises above `tau_on` for `K` consecutive steps, we reset part of the trace bank (partial/zero/obs strategies).
6. **ActorCritic + PPO Update** consume flattened traces and actions, producing policy logits, value predictions, and auxiliary prediction losses.
7. **Telemetry Layer** mirrors stats to stdout + W&B (returns, KL, entropy, gate mean), enabling long-run diagnosis.

# 3. Algorithm Outline
```
for update in range(total_steps / (num_envs * horizon)):
    batch = rollout(envs, ac, drift_monitor, predictor)
    adv, returns = compute_gae(batch)
    for epoch in range(epochs):
        perm = torch.randperm(batch_size, generator=rng)
        for mb in minibatches(perm):
            with autocast(device="cuda"):
                logits, values = ac(batch.obs[mb], ...)
                policy_loss = clipped_surrogate(logits, actions, adv)
                value_loss = (returns - values)^2 / 2
                pred_loss = mse(predictor(x_mem), x_mem_next)
                loss = policy + vf_coef * value - ent_coef * entropy + pred_coef * pred
            scaler.step(opt)  # if AMP float16
    ema_update(f_mem_teacher, ac.f_pol, tau=ema_tau)
```
Key implementation details:
- `GradScaler` only activates when AMP float16 is on; bfloat16 falls back to unscaled grads.
- Trace resets respect `reset_strategy` (`zero`, `obs`, `partial`) and only the long portion (`reset_long_fraction`) when partial.
- DriftMonitor enforces cooldown to avoid repeated resets during the same drift.

# 4.1 Baselines to run
- **AMT-PPO (ours):** adaptive gate on multi-timescale traces, partial resets, predictor loss, AMP fp16.
- **PPO-FF:** feed-forward PPO proxy via `alpha-base=max=1.0`, no resets, no predictor (removes memory/gating).
- **PPO Fixed Trace:** multi-timescale traces without gating/resets (`alpha-base=max`, no predictor) to isolate benefit of adaptation.
- **Zero Reset:** AMT-PPO but uses hard zero resets instead of partial.
- **No AMP / No Predictor:** ablations on precision and auxiliary loss.
- **Recurrent PPO:** LSTM core baseline (`policy=recurrent`) with same optimizer/hyperparams.

Config files (CartPole, 500k frames, 8 envs) live under `configs/cartpole/`:
`amt.yaml`, `no_amp.yaml`, `no_pred.yaml`, `zero_reset.yaml`, `ppo_ff.yaml`, `ppo_fixed_trace.yaml`, `recurrent.yaml`.

For the exact single-run commands and paper-style sweeps, use [`COMMANDS_TO_RUN.md`](COMMANDS_TO_RUN.md) (or [`run_paper.sh`](run_paper.sh) for multi-seed runs with consistent naming/W&B logging).

# 4. Benchmark Plan
| Component | Purpose | Default Values | Variations to Sweep |
| --- | --- | --- | --- |
| Environment | Baseline | `CartPole-v1`, 8 envs, horizon 256 | Add `Acrobot-v1`, higher env counts (16/32) |
| Drift params | Stress non-stationarity | `rho_s=0.1`, `rho_l=0.01`, `tau_on=2.5`, `tau_off=1.0`, `K=4` | Looser/tighter gates, longer cooldown |
| Reset strategy | Memory response | `partial`, `reset_long_fraction=0.5` | `zero`, `obs` |
| Predictor loss | Auxiliary signal | `lambda_pred=0.01`, `pred_coef=0.5` | {0, 0.25, 1.0} coefficients |
| AMP | Throughput | On (float16) | Off, bfloat16 |

**Metrics to report**
- Rolling return (`ret50`), episode length (`len50`), KL divergence, clip fraction.
- Drift health: average gate value, number of resets per 1000 steps.
- Throughput: environment frames/sec, optimizer steps/sec, GPU memory usage.

**Procedure**
1. Fix seeds {0,1,2}.  
2. Run `500k` frames per configuration.  
3. Log to W&B (`--wandb-project amt`) and export CSV summaries.  
4. Compare mean ± std of final `ret50` and adaptation lag (steps until post-drift return recovers to 80% of pre-drift level).

# 5. Results (to fill post-experiment)
| Config | ret50 @500k | len50 @500k | Gate Mean | Drift Resets / 1k steps | Throughput (fps) |
| --- | --- | --- | --- | --- | --- |
| Baseline (AMP fp16, partial reset) | _todo_ | _todo_ | _todo_ | _todo_ | _todo_ |
| PPO-FF (no memory) | _todo_ | _todo_ | _todo_ | _todo_ | _todo_ |
| PPO Fixed Trace | _todo_ | _todo_ | _todo_ | _todo_ | _todo_ |
| No AMP | _todo_ | _todo_ | _todo_ | _todo_ | _todo_ |
| No Predictor Loss | _todo_ | _todo_ | _todo_ | _todo_ | _todo_ |
| Zero Reset Strategy | _todo_ | _todo_ | _todo_ | _todo_ | _todo_ |

Add narrative once table is populated: highlight best configuration, adaptation lag, and any regressions.

# 6. Discussion & Future Work
- Investigate sequence models (GRU/Transformer) to replace exponential traces.
- Explore meta-learned drift thresholds instead of fixed `tau_on/off`.
- Extend benchmarks to continuous-control (MuJoCo) and stochastic drift schedules.

# References
- Schulman et al., “Proximal Policy Optimization Algorithms,” 2017.  
- Goyal et al., “Reinforcement Learning with Evolving Dynamics,” 2021.  
- W&B documentation for large-scale RL telemetry.
