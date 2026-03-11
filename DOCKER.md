# Docker Quickstart

This repo now includes:

- `Dockerfile` (Python 3.11 + system libs used by Gym/CarRacing/video tooling)
- `docker-compose.yml` (CPU service + optional GPU profile)
- `.dockerignore` (keeps build context small and avoids copying local logs/artifacts)

## 1) Build the image

```bash
docker compose build
```

## 2) Smoke check (prints CLI help)

```bash
docker compose run --rm app
```

## 3) Run tests inside the container

```bash
docker compose run --rm app -m pytest -q tests/test_components.py tests/test_algorithms.py tests/test_reporting.py
```

## 4) Run a CPU training command

```bash
docker compose run --rm app amg.py configs/benchmarks/cartpole/stationary_fullobs.yaml \
  --algo ppo \
  --run-note "docker cpu run" \
  --device cpu \
  --report \
  --report-dir reports/benchmarks \
  --report-run-name cartpole_stationary_fullobs_ppo_s0_docker_cpu
```

## 5) Run with GPU (requires NVIDIA Container Toolkit)

Use the GPU profile:

```bash
docker compose --profile gpu run --rm app-gpu amg.py configs/benchmarks/cartpole/stationary_fullobs.yaml \
  --algo ppo \
  --run-note "docker gpu run" \
  --device cuda \
  --cuda-id 0 \
  --report \
  --report-dir reports/benchmarks \
  --report-run-name cartpole_stationary_fullobs_ppo_s0_docker_gpu
```

## Notes

- The repo is bind-mounted into `/workspace`, so outputs (`reports/`, `wandb/`) stay on your host.
- `WANDB_MODE` defaults to `offline`; override at runtime if needed:
  `WANDB_MODE=online docker compose run --rm app ...`
