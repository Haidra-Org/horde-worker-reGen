# AI Horde Worker reGen

Share your GPU with the world. Earn [kudos](https://github.com/Haidra-Org/haidra-assets/blob/main/docs/kudos.md). Generate AI images faster.

The [AI Horde](https://aihorde.net/) is a free, open, decentralized platform where anyone can contribute GPU power to generate images. When your worker completes jobs, you earn **kudos** — the more you have, the faster your own image requests get processed.

## Quick Start

> **Prerequisites**: An NVIDIA GPU (8 GB+ VRAM recommended), [git](https://gitforwindows.org/) (Windows) or `git` (Linux), and an [AI Horde API key](https://aihorde.net/register).

### 1. Download the worker

**Windows** — open Command Prompt or PowerShell:

```cmd
git clone https://github.com/Haidra-Org/horde-worker-reGen.git
cd horde-worker-reGen
```

**Linux** — open a terminal:

```bash
git clone https://github.com/Haidra-Org/horde-worker-reGen.git
cd horde-worker-reGen
```

> **Tip**: Do not use spaces in the installation path (`C:\horde_worker` is fine, `C:\My Workers` is not).

<details>
<summary>No git? Download the zip instead.</summary>

Download the [latest zip](https://github.com/Haidra-Org/horde-worker-reGen/archive/refs/heads/main.zip), extract it, and open a terminal in the extracted folder.
</details>

### 2. Launch the interactive setup

Double-click (or run) the launcher for your OS:

| OS | Launcher |
|----|----------|
| Windows | `horde-worker.cmd` |
| Linux | `./horde-worker.sh` |

The launcher automatically installs dependencies on first run (no separate install step needed), then opens an **interactive terminal UI** that walks you through:

1. Entering your API key and choosing a worker name
2. Selecting your GPU type
3. Downloading AI models
4. Starting the worker

That's it — you're contributing to the horde!

### Alternative: command-line scripts

If you prefer non-interactive scripts:

1. **Install dependencies**: run `update-runtime.cmd` (Windows) or `./update-runtime.sh` (Linux).
2. **Edit config**: copy `bridgeData_template.yaml` to `bridgeData.yaml` and fill in your API key and worker name.
3. **Start the worker**: run `horde-bridge.cmd` (Windows) or `./horde-bridge.sh` (Linux).

## Contents

- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [GPU-Specific Setup](#gpu-specific-setup)
- [Running the Worker](#running-the-worker)
- [Updating](#updating)
- [Custom Models](#custom-models)
- [Docker](#docker)
- [Support & Troubleshooting](#support--troubleshooting)

## Configuration

### Basic Settings

If you used the interactive launcher, your config was created automatically. To edit it later, open `bridgeData.yaml` in any text editor.

If you're setting up manually:

1. Copy `bridgeData_template.yaml` to `bridgeData.yaml`.
2. Set your `api_key` (from [aihorde.net/register](https://aihorde.net/register)). **Keep this secret.**
3. Set a unique `dreamer_name`. If it's already taken, you'll get a "Wrong Credentials" error.

### Recommended Settings by GPU (Click to expand)

<details>
<summary><strong>24 GB+ VRAM</strong> (RTX 3090, 4090)</summary>

```yaml
queue_size: 1        # <32 GB RAM: 0, 32 GB: 1, >32 GB: 2
safety_on_gpu: true
high_memory_mode: true
high_performance_mode: true
unload_models_from_vram_often: false
max_threads: 1       # 2 is often viable for xx90 cards
post_process_job_overlap: true
max_power: 64        # Reduce if max_threads: 2
max_batch: 8         # Increase if max_threads: 1, decrease if max_threads: 2
allow_sdxl_controlnet: true
```
</details>

<details>
<summary><strong>12–16 GB VRAM</strong> (RTX 3080 Ti, 4070 Ti, 4080)</summary>

```yaml
queue_size: 1        # <32 GB RAM: 0, 32 GB: 1, >32 GB: 2
safety_on_gpu: true  # Consider false if using Cascade/Flux
moderate_performance_mode: true
unload_models_from_vram_often: false
max_threads: 1
max_power: 50
max_batch: 4         # Or higher
```
</details>

<details>
<summary><strong>8–10 GB VRAM</strong> (RTX 2080, 3060, 4060, 4060 Ti)</summary>

```yaml
queue_size: 1        # <32 GB RAM: 0, 32 GB: 1, >32 GB: 2
safety_on_gpu: false
max_threads: 1
max_power: 32        # No higher
max_batch: 4         # No higher
allow_post_processing: false  # If using SDXL/Flux, else can be true
allow_sdxl_controlnet: false
```

Minimize other VRAM-consuming apps while the worker runs.
</details>

<details>
<summary><strong>Lower-end GPUs / Under-performing workers</strong></summary>

- `extra_slow_worker: true` — gives more time per job, but requesters must opt-in. Only use if consistently under 0.3 MPS/s or 3000 kudos/hr with correct config.
- `limit_max_steps: true` — caps total steps per job based on model type.
- `preload_timeout: 120` — allows longer model load times.
</details>

<details>
<summary><strong>Systems with less than 32 GB RAM</strong></summary>

- Set `queue_size: 0` and stick to SD 1.5 models only.
- Set `load_large_models: false`.
- Add `ALL SDXL`, `ALL SD21`, and the unpruned models to `models_to_skip`.
</details>

### Hardware Tips

- **Use an SSD.** HDDs are too slow for multiple models — limit to one model with <60 s load time.
- **Configure 8 GB+ swap** (16 GB+ preferred), even on Linux.
- **Keep `max_threads` ≤ 2** unless you have a 48 GB+ VRAM data center GPU.
- **Disable sleep/power-saving** while the worker runs.
- SDXL needs ~9 GB free RAM (32 GB+ total recommended). Flux/Cascade need ~20 GB free RAM (48 GB+ total recommended).

## GPU-Specific Setup

### NVIDIA (default)

No extra steps. The standard scripts and the interactive launcher default to CUDA.

### AMD (ROCm) — Linux only

AMD support is **experimental** and Linux-only.

- Use `update-runtime-rocm.sh` and `horde-bridge-rocm.sh` instead of the standard versions.
- [WSL support](README_advanced.md#advanced-users-amd-rocm-inside-windows-wsl) is highly experimental.
- Join the [AMD discussion on Discord](https://discord.com/channels/781145214752129095/1076124012305993768) for help.

### DirectML — Windows (experimental)

DirectML is **several times slower** than CUDA or ROCm. Use only if you have no other option.

- Use `update-runtime-directml.cmd` and `horde-bridge-directml.cmd`.
- See [Running on DirectML](README_advanced.md#advanced-users-running-on-directml) for details.

## Running the Worker

### Starting

> The worker is resource-intensive. Avoid gaming or other heavy tasks while it runs.

**Recommended**: use `horde-worker.cmd` / `./horde-worker.sh` for the interactive launcher.

**Alternative**: use `horde-bridge.cmd` / `./horde-bridge.sh` (or the `-rocm` / `-directml` variants).

### Stopping

Press `Ctrl+C` in the worker's terminal. It will finish any in-progress jobs before exiting.

### Logs

Logs are saved in the `logs/` directory:

| File | Contents |
|------|----------|
| `bridge.log` | Main log (all info) |
| `bridge_n.log` | Per-process log |
| `trace.log` | Errors and warnings only |
| `trace_n.log` | Per-process errors |

### Multiple GPUs

> Future versions will not require multiple worker instances.

For now, run one worker per GPU. On Linux:

```bash
CUDA_VISIBLE_DEVICES=0 ./horde-bridge.sh -n "GPU-0"
CUDA_VISIBLE_DEVICES=1 ./horde-bridge.sh -n "GPU-1"
```

Running multiple workers needs high RAM (32–64 GB+). `queue_size` and `max_threads` multiply memory use.

## Updating

Stay up to date via our [Discord](https://discord.gg/3DxrhksKzn). Script names below assume Windows + NVIDIA; for Linux use `.sh`, for AMD use `-rocm` variants.

1. **Stop** the worker (`Ctrl+C`).
2. **Pull updates**:
   - Git users: `git pull`
   - Zip users: download the [latest zip](https://github.com/db0/horde-worker-reGen/archive/refs/heads/main.zip), extract over the existing folder.
3. **Update dependencies**: The interactive launcher handles this for you. You can also run `update-runtime.cmd` (or the relevant variant) to update manually.
4. **Start** the worker again.

> **Antivirus note**: Some antivirus (e.g. Avast) may interfere with downloads. If you see `CRYPT_E_NO_REVOCATION_CHECK` errors, temporarily disable it.

## Custom Models

Serving custom models requires the `customizer` role — request it on [Discord](https://discord.gg/3DxrhksKzn).

With the role:

1. Download your model files locally.
2. Add them to `bridgeData.yaml`:

   ```yaml
   custom_models:
     - name: My Custom Model
       baseline: stable_diffusion_xl
       filepath: /path/to/model/file.safetensors
   ```

   Supported baselines: `stable_diffusion_1`, `stable_diffusion_2_768`, `stable_diffusion_2_512`, `stable_diffusion_xl`, `stable_cascade`, `flux_1`.

   > **Warning**: Only Flux.schnell models are allowed. Flux.dev and its derivatives are **not** permitted.

3. Add the model `name` to your `models_to_load` list.

Custom model names can't conflict with existing horde model names. The horde treats them as SD 1.5 for kudos and safety purposes.

## Docker

Docker images: <https://hub.docker.com/r/tazlin/horde-worker-regen/tags>

See the [Docker guide](Dockerfiles/README.md) for setup instructions.

## Support & Troubleshooting

**Get help**: [#local-workers on Discord](https://discord.com/channels/781145214752129095/1076124012305993768) or [open an issue](https://github.com/Haidra-Org/horde-worker-reGen/issues).

| Problem | Fix |
|---------|-----|
| Download failures | Check disk space and internet connection. |
| Job timeouts | Remove large models (Flux, Cascade, SDXL), lower `max_power`, disable post-processing/controlnet/lora. |
| Out of memory | Lower `max_threads`, `max_batch`, or `queue_size`. Close other programs. |
| Less kudos than expected | New workers have 50% of job kudos and 100% of uptime kudos held in escrow for ~1 week until you become trusted. |
| Worker in maintenance mode | Log into [artbot](https://tinybots.net/artbot/settings?panel=workers) with the worker running and click "unpause". Check [logs](logs/README.md) for ERROR entries to find the root cause. |

For advanced setup options (manual `uv` usage, custom environments, etc.), see [README_advanced.md](README_advanced.md).

## Model Usage & Licenses

Many bundled models use the [CreativeML OpenRAIL License](https://huggingface.co/spaces/CompVis/stable-diffusion-license). Please review it before use.
