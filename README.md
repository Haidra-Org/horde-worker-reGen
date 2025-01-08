# AI Horde Worker reGen

Welcome to the [AI Horde](https://github.com/Haidra-Org/AI-Horde), a free and open decentralized platform for collaborative AI! The AI Horde enables people from around the world to contribute their GPU power to generate images, text, and more. By running a worker on your local machine, you can earn [kudos](https://github.com/Haidra-Org/haidra-assets/blob/main/docs/kudos.md) which give you priority when making your own requests to the horde.

A worker is a piece of software that handles jobs from the AI Horde, such as generating an image from a text prompt. When your worker successfully completes a job, you are rewarded with kudos. The more kudos you have, the faster your own requests will be processed.

You can read about [kudos](https://github.com/Haidra-Org/haidra-assets/blob/main/docs/kudos.md), the reward granted to you for running a worker, including some reasons for running a worker on our [detailed kudos explanation](https://github.com/Haidra-Org/haidra-assets/blob/main/docs/kudos.md).

## Contents

- [AI Horde Worker reGen](#ai-horde-worker-regen)
  - [Contents](#contents)
  - [Before You Begin](#before-you-begin)
  - [Installation](#installation)
    - [Windows](#windows)
      - [Option 1: Using Git (Recommended)](#option-1-using-git-recommended)
      - [Option 2: Without Git](#option-2-without-git)
    - [Linux](#linux)
    - [AMD GPUs](#amd-gpus)
    - [DirectML](#directml)
  - [Configuration](#configuration)
    - [Basic Settings](#basic-settings)
    - [Suggested Settings](#suggested-settings)
    - [Important Notes](#important-notes)
  - [Running the Worker](#running-the-worker)
    - [Starting](#starting)
    - [Stopping](#stopping)
    - [Monitoring](#monitoring)
    - [Multiple GPUs](#multiple-gpus)
  - [Updating](#updating)
    - [Updating the Worker](#updating-the-worker)
    - [Updating the Runtime](#updating-the-runtime)
  - [Custom Models](#custom-models)
  - [Docker](#docker)
  - [Support \& Troubleshooting](#support--troubleshooting)
  - [Model Usage \& Licenses](#model-usage--licenses)

## Before You Begin

Before installing the worker:

1. Register an account on the [AI Horde website](https://aihorde.net/register).
2. Securely store the API key you receive. **Treat this key like a password**.


## Installation

### Windows

#### Option 1: Using Git (Recommended)

1. Install [git for Windows](https://gitforwindows.org/) if you haven't already.
2. Open PowerShell or Command Prompt.
3. Navigate to the folder where you want to install the worker:

   ```cmd
   cd C:\path\to\install\folder
   ```

4. Clone the repository:

   ```cmd
   git clone https://github.com/Haidra-Org/horde-worker-reGen.git
   cd horde-worker-reGen
   ```

#### Option 2: Without Git

1. Download the [zipped worker files](https://github.com/Haidra-Org/horde-worker-reGen/archive/refs/heads/main.zip).
2. Extract to a folder of your choice.

### Linux

Open a terminal and run:

```bash
git clone https://github.com/Haidra-Org/horde-worker-reGen.git
cd horde-worker-reGen
```

### AMD GPUs

AMD support is experimental, and **Linux-only** for now:

- Use `horde-bridge-rocm.sh` and `update-runtime-rocm.sh` in place of the standard versions.
- [WSL support](README_advanced.md#advanced-users-amd-rocm-inside-windows-wsl) is highly experimental.
- Join the [AMD discussion on Discord](https://discord.com/channels/781145214752129095/1076124012305993768) if you're interested in trying.

### DirectML

**Experimental** Support for DirectML has been added. See [Running on DirectML](README_advanced.md#advanced-users-running-on-directml) for more information and further instructions. You can now follow this guide using  `update-runtime-directml.cmd` and `horde-bridge-directml.cmd` where appropriate. Please note that DirectML is several times slower than *ANY* other methods of running the worker.

## Configuration

### Basic Settings

1. Copy `bridgeData_template.yaml` to `bridgeData.yaml`.
2. Edit `bridgeData.yaml` following the instructions inside.
3. Set a unique `dreamer_name`
  - If the name is already taken, you'll get a "Wrong Credentials" error. The name must be unique across the entire horde network.

### Suggested Settings

Tailor settings to your GPU, following these pointers:

- **24GB+ VRAM** (e.g. 3090, 4090):

  ```yaml
  - queue_size: 1 # <32GB RAM: 0, 32GB: 1, >32GB: 2
  - safety_on_gpu: true
  - high_memory_mode: true
  - high_performance_mode: true
  - unload_models_from_vram_often: false
  - max_threads: 1 # 2 is often viable for xx90 cards
  - post_process_job_overlap: true
  - queue_size: 2 # Set to 1 if max_threads: 2
  - max_power: 64 # Reduce if max_threads: 2
  - max_batch: 8 # Increase if max_threads: 1, decrease if max_threads: 2
  - allow_sdxl_controlnet: true
  ```

- **12-16GB VRAM** (e.g. 3080 Ti, 4070 Ti, 4080):

  ```yaml  
  - queue_size: 1 # <32GB RAM: 0, 32GB: 1, >32GB: 2
  - safety_on_gpu: true # Consider false if using Cascade/Flux
  - moderate_performance_mode: true
  - unload_models_from_vram_often: false
  - max_threads: 1
  - max_power: 50
  - max_batch: 4 # Or higher
  ```

- **8-10GB VRAM** (e.g. 2080, 3060, 4060, 4060 Ti):

  ```yaml
  - queue_size: 1 # <32GB RAM: 0, 32GB: 1, >32GB: 2
  - safety_on_gpu: false
  - max_threads: 1
  - max_power: 32 # No higher
  - max_batch: 4 # No higher
  - allow_post_processing: false # If using SDXL/Flux, else can be true
  - allow_sdxl_controlnet: false
  ```

  - Also minimize other VRAM-consuming apps while the worker runs.

- **Lower-end GPUs / Under-performing Workers**:
  - `extra_slow_worker: true` gives more time per job, but users must opt-in. Only use if <0.3 MPS/S or <3000 kudos/hr consistently with correct config.
  - `limit_max_steps: true` caps total steps per job based on model.
  - `preload_timeout: 120` allows longer model load times. Avoid misusing to prevent kudos loss or maintenance mode.

- **Systems with less than 32GB of System RAM**:
  - Be sure to only run SD15 models and queue_size: 0.
    - Set `load_large_models: false`
    - To your `models_to_skip` add `ALL SDXL`, `ALL SD21`, and the 'unpruned' models (see config) to prevent running out of memory

### Important Notes

- Use an SSD, especially for multiple models. HDDs should offer one model only, loading within 60s.
- Configure 8GB (preferably 16GB+) of swap space, even on Linux.
- Keep `threads` ≤2 unless using a 48GB+ VRAM data center GPU.
- Worker RAM usage scales with `queue_size`. Use 1 for <32GB RAM, and optimize further for <16GB.
- SDXL needs ~9GB free RAM consistently (32GB+ total recommended).
- Flux and Stable Cascade need ~20GB free RAM consistently (48GB+ total recommended).
- Disable sleep/power-saving modes while the worker runs.

## Running the Worker

### Starting

> **Note**: The worker is resource-intensive. Avoid gaming or other heavy tasks while it runs. Turn it off or limit to small models at reduced settings if needed.

1. Install the worker as described in the [Installation](#installation) section.
2. Run `horde-bridge.cmd` (Windows) or `horde-bridge.sh` (Linux).
   - **AMD**: Use `horde-bridge-rocm` versions.

### Stopping

- Press `Ctrl+C` in the worker's terminal.
- It will finish current jobs before exiting.

### Monitoring

Watch the terminal for progress, completed jobs, kudos earned, stats, and errors.

Detailed logs are in the `logs` directory:

- `bridge*.log`: All info
  - `bridge.log` is the main window
  - `bridge_n.log` is process-specific (`n` is the process number)
- `trace*.log`: Errors and warnings only
  - `trace.log` is the main window
  - `trace_n.log` is process-specific

### Multiple GPUs

> **Future versions won't need multiple worker instances**

For now, start a separate worker per GPU.

On Linux, specify the GPU for each instance:

```bash
CUDA_VISIBLE_DEVICES=0 ./horde-bridge.sh -n "Instance 1"
CUDA_VISIBLE_DEVICES=1 ./horde-bridge.sh -n "Instance 2"
```

**Warning**: High RAM (32-64GB+) is needed for multiple workers. `queue_size` and `max_threads` greatly impact RAM per worker.

## Updating

The worker is constantly improving. Follow development and get update notifications in our [Discord](https://discord.gg/3DxrhksKzn).

Script names below assume Windows (`.cmd`) and NVIDIA. For Linux use `.sh`, for AMD use `-rocm` versions.

### Updating the Worker

1. Stop the worker with `Ctrl+C`.
2. Update the files:
   - If you used `git clone`:
     - Open a terminal in the worker folder
     - Run `git pull`
   - If you used the zip download:
     - Delete the old `horde_worker_regen` folder
     - Download the [latest zip](https://github.com/db0/horde-worker-reGen/archive/refs/heads/main.zip)
     - Extract to the original location, overwriting existing files
3. Continue with [Updating the Runtime](#updating-the-runtime) below.

### Updating the Runtime

> **Warning**: Some antivirus software (e.g. Avast) may interfere with the update. If you get `CRYPT_E_NO_REVOCATION_CHECK` errors, disable antivirus, retry, then re-enable.

4. Run `update-runtime` for your OS to update dependencies.
   - Not all updates require this, but run it if unsure
   - **Advanced users**: see [README_advanced.md](README_advanced.md) for manual options
5. [Start the worker](#starting) again

## Custom Models

Serving custom models not in our reference requires the `customizer` role. Request it on Discord.

With the role:

1. Download your model files locally.
2. Reference them in `bridgeData.yaml`:

   ```yaml
   custom_models:
     - name: My Custom Model
       baseline: stable_diffusion_xl
       filepath: /path/to/model/file.safetensors
   ```

3. Add the model `name` to your `models_to_load` list.

If set up correctly, `custom_models.json` will appear in the worker directory on startup.

Notes:

- Custom model names can't match our existing model names
- The horde will treat them as SD 1.5 for kudos rewards and safety checks

## Docker

Docker images are at <https://hub.docker.com/r/tazlin/horde-worker-regen/tags>.

Detailed guide: [Dockerfiles/README.md](Dockerfiles/README.md)

Manual worker setup: [README_advanced.md](README_advanced.md)

## Support & Troubleshooting

Check the [#local-workers Discord channel](https://discord.com/channels/781145214752129095/1076124012305993768) for the latest info and community support.

Common issues and fixes:

- **Download failures**: Check disk space and internet connection.
- **Job timeouts**:
  - Remove large models (Flux, Cascade, SDXL)
  - Lower `max_power`
  - Disable `allow_post_processing`, `allow_controlnet`, `allow_sdxl_controlnet`, and/or `allow_lora`
- **Out of memory**: Decrease `max_threads`, `max_batch`, or `queue_size` to reduce VRAM/RAM use. Close other intensive programs.
- **I have less kudos than I expect**: As a new user, 50% of your job reward kudos and 100% of uptime kudos are held in escrow until you become trusted after ~1 week of worker uptime. You'll then receive the escrowed kudos and earn full rewards immediately going forward.
- **My worker is in [maintenance mode](https://github.com/Haidra-Org/haidra-assets/blob/main/docs/definitions.md#maintenance)**: You can log into [artbot here](https://tinybots.net/artbot/settings) and use the [manage workers](https://tinybots.net/artbot/settings?panel=workers) page **with the worker on** and click "unpause" to take your worker out of maintenance mode.
  - **Note**: Workers are put into maintenance mode automatically by the server when the worker is failing to perform fast enough or if it is reporting that it failed too many jobs. You should investigate the [logs](logs/README.md) (search for "ERROR") to see what led to the issue. You can also [open an issue](https://github.com/Haidra-Org/horde-worker-reGen/issues) or ask in the [#local-workers channel](https://discord.com/channels/781145214752129095/1076124012305993768) in our [Discord](https://discord.gg/3DxrhksKzn).

[Open an issue](https://github.com/Haidra-Org/horde-worker-reGen/issues) to report bugs or request features. We appreciate your help!

## Model Usage & Licenses

Many bundled models use the [CreativeML OpenRAIL License](https://huggingface.co/spaces/CompVis/stable-diffusion-license). Please review it before use.
