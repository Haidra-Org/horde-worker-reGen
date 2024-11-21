# AI Horde Worker reGen

Welcome to the AI Horde, a free and open decentralized platform for collaborative AI! The AI Horde enables people from around the world to contribute their GPU power to generate images, text, and more. By running a worker on your local machine, you can earn [kudos](https://github.com/Haidra-Org/haidra-assets/blob/main/docs/kudos.md) which give you priority when making your own requests to the horde.

A worker is a piece of software that handles jobs from the AI Horde, such as generating an image from a text prompt. When your worker successfully completes a job, you are rewarded with kudos. The more kudos you have, the faster your own requests will be processed.

Not only does running a worker earn you kudos, it also helps support the AI Horde ecosystem and puts your GPU to work during idle cycles. Whether you're an artist looking to generate custom assets, a developer needing to process images at scale, or just someone excited about democratizing AI, the Horde has something to offer.

## Table of Contents

- [AI Horde Worker reGen](#ai-horde-worker-regen)
  - [Table of Contents](#table-of-contents)
  - [Installing](#installing)
    - [Windows](#windows)
      - [Using git (recommended)](#using-git-recommended)
      - [Without git](#without-git)
    - [Linux](#linux)
    - [AMD](#amd)
  - [Basic Usage](#basic-usage)
    - [Configure](#configure)
      - [Suggested settings](#suggested-settings)
        - [Important Notes](#important-notes)
  - [Updating](#updating)
    - [git method](#git-method)
      - [zip method](#zip-method)
    - [Updating runtime](#updating-runtime)
  - [Starting/stopping](#startingstopping)
    - [Starting the worker](#starting-the-worker)
    - [Stopping the worker](#stopping-the-worker)
    - [Monitoring](#monitoring)
    - [Running with multiple GPUs](#running-with-multiple-gpus)
  - [Custom Models](#custom-models)
  - [Docker](#docker)
  - [Support](#support)
  - [Model Usage](#model-usage)

## Installing

If you haven't already, go to [AI Horde and register an account](https://aihorde.net/register), then store your API key somewhere secure. Treat your API key like a password. You will need it later in these instructions. This will allow your worker to gather kudos for your account.

### Windows

#### Using git (recommended)

Use these instructions if you have installed [git for windows](https://gitforwindows.org/).

This option is recommended as it will make keeping your repository up to date much easier.

1. Open `powershell` (also referred to as terminal) or `cmd` from the start menu.

2. Using `cd`, navigate to a folder that you want the worker installed in.

   - Note that the folder you are in will create a folder named `horde-worker-reGen`. This folder should not exist before you run the following commands.
   - If you want it to be installed in `C:\horde\`, run the following:

     ```cmd
     cd C:\horde
     ```

     If the `horde` folder doesn't exist:

     ```cmd
     cd C:\
     mkdir horde
     cd C:\horde
     ```

   - If you are using `cmd` and wish to install on a different drive, include the `/d` option, as so:

     ```cmd
     cd /d G:\horde
     ```

3. Run the following commands within the folder chosen (the folder `horde` if using the example above)

   ```cmd
   git clone https://github.com/Haidra-Org/horde-worker-reGen.git
   cd horde-worker-reGen
   ```

4. Continue with the [Basic Usage](#basic-usage) instructions

#### Without git

Use these instructions if you do not have git for windows and do not want to install it. These instructions make updating the worker a bit more difficult down the line.

1. Download [the zipped version](https://github.com/Haidra-Org/horde-worker-reGen/archive/refs/heads/main.zip)
2. Extract it to any folder of your choice
3. Continue with the [Basic Usage](#basic-usage) instructions

### Linux

This assumes you have git installed

Open a bash terminal and run these commands (just copy-paste them all together)

```bash
git clone https://github.com/Haidra-Org/horde-worker-reGen.git
cd horde-worker-reGen
```

Continue with the [Basic Usage](#basic-usage) instructions

### AMD

**AMD** now has been shown to have better support but for **linux machines only** - it's best to have linux installed directly on your machine. [WSL support](README_advanced.md#advanced-users-amd-rocm-inside-windows-wsl) is highly experimental. You can now follow this guide using  `horde-bridge-rocm.sh` and `update-runtime-rocm.sh` where appropriate.

If you are willing to try with your AMD card, join the [discord discussion](https://discord.com/channels/781145214752129095/1076124012305993768) in the [official discord](https://discord.gg/3DxrhksKzn).

## Basic Usage

### Configure

1. Make a copy of `bridgeData_template.yaml` to `bridgeData.yaml`
2. Edit `bridgeData.yaml` and follow the instructions within to fill in your details.

#### Suggested settings

##### Important Notes

To ensure a smooth experience running a worker, please keep the following in mind:

- For optimal performance, we strongly recommend using an SSD, especially if you plan to offer more than one model. If using an HDD, you should only offer one model and ensure it can load within 60s.
- We **strongly recommend** you configure at least 8gb (preferably 16gb+) of memory swap space. This recommendation applies to linux too.
- To maintain stability, keep `threads` at 2 or lower unless you have a data-center grade card with 48GB+ VRAM.
- Your worker's memory usage will increase up to the `queue_size` specified in the config. If you have less than 32GB of system RAM, stick with a `queue_size` of 1. For less than 16GB RAM, additional optimizations are needed (detailed below).
- Running the SDXL models requires around 9GB of consistently free system RAM. 32GB+ installed is recommended.
- Flux and Stable Cascade require around 20GB of consistently free system RAM. 48GB+ installed is recommended.

Models are loaded as needed and just-in-time. You can offer as many models as you want **provided you have an SSD, at least 32gb of ram, and at least 8gb of VRAM (see [Important Notes](#important-notes)**). Workers with HDDs are not recommended at this time but those with HDDs should run exactly 1 model. A typical SD1.5 model is around 2gb each, while a typical SDXL model is around 7gb each. Offering `all` models is currently around 700gb total and we commit to keeping that number below 1TB with any future changes.

> **Note**: We suggest you disable any 'sleep' or reduced power modes for your system while the worker is running.

- If you have a **24gb+ vram card** (e.g., 4090, 3090):

  ```yaml
  - safety_on_gpu: true
  - high_performance_mode: true
  - post_process_job_overlap: true
  - unload_models_from_vram_often: false
  - max_threads: 1 # If you have Flux/Cascade loaded, otherwise 2 max
  - queue_size: 2 # You can set to 3 if you have 64GB or more of RAM
  - max_batch: 8 # or higher
  ```

- If you have a **12gb - 16gb card** (e.g., 3080 TI, 4070, 4080/4080 Super):

  ```yaml
  - safety_on_gpu: true # Consider setting to `false` if offering Cascade or Flux
  - moderate_performance_mode: true
  - unload_models_from_vram_often: false
  - max_threads: 1
  - max_batch: 4 # or higher
  ```

- If you have an **8gb-10gb vram card**(1080, 2080, 3060, 4060/4060 TI):

  ```yaml
  - queue_size: 1 # max **or** only offer flux
  - safety_on_gpu: false
  - max_threads: 1
  - max_power: 32 # no higher than 32
  - max_batch: 4 # no higher than 4
  - allow_post_processing: false # If offering SDXL or Flux, otherwise you may set to true
  - allow_sdxl_controlnet: false
  ```

  - Be sure to shut every single VRAM consuming application you can and do not use the computer with the worker running for any purpose.

- Workers which have **low end cards or have low performance for other reasons**:
  - `- extra_slow_worker: true`
    - gives you considerably more time to finish job, but requests will not go to your worker unless the requester opts-in (even anon users do not use extra_slow_workers by default). You should only consider using this if you have historically had less than 0.3 MPS/S or less than 3000 kudos/hr consistently **and** you are sure the worker is otherwise configured correctly.
  - `- limit_max_steps: true`
    - reduces the maximum total number of steps in a single job you will receive based on the model baseline.
  - `- preload_timeout: 120`
    - gives you more time to load models off disk. **Note**: Abusing this value can lead to a major loss of kudos and may also lead to maintainance mode, even with `extra_slow_worker: true`.

## Updating

The AI Horde workers are under constant improvement. You can follow progress [in our discord](https://discord.gg/3DxrhksKzn) and get notifications about updates there. If you are interested in receiving notifications for worker updates or betas, go to the [#get-roles channel](https://discord.com/channels/781145214752129095/977498954616954890) and get the appropriate role(s).

The below instructions refers to `horde-bridge` or `update-runtime`. Depending on your OS, append `.cmd` for windows, or `.sh` for linux

- For example, `horde-bridge.cmd` and `update-runtime.cmd` for windows with a NVIDIA card.
- If you have an **AMD** card and you are on linux you should use `horde-bridge-rocm.sh` and `update-runtime-rocm.sh` where appropriate.
  - All Windows AMD users should use WSL or [Docker](#docker).
To update:

1. Shut down your worker by pressing `Ctrl+C` once and waiting for the worker to stop.

2. Update this repo using the appropriate method:

You can double click the provided script files below from a file explorer or run it from a terminal like `bash`, `cmd` depending on your OS. The latter option will allow you to **see errors in case of a crash**, so it's recommended.

### git method

Use this approach if you cloned the original repository using `git clone`

1. Open a `bash`, `cmd`, or `powershell` terminal depending on your OS
2. Navigate to the folder you have the AI Horde Worker repository installed if you're not already there.
3. Run `git pull`

See [Updating runtime](#updating-runtime)

#### zip method

Use this approach if you downloaded the git repository as a zip file and extracted it somewhere.

1. Delete the `horde_worker_regen/` directory from your folder
2. Download the [repository from GitHub as a zip file](https://github.com/db0/horde-worker-reGen/archive/refs/heads/main.zip)
3. Extract its contents into the same folder you have the AI Horde Worker repository installed, overwriting any existing files

See [Updating runtime](#updating-runtime)

### Updating runtime
>
> **Warning**: Certain antiviruses (including Avast) have been reported to interfere with the install. If you get the error `curl: (35) schannel: next InitializeSecurityContext failed: CRYPT_E_NO_REVOCATION_CHECK` when running this file, disable your antivirus, run the file again, and then re-enable your antivirus.

4. Run the `update-runtime` script for your OS. This will update all dependencies if required.
   - Some updates may not require this and the update notification will tell you if this is the case.
   - When in doubt, you should run it anyway.
   - **Advanced users**: If you do not want to use mamba or you are comfortable with python/venvs, see [README_advanced.md](README_advanced.md).

5. Continue with [Starting/stopping](#startingstopping) instructions below

## Starting/stopping

### Starting the worker

> **Note**: The worker is a very system and GPU intensive program. You should not play video games or do other intensive tasks (such as image/video editing) whenever possible. If you would like to engage in those activities, either turn off the worker or configure it to use only small models at limited settings and closely watch your system monitor.

1. If it's the first time you're installing, or when updates are required, see [Updating](#updating) for instructions.

2. Run `horde-bridge` (.cmd for Windows, .sh for Linux).
   - **AMD**: Use the `horde-bridge-rocm` versions of the file.

### Stopping the worker

- In the terminal in which it's running, press `Ctrl+C` together.
- The worker will finish the current jobs before exiting.

### Monitoring

While the worker is running, you can monitor its progress directly in the terminal. Look for logs indicating successful job completion, kudos earned, performance stats, and any errors.

For more detailed monitoring, check out the `logs` directory which contains daily log files.

- All info appears in the `bridge*.log` files.
  - `bridge.log` is the main window you see pop up.
  - `bridge_n.log` corresponds to each process that appears in the main log file. "Process 1" would be `bridge_1.log`.
- A list of only errors/warnings will appear in the `trace*.log` files.
  - `trace.log` is the main window you see pop up.
  - `trace_n.log` corresponds to each process that appears in the main log file. "Process 1" would be `trace_1.log`.

### Running with multiple GPUs

> In the future you will not need to run multiple worker instances

To use multiple GPUs, each has to start their own instance. For Linux, you just need to limit the run to a specific card:

```bash
CUDA_VISIBLE_DEVICES=0 ./horde-bridge.sh -n "My awesome instance #1"
CUDA_VISIBLE_DEVICES=1 ./horde-bridge.sh -n "My awesome instance #2"
```

etc.

**Be warned** that you will need a very high (32-64gb+) amount of system RAM depending on your settings. `queue_size` and `max_threads` increase the amount of RAM required per worker substantially.

## Custom Models

You can host your own image models on the horde which are not available in our model reference, but this process is a bit more complex.

To start with, you need to manually request the `customizer` role from the horde team. You can ask for it in the discord channel. This is a manually assigned role to prevent abuse of this feature.

Once you have the customizer role:

1. Download the model files you want to host. Place them in any location on your system.

2. Point your worker to their location and provide some information about them. In your `bridgeData.yaml`, add lines like the following:

   ```yaml
   custom_models:
     - name: Movable figure model XL
       baseline: stable_diffusion_xl
       filepath: /home/db0/projects/CUSTOM_MODELS/PVCStyleModelMovable_beta25Realistic.safetensors
   ```

3. Add the same "name" to your `models_to_load`.

If everything was set up correctly, you should now see a `custom_models.json` in your worker directory after the worker starts, and the model should be offered by your worker.

Note that:

- You cannot serve custom models with the same name as any of our regular models
- The horde doesn't know your model, so it will treat it as a SD 1.5 model for kudos rewards and cannot warn people using the wrong parameters such as `clip_skip`

## Docker

You can find the docker images at https://hub.docker.com/r/tazlin/horde-worker-regen/tags.

See [Dockerfiles/README.md](Dockerfiles/README.md) for a detailed guide on the supported docker functionality.

See also [README_advanced.md](README_advanced.md) for further details on running the worker manually.

## Support

For the latest info and troubleshooting help, check out the [#local-workers channel](https://discord.com/channels/781145214752129095/1076124012305993768) in our [Discord](https://discord.gg/3DxrhksKzn). The community is always happy to lend a hand!

Some common issues and their solutions:

- **Models failing to download**: Ensure you have enough free disk space and a stable internet connection.
- **Jobs timing out**:
  - Consider removing large models (Flux, cascade, SDXL).
  - Reduce your `max_power`.
  - Disable `allow_post_processing`, `allow_controlnet`, `allow_sdxl_controlnet`, and/or `allow_lora`.
- **Out of memory errors**: Decrease `max_threads`, `max_batch`, or `queue_size` in your config (in that order of preference) to reduce VRAM/RAM usage. Avoid running other intensive programs while the worker is active.

If you encounter a bug or have a feature request, please [open an issue](https://github.com/Haidra-Org/horde-worker-reGen/issues) on the repo. We appreciate your contributions!

## Model Usage

Many models in this project use the CreativeML OpenRAIL License.  [Please read the full license here.](https://huggingface.co/spaces/CompVis/stable-diffusion-license)
