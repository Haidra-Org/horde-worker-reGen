## If you want the latest information or have questions, come to the [#local-workers](https://discord.com/channels/781145214752129095/1076124012305993768) channel in the [official discord](https://discord.gg/3DxrhksKzn)

> **Most users should follow the [Quick Start](README.md#quick-start) in the main README instead.** This document is for advanced users who want manual control over their environment.

**Some important details you should know before you start:**

- When submitting debug information **do not publish `.log` files in the server channels — send them to tazlin directly** as your API key could appear in them.
- Workers interested in logs should note that there is a main log (`bridge.log`) and a log for each subprocess. `bridge_0.log` is the safety process; `bridge_1.log`, `bridge_2.log`, etc. are inference processes.
- You can `Get-Content bridge_1.log -Wait` (Windows) or `less +F bridge_1.log` (Linux) to follow a log in real time.

## Advanced users, AMD ROCm inside Windows WSL

### Caveats and Limitations
>
> WSL will probably be slower than a native Linux System. Unless you have a lot of RAM, you might also run into memory issues. It might be neccessary to increase WSL memory limits or configure SWAP like described here: <https://learn.microsoft.com/en-us/windows/wsl/wsl-config>

### System setup

- Make sure your Windows OS and AMD drivers are up to date.
- You need to enable and install WSL on your system. Open a command prompt with Administrative privileges (search for cmd, then click "Run as Administrator")

### Ubuntu ROCm install

- First we need to update the image, then install ROCm. All these actions require root privileges, so switch to root for now and enter your password:

```bash
sudo su
```

- Now update the system and install a few tools:

```bash
apt update && apt full-upgrade -y && apt autopurge -y
apt install -y curl git nano wget
```

- Now we can install ROCm. Command 3 will take a while to download and install everything:

```bash
wget -r -nd -np -A 'amdgpu-install*all.deb' "https://repo.radeon.com/amdgpu-install/6.1.3/ubuntu/jammy/"
apt-get install -y ./amdgpu-install*all.deb
amdgpu-install -y --usecase=rocm,wsl --no-dkms
```

- We can now check whether ROCm was installed successfully with the `rocminfo` command.

```bash
rocminfo
```

- It should return something like:

```
WSL environment detected.
=====================
HSA System Attributes
=====================
Runtime Version:         1.1
Runtime Ext Version:     1.6
System Timestamp Freq.:  1000.000000MHz
Sig. Max Wait Duration:  18446744073709551615 (0xFFFFFFFFFFFFFFFF) (timestamp count)
Machine Model:           LARGE
System Endianness:       LITTLE
Mwaitx:                  DISABLED
DMAbuf Support:          NO

==========
HSA Agents
==========
...
```

- Now type `exit` to leave the root shell

```bash
exit
```

### Installing the worker

- From here the steps are the same as running on any other Linux system with AMD: see [Quick Start](README.md#quick-start) and use the ROCm scripts.

## Advanced users, local install

### Simple usage

### Prerequisites

- Install [git](https://git-scm.com/) in your system.

- Install CUDA/RoCM if you haven't already.
- Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (or use the `update-runtime` scripts which download it automatically).
- We **strongly recommend** you configure at least 8gb (preferably 16gb+) of memory swap space. This recommendation applies to linux too.
- Clone the worker to your system
   `git clone https://github.com/Haidra-Org/horde-worker-reGen.git`

### Install dependencies with uv

- CUDA: `uv sync --locked --extra cu128`
- ROCm: `uv sync --locked --extra rocm`
- DirectML: `uv sync --locked --extra directml`
- CPU only: `uv sync --locked --extra cpu`

### Run worker

- Set your config now, copying `bridgeData_template.yaml` to `bridgeData.yaml`, being sure to set an API key and worker name at a minimum
- `uv run python download_models.py` (**critical - must be run first every time**)
- `uv run python run_worker.py` (to start working)

Pressing control-c will stop the worker but will first have the worker complete any jobs in progress before ending. Please try and avoid hard killing it unless you are seeing many major errors. You can force kill by repeatedly pressing control+c or doing a SIGKILL.

### Important note if manually managing your environment

- You should run `uv sync --locked --extra cu128` (or the relevant extra) every time you `git pull` to keep dependencies in sync.

## Advanced users, running on directml

### DirectML Caveats and Limitations
>
> DirectML is anywhere from 3x to 10x slower than other methods and will max out your VRAM at 100%. It is also not compatible with Flux.1. If you can use *ANY* other method, do that instead. Unless you have a lot of RAM, you might also run into memory issues. You should limit yourself to the smallest models and easiest jobs, even if you have a decent GPU in theory.

### DirectML Prerequisites

- Install [git](https://git-scm.com/) in your system.

- Make sure your Windows OS and GPU drivers are up to date.

### General Use

- The first steps are identical to the normal process: follow the [Quick Start](README.md#quick-start) to install and [configure](README.md#configuration) the worker. Remember to stick with the lowest-end settings.

- Now run `update-runtime-directml.cmd` to install DirectML dependencies. Use this script for future updates as well.

- To run the worker, use `horde-bridge-directml.cmd`.

For more direct support, join the [discord discussion](https://discord.com/channels/781145214752129095/1076124012305993768) in the [official discord](https://discord.gg/3DxrhksKzn).

## Advanced users, container install

You can find the docker images at <https://hub.docker.com/r/tazlin/horde-worker-regen/tags>.

See [Dockerfiles/README.md](Dockerfiles/README.md) for a detailed guide on the supported docker functionality.
