##  If you want the latest information or have questions, come to the [#local-workers](https://discord.com/channels/781145214752129095/1076124012305993768) channel in the [official discord](https://discord.gg/3DxrhksKzn).


**Some important details you should know before you start:**

> See [this important info](README.md/#important-info) first.

- When submitting debug information **do not publish `.log` files in the server channels - send them to tazlin directly** as we cannot guarantee that your API key would not be in it (though, this warning should relax over time).
- Workers especially interested in logs should note that there is a main log (`bridge.log`) and a log for each subprocess. `bridge_0.log` is the safety process, and all ones after that (`bridge_1.log`, `brige_2.log`, etc) are inference processes.
- You could `Get-Content bridge_1.log -Wait` each log on windows , or `less +F bridge_1.log` on linux to monitor these logs.
- **The worker does not download models on worker start** for the moment (this will change.) You can download all models configured in your bridge data by invoking `python download_models.py`.


## Advanced users, AMD ROCm inside Windows WSL:

### Caveats and Limitations:
> WSL will probably be slower than a native Linux System. Unless you have a lot of RAM, you might also run into memory issues. It might be neccessary to increase WSL memory limits or configure SWAP like described here: https://learn.microsoft.com/en-us/windows/wsl/wsl-config

### System setup:
* Make sure your Windows OS and AMD drivers are up to date.
* You need to enable and install WSL on your system. Open a command prompt with Administrative privileges (search for cmd, then click "Run as Administrator")
* Type the following to download and enable WSL and install the Ubuntu 22.04 image:
  - If that command throws an error about WSL not being installed/enabled, you might need to run just `wsl --install` before being able install a specific distribution.
```
wsl --install -d Ubuntu-22.04
```
* If you have previously used Ubuntu-22.04 WSL, please reset the image (Note: this will delete the data inside the WSL image, make sure it's saved elsewhere):
```
wsl --unregister Ubuntu-22.04
```
* When the terminal asks you for a "unix username" type in a simple username. It will then ask for a password. Type in the password you want to use, press enter to confirm and repeat. It will not show any output, but your key presses are still registered.
* To open your Ubuntu image after closing the terminal window you can search for `Ubuntu 22.04` in the Start Menu, or open a Termial and enter the command `wsl -d Ubuntu-22.04`

### Ubuntu ROCm install:
* First we need to update the image, then install ROCm. All these actions require root privileges, so switch to root for now and enter your password:
```
sudo su
```
* Now update the system and install a few tools:
```
apt update && apt full-upgrade -y && apt autopurge -y
apt install -y curl git nano wget
```
* Now we can install ROCm. Command 3 will take a while to download and install everything:
```
wget -r -nd -np -A 'amdgpu-install*all.deb' "https://repo.radeon.com/amdgpu-install/6.1.3/ubuntu/jammy/"
apt-get install -y ./amdgpu-install*all.deb
amdgpu-install -y --usecase=rocm,wsl --no-dkms
```
* We can now check whether ROCm was installed successfully with the `rocminfo` command.
```
rocminfo
```
* It should return something like:
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
* Now type `exit` to leave the root shell
```
exit
```

### Installing the worker:
* From here steps are the same as running on any other Linux System with AMD: [Installing](README.md/#linux)


## Advanced users, local install:

### Simple usage

### Prerequisites
* Install [git](https://git-scm.com/) in your system.
* Install CUDA/RoCM if you haven't already.
* Install Python 3.10 or 3.11.
  * If using the official python installer **and** you do not already regularly already use python, be sure to check the box that says `Add python.exe to PATH` at the first screen.
* We **strongly recommend** you configure at least 8gb (preferably 16gb+) of memory swap space. This recommendation applies to linux too.
* Clone the worker to your system
   `git clone https://github.com/Haidra-Org/horde-worker-reGen.git`

### Setup venv
- `python -m venv regen` (only needed the first time you follow these instructions)
  - (certain windows setups) `py -3.11 -m venv regen`
- (windows) `regen\Scripts\activate.bat` for cmd or `regen\Scripts\Activate.ps1` for power shell
- (linux) `source regen/bin/activate`
- **Important**: You should now see `(regen)` prepended in your shell. If you do not, try again or ask for help.

### Get worker files and install dependencies
- `git clone https://github.com/Haidra-Org/horde-worker-reGen.git`
- `cd .\horde-worker-reGen\`
- Install the requirements:
  - CUDA: `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124`
  - RoCM: `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/rocm6.2`

### Run worker
- Set your config now, copying `bridgeData_template.yaml` to `bridgeData.yaml`, being sure to set an API key and worker name at a minimum
- `python download_models.py` (**critical - must be run first every time**)
- `python run_worker.py` (to start working)

Pressing control-c will stop the worker but will first have the worker complete any jobs in progress before ending. Please try and avoid hard killing it unless you are seeing many major errors. You can force kill by repeatedly pressing control+c or doing a SIGKILL.

### Important note if manually manage your venvs
- You should be running `python -m pip install -r requirements.txt -U https://download.pytorch.org/whl/cu124` every time you `git pull`. (Use `/whl/rocm6.2` instead if applicable)


## Advanced users, running on directml

### Caveats and Limitations:
> DirectML is anywhere from 3x to 10x slower than other methods and will max out your VRAM at 100%. It is also not compatible with Flux.1. If you can use *ANY* other method, do that instead. Unless you have a lot of RAM, you might also run into memory issues. You should limit yourself to the smallest models and easiest jobs, even if you have a decent GPU in theory.

### Prerequisites
* Install [git](https://git-scm.com/) in your system.
* Make sure your Windows OS and GPU drivers are up to date.

### General Use:
- The first steps are identical to the normal process: [install](README.md/#installing) and [configure](README.md/#configure) the worker. Remember to stick with the lowest end settings.

- Now [update](README.md/#updating) the runtime, but make sure to use the `update-runtime-directml.cmd` script. Follow the linked instructions and run this script for future updates as well.

- To run the worker again follow [starting/stopping](README.md/#startingstopping), making sure to use the `horde-bridge-directml.cmd` script.

For more direct support, join the [discord discussion](https://discord.com/channels/781145214752129095/1076124012305993768) in the [official discord](https://discord.gg/3DxrhksKzn).


## Advanced users, container install

You can find the docker images at https://hub.docker.com/r/tazlin/horde-worker-regen/tags.

See [Dockerfiles/README.md](Dockerfiles/README.md) for a detailed guide on the supported docker functionality.
