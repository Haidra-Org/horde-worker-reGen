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

### Step 1: Install the Windows Build Tools & SDK (On your Windows Host)
Before touching Linux, you need the Microsoft development headers so the Linux compiler can understand how to talk to Windows.

* Download the **Visual Studio Build Tools** from Microsoft's developer website.
* Run the installer and select the **"Desktop development with C++"** workload.
* In the installation details panel on the right, ensure that **Windows 11 SDK (10.0.26100.0)** is checked, then click Install.
* Once installed, confirm the headers exist by navigating your Windows File Explorer to `C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\`. We will need this path in Step 5.

### Step 2: System setup
* Make sure your Windows OS and AMD drivers are up to date.
* You need to enable and install WSL on your system. Open a command prompt with Administrative privileges (search for cmd, then click "Run as Administrator")
* Type the following to download and enable WSL and install the Ubuntu 24.04 image:
  - If that command throws an error about WSL not being installed/enabled, you might need to run just `wsl --install` before being able install a specific distribution.
```
wsl --install -d Ubuntu-24.04
```
* If you have previously used Ubuntu-24.04 WSL, please reset the image (Note: this will delete the data inside the WSL image, make sure it's saved elsewhere):
```
wsl --unregister Ubuntu-24.04
```
* When the terminal asks you for a "unix username" type in a simple username. It will then ask for a password. Type in the password you want to use, press enter to confirm and repeat. It will not show any output, but your key presses are still registered.
* To open your Ubuntu image after closing the terminal window you can search for `Ubuntu 24.04` in the Start Menu, or open a Termial and enter the command `wsl -d Ubuntu-24.04`

### Step 3: Install Ubuntu 24.04 Build Dependencies
Boot up your Ubuntu 24.04 WSL terminal and install the base tools needed to compile code:

```bash
sudo apt update && sudo apt install -y build-essential cmake git ca-certificates wget gpg
```

### Step 4: Add the ROCm Repo, PIN IT, and Install
We will add the repository, set the priority pin so Ubuntu doesn't get confused, and then install. Run these block by block:

```bash
# 1. Download and add the AMD GPG key
sudo mkdir --parents --mode=0755 /etc/apt/keyrings
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null

# 2. Add the ROCm repository specifically for Ubuntu 24.04 (Noble)
sudo tee /etc/apt/sources.list.d/rocm.list << EOF
deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/latest noble main
EOF

# 3. Set the Apt priority pin to prefer AMD's repo over Ubuntu's defaults
echo -e "Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600" | sudo tee /etc/apt/preferences.d/rocm-pin-600

# 4. Update your package list and install the ROCm stack
sudo apt update
sudo apt install -y rocm rocm-dev
```
*(Grab a coffee, step 4 will download a few gigabytes of data).*

### Step 5: Clone and Build librocdxg
Now we compile the bridge library. We will point the Linux compiler directly to your mounted Windows C: drive so it can read the Windows SDK headers.

```bash
# 1. Clone the repository to your local WSL
git clone https://github.com/ROCm/librocdxg.git
cd librocdxg

# 2. Set the Windows SDK path
# IMPORTANT: Change "10.0.26100.0" below to match the folder version you found in Step 1!
export win_sdk='/mnt/c/Program Files (x86)/Windows Kits/10/Include/10.0.26100.0/'

# 3. Configure the build environment
mkdir -p build
cd build
cmake .. -DWIN_SDK="${win_sdk}/shared"

# 4. Compile and install the library
make
sudo make install
```

### Step 6: Enable Detection and Verify
Finally, we need to set the environment variables so ROCm knows to route through the newly built DXG library.

```bash
# 1. Add the environment variables to your bash profile
echo 'export HSA_ENABLE_DXG_DETECTION=1' >> ~/.bashrc
echo 'export PATH=/opt/rocm/bin:$PATH' >> ~/.bashrc

# 2. Reload your profile
source ~/.bashrc
```

Once that is all done, run the moment of truth:

```bash
rocminfo
```

* It should return something like:
```text
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
  - CUDA: `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130`
  - RoCM: `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/rocm6.2`

### Run worker
- Set your config now, copying `bridgeData_template.yaml` to `bridgeData.yaml`, being sure to set an API key and worker name at a minimum
- `python download_models.py` (**critical - must be run first every time**)
- `python run_worker.py` (to start working)

Pressing control-c will stop the worker but will first have the worker complete any jobs in progress before ending. Please try and avoid hard killing it unless you are seeing many major errors. You can force kill by repeatedly pressing control+c or doing a SIGKILL.

### Important note if manually manage your venvs
- You should be running `python -m pip install -r requirements.txt -U https://download.pytorch.org/whl/cu130` every time you `git pull`. (Use `/whl/rocm6.2` instead if applicable)


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
