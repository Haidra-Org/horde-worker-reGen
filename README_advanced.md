##  If you want the latest information or have questions, come to the [#local-workers](https://discord.com/channels/781145214752129095/1076124012305993768) channel in the [official discord](https://discord.gg/3DxrhksKzn).


**Some important details you should know before you start:**

> See [this important info](README.md/#important-info) first.

- When submitting debug information **do not publish `.log` files in the server channels - send them to tazlin directly** as we cannot guarantee that your API key would not be in it (though, this warning should relax over time).
- Workers especially interested in logs should note that there is a main log (`bridge.log`) and a log for each subprocess. `bridge_0.log` is the safety process, and all ones after that (`bridge_1.log`, `brige_2.log`, etc) are inference processes.
- You could `Get-Content bridge_1.log -Wait` each log on windows , or `less +F bridge_1.log` on linux to monitor these logs.
- **The worker does not download models on worker start** for the moment (this will change.) You can download all models configured in your bridge data by invoking `python download_models.py`.


## Advanced users, local install:

### Simple usage

### Prerequisites
* Install [git](https://git-scm.com/) in your system.
* Install CUDA/RoCM if you haven't already.
* Install Python 3.10 or 3.11.
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

## Advanced users, container install

You can find the docker images at https://hub.docker.com/r/tazlin/horde-worker-regen/tags.

See [Dockerfiles/README.md](Dockerfiles/README.md) for a detailed guide on the supported docker functionality.
