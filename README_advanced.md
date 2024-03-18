##  If you want the latest information or have questions, come to the [#local-workers](https://discord.com/channels/781145214752129095/1076124012305993768) channel in discord


**Some important details you should know before you start:**

- When submitting debug information **do not publish `.log` files in the server channels - send them to tazlin directly** as we cannot guarantee that your API key would not be in it (though, this warning should relax over time).
- Workers especially interested in logs should note that there is a main log (`bridge.log`) and a log for each subprocess. `bridge_0.log` is the safety process, and all ones after that (`bridge_1.log`, `brige_2.log`, etc) are inference processes.
- You could `Get-Content bridge_1.log -Wait` each log on windows , or `less +F bridge_1.log` on linux to monitor these logs.
- **The worker does not download models on worker start** for the moment (this will change.) You can download all models configured in your bridge data by invoking `python download_models.py`.
- Use a fresh bridge data file (`bridgeData_template.yaml` -> `bridgeData.yaml`).
- Your memory usage will increase up until the number of queued jobs. Its my belief that you should set your queue size to at least 1, and if you're using any number of threads>1, queue size should be 2.
  - Feel free to try queue size 2 with threads at one and let me know if your kudos/hr goes up or down.
- If you have a **low amount of system memory** (16gb or under), do not attempt a queue size greater than 1 if you have more than one model set to load.
- **If you plan on running SDXL**, you will need to ensure at least 9 gb of system ram remains free.
- **If you have an 8 gb card**, SDXL will only reliably work at max_power values close to 32. 42 was too high for my 2080 in certain cases.
- All workers with **less than 24gb VRAM** should **avoid running Stable Cascade 1.0**.
-
## Advanced users, local install:

### Simple usage

### Prerequisites
* Install [git](https://git-scm.com/) in your system.
* Install CUDA if you haven't already.
* Install Python 3.10 or 3.11.
* Clone the worker to your system
   `git clone https://github.com/Haidra-Org/horde-worker-reGen.git`
- Returning workers upgrading from `AI-Horde-Worker`:
  - If you did not set `cache_home` before, set it to your old `AI-Horde-Worker`folder for now to avoid redownloading models. IE, `cache_home: "C:/Horde/AI-Horde-Worker/nataili"` (or `models`, depending on when you first installed the worker).
  - If you had previously set `cache_home`, set it to what you were using before.

### Setup venv
- `python -m venv regen`
  - (certain windows setups) `py -3.11 -m venv regen`
- (windows) `regen\Scripts\activate.bat` for cmd or `regen\Scripts\Activate.ps1` for power shell
- (linux) `source regen/bin/activate`
- **Important**: You should now see `(regen)` prepended in your shell. If you do not, try again or ask for help.

### Get worker beta
- `git clone https://github.com/Haidra-Org/horde-worker-reGen.git`
- `cd .\horde-worker-reGen\`
- `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121`

### Run worker
- Set your config now, copying `bridgeData_template.yaml` to `bridgeData.yaml`, being sure to set an API key and worker name at a minimum
- `python download_models.py` (**critical - must be run first every time**)
  - Make sure the folders that are being downloaded to look correct.
- `python run_worker.py` (to start working)

Pressing control-c will have the worker complete any jobs in progress before ending. Please try and avoid hard killing it unless you are seeing many major errors. You can force kill by repeatedly pressing control-c or doing a SIGKILL.

## Advanced users, container install
