##  If you want the latest information or have questions, come to the thread in discord
https://discord.com/channels/781145214752129095/1159154031151812650

## This worker is still in beta testing

**Some important details you should know before you start:**

- When submitting debug information **do not publish `.log` files in the server channels - send them to tazlin directly** as we cannot guarantee that your API key would not be in it (though, this warning should relax over time).
- You will need to monitor the worker a little closer during the beta, as new ways of failing are possible and potentially not yet accounted for.
  - Workers especially interested in logs should note that there is a main log (`bridge.log`) and a log for each subprocess. `bridge_0.log` is the safety process, and all ones after that (`bridge_1.log`, `brige_2.log`, etc) are inference processes.
  - You could `Get-Content bridge_1.log -Wait` each log on windows , or `less +F bridge_1.log` on linux to monitor these logs.
- The worker does not download models on worker start for the moment (this will change.) You can download all models configured in your bridge data by invoking `python download_models.py`.
- Dynamic models is not implemented
- Style meta load commands like `ALL SFW` are not implemented, but `BOTTOM n` has been added.
- I recommend you start with a fresh bridge data file (`bridgeData_template.yaml` -> `bridgeData.yaml`).

- Your memory usage will increase up until the number of queued jobs. Its my belief that you should set your queue size to at least 1, and if you're using threads at least max_threads + 1.
  - Feel free to try queue size 2 with threads at one and let me know if your kudos/hr goes up or down.
- If you have a low amount of system memory (16gb or under), do not attempt a queue size greater than 1 if you have more than one model set to load.
- If you plan on running SDXL, you will need to ensure at least 9 gb of system ram remains free.
- If you have an 8 gb card, SDXL will only reliably work at max_power values close to 32. 42 was too high for my 2080 in certain cases.

# To run the reGen worker (Please read above first)

## Simple usage

**You do not need to have anything installed for this section**

Choose one of the two download options

### Download Option 1 (Recommended).

* Install [git](https://git-scm.com/) in your system.
* using git commands, clone the worker to your system
   `git clone https://github.com/Haidra-Org/horde-worker-reGen.git`
* Continue with the Setup steps

### Download Option 2

(This approach will make it more difficult to keep the worker up to date later on)

* Download the zip file from the repo
* Extract the zip file in any folder of your choice
* Continue with the Setup steps


### Setup and run

1. Inside the folder where you installed your worker, run `update_runtime.sh`
1. Make a copy of `bridgeData_template.yaml` to `bridgeData.yaml`
1. Edit `bridgeData.yaml` and follow the instructions within to fill in your details.


## Advanced users:

**You must have git, cuda and python installed on your system for this section**

  - If you did not set `cache_home` before, set it to your old `AI-Horde-Worker`folder for now to avoid redownloading models. IE, `cache_home: "C:/Horde/AI-Horde-Worker/nataili"` (or `models`, depending on when you first installed the worker).
  - If you had previously set `cache_home`, set it to what you were using before.

# Setup venv
- `python -m venv regen`
  - (certain windows setups) `py -3.10 -m venv regen`
- (windows) `regen\Scripts\activate.bat` for cmd or `regen\Scripts\Activate.ps1` for power shell
- (linux) `source regen/bin/activate`
- **Important**: You should now see `(regen)` prepended in your shell. If you do not, try again or ask for help.

# Get worker beta
- `git clone https://github.com/Haidra-Org/horde-worker-reGen.git`
- `cd .\horde-worker-reGen\`
- `pip install -r requirements.txt`

# Run worker
- (Set your config now, adding `SDXL 1.0` if desired)
- `python download_models.py` (To download `SDXL 1.0`) (**critical**)
  - Make sure the folders that are being downloaded to look correct.
- `python run_worker.py` (to start working)
  - Please let me know if you see a variation in your kudos/hr. It does not currently report the models bonus as part of the calculation.

Pressing control-c will have the worker complete any jobs in progress before ending. Please try and avoid hard killing it unless you are seeing many major errors.
