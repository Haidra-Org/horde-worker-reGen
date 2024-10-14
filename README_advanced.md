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
- `python -m venv regen` (only needed the first time you follow these instructions)
  - (certain windows setups) `py -3.11 -m venv regen`
- (windows) `regen\Scripts\activate.bat` for cmd or `regen\Scripts\Activate.ps1` for power shell
- (linux) `source regen/bin/activate`
- **Important**: You should now see `(regen)` prepended in your shell. If you do not, try again or ask for help.

### Get worker files and install dependencies
- `git clone https://github.com/Haidra-Org/horde-worker-reGen.git`
- `cd .\horde-worker-reGen\`
- `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124`

### Run worker
- Set your config now, copying `bridgeData_template.yaml` to `bridgeData.yaml`, being sure to set an API key and worker name at a minimum
- `python download_models.py` (**critical - must be run first every time**)
  - Make sure the folders that are being downloaded to look correct.
- `python run_worker.py` (to start working)

Pressing control-c will have the worker complete any jobs in progress before ending. Please try and avoid hard killing it unless you are seeing many major errors. You can force kill by repeatedly pressing control-c or doing a SIGKILL.

### Important note if manually manage your venvs
- If you manually manage your venvs you should be running `python -m pip install -r requirements.txt -U` everytime you `git pull`.
- `hordelib` has been renamed on pypi to `horde-engine`. The worker will no longer start if `hordelib` is installed. You must manually run `python -m pip uninstall hordelib -y` to be sure hordelib is uninstalled.

## Advanced users, container install

You can find the docker images at https://hub.docker.com/r/tazlin/horde-worker-regen/tags.

> **Important**: Be sure to select the correct Cuda version for your machine. **The physical host must have at least the version of Cuda installed as the image**.

You can set all of the settings for the docker worker via environment variables.
Alternatively it is possible to mount your existing `bridgeData.yaml` file inside the containers working directory:
  - either append `-v ./bridgeData.yaml:/horde-worker-reGen/bridgeData.yaml:ro` to your `docker run` command
  - or include the following in your docker compose config:
```
    volumes:
      - ./bridgeData.yaml:/horde-worker-reGen/bridgeData.yaml:ro
```

It is also possible to bind mount the model directory outside the container so it isn't cleared out after every update.
To do this just mount `./models/:/horde-worker-reGen/models/`. This has to be done with write permissions so the container can actually download the models.

A typical config might include (be sure to change any settings as appropriate as these settings will not work for every machine):

```
AIWORKER_API_KEY=your_api_key_here          # Important
AIWORKER_CACHE_HOME=/workspace/models       # Important
AIWORKER_DREAMER_NAME=your_worker_name_here # Important
AIWORKER_ALLOW_CONTROLNET=True
AIWORKER_ALLOW_LORA=True
AIWORKER_MAX_LORA_CACHE_SIZE=50
AIWORKER_ALLOW_PAINTING=True
AIWORKER_MAX_POWER=38
AIWORKER_MAX_THREADS=1 # Only set to 2 on high end or xx90 machines
AIWORKER_MODELS_TO_LOAD=['TOP 3', 'AlbedoBase XL (SDXL)'] # Be mindful of download times; each model average 2-8 gb
AIWORKER_MODELS_TO_SKIP=['pix2pix', 'SDXL_beta::stability.ai#6901']
AIWORKER_QUEUE_SIZE=2
AIWORKER_MAX_BATCH=4
AIWORKER_SAFETY_ON_GPU=True
AIWORKER_CIVITAI_API_TOKEN=your_token_here
```

See the bridgeData_template.yaml for more specific information.

If you have a local install of the worker, you can use the script `convert_config_to_env.py` to convert a bridgeData.yaml to a valid .env file, as seen here:

- update-runtime users, windows
  ```
  .\runtime.cmd python -s -m convert_config_to_env --file .\bridgeData.yaml
  ```

- update-runtime users, linux
  ```
  ./runtime.sh python -s -m convert_config_to_env --file .\bridgeData.yaml
  ```

- venv users
  ```
  python -m convert_config_to_env --file .\bridgeData.yaml
  ```

... which will write a file to your current working directory named `bridgeData.env`, which is suitable for passing to `docker run` with the `--env-file` cli option. Note that the models_to_load and models_to_skip will be resolved to a list of models if you specified a meta-load command such as `TOP 5` (it would write out the top 5 at that time, **not** the literal `TOP 5`). If you want the dynamic nature of those commands, you should specify them manually.
