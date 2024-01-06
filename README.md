This repository allows you to set up a AI Horde Worker to generate, post-process or analyze images for others

> Note: **This worker is still in beta testing**

If you want the latest information or have questions, come to [the thread in discord](https://discord.com/channels/781145214752129095/1159154031151812650)


# AI Horde Worker reGen

This repo contains the latest implementation for the [AI Horde](https://aihorde.net) Worker. This will turn your graphics card(s) into a worker for the AI Horde where you will create images for others. You you will receive in turn earn 'kudos' which will give you priority for your own generations.

Alternatively you can become an Alchemist worker which is much more lightweight and can even run certain modes on CPU (i.e. without a GPU).

Please note that **AMD card are not currently well supported**, but may be in the future. If you are willing to try with your AMD card, join the [discord thread](https://discord.com/channels/781145214752129095/1159154031151812650).


## Some important details you should know before you start

- When submitting debug information **do not publish `.log` files in the discord server channels - send them to tazlin directly** as we cannot guarantee that your API key would not be in it (though, this warning should relax over time).
- You will need to monitor the worker a little closer during the beta, as new ways of failing are possible and potentially not yet accounted for.
  - Workers especially interested in logs should note that there is a main log (`bridge.log`) and a log for each subprocess. `bridge_0.log` is the safety process, and all ones after that (`bridge_1.log`, `brige_2.log`, etc) are inference processes.
  - You could `Get-Content bridge_1.log -Wait` each log on windows , or `less +F bridge_1.log` on linux to monitor these logs.
- Dynamic models is not implemented
- Style meta load commands like `ALL SFW` are not implemented, but `BOTTOM n` has been added.
- We recommend you start with a fresh bridge data file (`bridgeData_template.yaml` -> `bridgeData.yaml`). See Configure section

- Do not set threads higher than 2.
- Your memory usage will increase up until the number of queued jobs. It is our belief that you should set your queue size to at least 1.
  - Feel free to try queue size 2 with threads at one or two and let me know if your kudos/hr goes up or down.
- If you have a low amount of system memory (16gb or under), do not attempt a queue size greater than 1 if you have more than one model set to load.
- If you plan on running SDXL, you will need to ensure at least 9 gb of system ram remains free.
- If you have an 8 gb card, SDXL will only reliably work at max_power values close to 32. 42 was too high for tests on a 2080 in certain cases.

# Installing

**Please see the prior section before proceeding.**

If you haven't already, go to [AI Horde and register an account](https://aihorde.net/register), then store your API key somewhere secure. Treat your API key like a password. You will need it later in these instructions. This will allow your worker to gather kudos for your account.


### Windows

#### Using git (recommended)

Use these instructions if you have installed [git for windows](https://gitforwindows.org/).

This option is recommended as it will make keeping your repository up to date much easier.

1. Use your start menu to open `git GUI`
1. Select "Clone Existing Repository".
1. In the Source location put `https://github.com/Haidra-Org/horde-worker-reGen.git`
1. In the target directory, browse to any folder you want to put the horde worker folder.
1. Press `Clone`
1. Open powershell (also referred to as terminal) or `cmd` from the start menu.
2. continue with the [Basic Usage](#Basic-Usage) instructions

#### Without git

Use these instructions if you do not have git for windows and do not want to install it. These instructions make updating the worker a bit more difficult down the line.

1. Download [the zipped version](https://github.com/Haidra-Org/horde-worker-reGen/archive/refs/heads/main.zip)
1. Extract it to any folder of your choice
1. continue with the [Basic Usage](#Basic-Usage) instructions

### Linux

This assumes you have git installed

Open a bash terminal and run these commands (just copy-paste them all together)

```bash
git clone https://github.com/Haidra-Org/horde-worker-reGen.git
cd horde-worker-reGen
```

Continue with the [Basic Usage](#Basic-Usage) instructions



## Basic Usage

The below instructions refers to `horde-bridge` or `update-runtime`. Depending on your OS, append `.cmd` for windows, or `.sh` for linux (for example, `horde-bridge.cmd` and `update-runtime.cmd` for windows).

You can double click the provided script files below from a file explorer or run it from a terminal like `bash`, `cmd` depending on your OS. The latter option will allow you to **see errors in case of a crash**, so it's recommended.


### Configure

#### Manually

1. Make a copy of `bridgeData_template.yaml` to `bridgeData.yaml`
1. Edit `bridgeData.yaml` and follow the instructions within to fill in your details.

#### WebUI

- **WebUI config is not available on reGen yet**


### Starting/stopping

#### Starting the worker
1. When updates are required, see [Updating](#Updating) for instructions.

2. Depending on the type of worker:
   - 'Dreamer' worker (image generation): run `horde-bridge`.
      * **Warning:** This requires a powerful GPU. You will need a GPU with at least 6GB VRAM and 16GB+ of system RAM.
   - 'Alchemy' worker (upscaling, interrogation, etc) is not current supported and will come in a future version of reGen.


#### Stopping the worker

* In the terminal in which it's running, simply press `Ctrl+C` together.
* The worker will finish the current jobs before exiting.


#### Running with multiple GPUs

> In the future you will not need to run multiple worker instances

To use multiple GPUs each has to start their own instance. For linux, you just need to limit the run to a specific card:

```
CUDA_VISIBLE_DEVICES=0 ./horde-bridge.sh -n "My awesome instance #1"
CUDA_VISIBLE_DEVICES=1 ./horde-bridge.sh -n "My awesome instance #2"
```
etc

**Be warned** that you will need a very high (32-64gb+) amount of system ram depending on your settings. `queue_size` and `max_threads` increases the amount of ram required per worker substantially.

## Updating

The AI Horde workers are under constant improvement. You can follow progress [in our discord](https://discord.gg/3DxrhksKzn) and get notifications about updates there. If you are interested in beta notifications, go to the [#get-roles channel](https://discord.com/channels/781145214752129095/977498954616954890) and get the appropriate role.

To update:

1. Shut down your worker by pressing ctrl+c once and waiting for the worker to stop.

1. Update this repo using the appropriate method:
    ### git method

    Use this approach if you cloned the original repository using `git clone`

    1. Open a or `bash`, `cmd`, or `powershell` terminal depending on your OS
    2. Navigate to the folder you have the AI Horde Worker repository installed if you're not already there.
    3. run `git pull`

    Afterwards run the `horde-bridge` script for your OS as usual.

    ### zip method

    Use this approach if you downloaded the git repository as a zip file and extracted it somewhere.


    1. delete the `worker/` directory from your folder
    1. Download the [repository from github as a zip file](https://github.com/db0/horde-worker-reGen/archive/refs/heads/main.zip)
    1. Extract its contents into the same the folder you have the AI Horde Worker repository installed, overwriting any existing files

1. Run the `update-runtime` script for your OS. This will update all dependencies if required.
   - Some updates may not require this and the update notification will tell you if this is the case.
   - When in doubt, you should run it anyway.
1. Continue with [Starting/stopping](#Starting/stopping) instructions above


# Model Usage
Many models in this project use the CreativeML OpenRAIL License.  [Please read the full license here.](https://huggingface.co/spaces/CompVis/stable-diffusion-license)


# Docker

Not yet supported in reGen.
