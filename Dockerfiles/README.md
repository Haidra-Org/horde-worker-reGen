# How to Use horde-worker-reGen Dockerfiles (CUDA and ROCm)

This guide explains how to use the Dockerfiles for building the horde-worker-reGen application, supporting both NVIDIA (CUDA) and AMD (ROCm) GPUs.

## Prerequisites

- Docker installed on your system
- Git installed on your system
- NVIDIA GPU with appropriate drivers (for CUDA version)
- AMD GPU with appropriate drivers (for ROCm version)

## Checkout only this directory

```bash
git clone --sparse https://github.com/Haidra-Org/horde-worker-reGen.git
cd horde-worker-reGen
git sparse-checkout set --no-cone Dockerfiles /bridgeData_template.yaml
```

# Basic setup
## Using docker compose

If your system is set up properly (see [Prerequisites](#prerequisites))
you can just [setup](https://github.com/Haidra-Org/horde-worker-reGen?tab=readme-ov-file#configure) your bridgeData.yaml file and then run
```bash
docker compose -f Dockerfiles/compse.[cuda|rocm].yaml build --pull
docker compose -f Dockerfiles/compse.[cuda|rocm].yaml up -dV
```

> **Warning**: The compose files will automatically pull in your bridgeData.yaml as a mount inside the container. If you have any configuration options set to absolute or windows-style directories (**especially `cache_home`**), this will cause the worker inside the container to not work as expected. You can set `AIWORKER_BRIDGE_DATA_LOCATION` environment variable to set the location of the config file you would like to use if you need an alternative file and you can set `AIWORKER_CACHE_HOME` to set the location of your models folder on the host.

Remember to replace placeholders (e.g. `[cuda|rocm]`) with appropriate values for your setup.
If you want to monitor the containers progress downloading models and working through jobs: [start-or-monitor-running-container](#start-or-monitor-running-container).
The compose file creates a `models` directory in your `horde-worker-reGen` to avoid having to download selected models again.

### Start or monitor running container
To start a container or look at a running containers output.
CTRL+C detaches the container, but leaves it running in the background:
```bash
docker start -ai reGen
```

### Stop
To stop a running container (set it to maintenance mode first using a mangement site like [Artbot](https://tinybots.net/artbot/settings?panel=workers)):
```bash
docker stop reGen
```

### Start detached
To start a container detached (running in the background):
```bash
docker start reGen
```

## Updating your worker
You just need to go to the `horde-worker-reGen` directory, update the git repo, build the new image and let compose recreate the container:
```
cd horde-worker-reGen
git pull
docker compose -f Dockerfiles/compse.[cuda|rocm].yaml build --pull
docker compose -f Dockerfiles/compse.[cuda|rocm].yaml up -dV
```

# Advanced options
## Dockerfile Overview

Two Dockerfiles are provided:
- `Dockerfile.cuda` for NVIDIA GPUs
- `Dockerfile.rocm` for AMD GPUs

Both use multi-stage builds and support customization through build arguments.

## Build Arguments

These can be set either in the `compose.[cuda|rocm].yaml` file, in the `Dockerfile.[cuda|rocm]` or as [CLI arguments](#building-docker-images) for a manual build without compose.
Common build arguments for both Dockerfiles:

- `PYTHON_VERSION`: Python version to install (default: 3.11)
- `GIT_BRANCH`: Branch of the repository to clone (default: main)
- `GIT_OWNER`: Owner of the GitHub repository (default: Haidra-Org)
- `USE_PIP_CACHE`: Whether to use pip caching (default: true)

Specific build arguments:
- For CUDA: `CUDA_VERSION` (default: 12.4.1)
- For ROCm: `ROCM_VERSION` (default: 6.1.2)

## Building Docker Images

### NVIDIA (CUDA) Version

```bash
docker build -f Dockerfile.cuda \
  --build-arg CUDA_VERSION=12.4.1 \
  --build-arg PYTHON_VERSION=3.11 \
  --build-arg GIT_BRANCH=main \
  --build-arg GIT_OWNER=Haidra-Org \
  --build-arg USE_PIP_CACHE=true \
  -t horde-worker-regen:cuda .
```

### AMD (ROCm) Version

```bash
docker build -f Dockerfile.rocm \
  --build-arg ROCM_VERSION=6.1.2 \
  --build-arg PYTHON_VERSION=3.11 \
  --build-arg GIT_BRANCH=main \
  --build-arg GIT_OWNER=Haidra-Org \
  --build-arg USE_PIP_CACHE=true \
  -t horde-worker-regen:rocm .
```

## Running Containers

### NVIDIA (CUDA) Version

```bash
docker run -it --gpus all horde-worker-regen:cuda
```

### AMD (ROCm) Version

```bash
docker run -it --device=/dev/kfd --device=/dev/dri --group-add video horde-worker-regen:rocm
```

## Configuration

- The entrypoint script (`entrypoint.sh`) automatically detects the GPU environment (CUDA or ROCm) and sets up accordingly.
- If `bridgeData.yaml` exists in the container, it will be used for configuration. Otherwise, environment variables will be used.

### Setting config by environment variables

You can set all of the settings for the docker worker via environment variables. Any configuration option in the `bridgeData_template.yaml` can be set this way be prepending `AIWORKER_` to it; see below for some examples.

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

See the bridgeData_template.yaml for more options and specific information about each.

#### Generating an `.env` file from a `bridgeData.yaml`
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

## Customization

- To add GPU-specific setup steps, create `setup_cuda.sh` or `setup_rocm.sh` in the project root and include them in the respective Dockerfile.

## Development Use Cases for GIT_* Variables

1. **Testing Branches in Haidra-Org**:
   ```bash
   --build-arg GIT_BRANCH=feature-new-model
   ```
   Test new features before merging.

2. **Working with Forks**:
   ```bash
   --build-arg GIT_OWNER=your-github-username \
   --build-arg GIT_BRANCH=your-feature-branch
   ```
   Easily switch between forks and branches.

3. **CI/CD Integration**:
   ```bash
   --build-arg GIT_BRANCH=${CI_COMMIT_BRANCH} \
   --build-arg GIT_OWNER=${CI_PROJECT_NAMESPACE}
   ```
   Automate testing and deployment.

4. **Pull Request Review**:
   Reviewers can build and test changes directly:
   ```bash
   --build-arg GIT_OWNER=contributor-username \
   --build-arg GIT_BRANCH=feature-branch
   ```

## Troubleshooting

1. If encountering pip cache-related errors, try building without cache: `--build-arg USE_PIP_CACHE=false`
2. Ensure you have the necessary GPU drivers installed on your host system.
3. For ROCm, make sure your system supports the specified ROCm version.

## Updating

To update the worker:

1. Rebuild the Docker image with the latest code:
   ```bash
   docker build -f Dockerfile.[cuda|amd] -t horde-worker-regen:[cuda|rocm] .
   ```
2. Stop the existing container and start a new one with the updated image.

Remember to replace placeholders (e.g. `[cuda|amd]`) with appropriate values for your setup.
