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
git sparse-checkout set --no-cone Dockerfiles
```

## Dockerfile Overview

Two Dockerfiles are provided:
- `Dockerfile.cuda` for NVIDIA GPUs
- `Dockerfile.amd` for AMD GPUs

Both use multi-stage builds and support customization through build arguments.

## Build Arguments

Common build arguments for both Dockerfiles:

- `PYTHON_VERSION`: Python version to install (default: 3.11)
- `GIT_BRANCH`: Branch of the repository to clone (default: main)
- `GIT_OWNER`: Owner of the GitHub repository (default: Haidra-Org)
- `USE_PIP_CACHE`: Whether to use pip caching (default: true)

Specific build arguments:
- For CUDA: `CUDA_VERSION` (default: 12.4.1)
- For ROCm: `ROCM_VERSION` (default: 6.0.2)

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
docker build -f Dockerfile.amd \
  --build-arg ROCM_VERSION=6.0.2 \
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

Remember to replace placeholders (e.g., `[cuda|amd]`) with appropriate values for your setup.
