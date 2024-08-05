#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ignore_hordelib=false

# Parse command line arguments
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --hordelib)
    hordelib=true
    shift # past argument
    ;;
    --scribe)
    scribe=true
    shift
    ;;
    *)    # unknown option
    echo "Unknown option: $key"
    exit 1
    ;;
esac
shift # past argument or value
done

CONDA_ENVIRONMENT_FILE=environment.yaml

wget -qO- https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64.tar.bz2 | tar -xvj "$SCRIPT_DIR/bin/micromamba"
if [ ! -f "$SCRIPT_DIR/conda/envs/linux/bin/python" ]; then
    bin/micromamba create --no-shortcuts -r "$SCRIPT_DIR/conda" -n linux -f ${CONDA_ENVIRONMENT_FILE} -y
fi
bin/micromamba create --no-shortcuts -r "$SCRIPT_DIR/conda" -n linux -f ${CONDA_ENVIRONMENT_FILE} -y

if [ "$hordelib" = true ]; then
 bin/micromamba run -r "$SCRIPT_DIR/conda" -n linux python -s -m pip uninstall -y hordelib horde_engine horde_sdk horde_model_reference
 bin/micromamba run -r "$SCRIPT_DIR/conda" -n linux python -s -m pip install horde_engine horde_model_reference --extra-index-url https://download.pytorch.org/whl/cu121
else
 bin/micromamba run -r "$SCRIPT_DIR/conda" -n linux python -s -m pip install -r "$SCRIPT_DIR/requirements.txt" -U --extra-index-url https://download.pytorch.org/whl/cu121
fi
