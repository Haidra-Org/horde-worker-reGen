#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -f "$SCRIPT_DIR/conda/envs/linux/bin/python" ]; then
"$SCRIPT_DIR/update-runtime-rocm.sh"
fi
"$SCRIPT_DIR/bin/micromamba" run -r "$SCRIPT_DIR/conda" -n linux "$@"
