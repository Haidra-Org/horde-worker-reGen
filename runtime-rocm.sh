#!/bin/bash
if [ ! -f "conda/envs/linux/bin/python" ]; then
./update-runtime-rocm.sh
fi
bin/micromamba run -r conda -n linux "$@"
