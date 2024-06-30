#!/bin/bash

# Determine if the user has a flash attention supported card.
SUPPORTED_CARD=$(rocminfo | grep -c -e gfx1100 -e gfx1101 -e gfx1102)

if [ "$SUPPORTED_CARD" -gt 0 ]; then
    if ! python -s -m pip install -U git+https://github.com/ROCm/flash-attention@howiejay/navi_support; then
		echo "Tried to install flash attention and failed!"
	else
		echo "Installed flash attn."
		PY_SITE_DIR=$(python -c "import sysconfig; print(sysconfig.get_path('purelib'))")
		if ! cp horde_worker_regen/amd_go_fast/amd_go_fast.py "${PY_SITE_DIR}"/hordelib/nodes/; then
			echo "Failed to install AMD GO FAST."
		else
			echo "Installed AMD GO FAST."
		fi
    fi
else
	echo "Did not detect support for AMD GO FAST"
fi
