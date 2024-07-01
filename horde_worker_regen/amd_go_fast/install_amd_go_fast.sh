#!/bin/bash
# Currently not sure which flasth-attn is needed for older cards. Leaving this code commented here incase it's needed.
# Determine if the user has an RDNA 3 card. If so there's a different flash-attn to install
#RDNA3_CARD=$(rocminfo | grep -c " gfx11")
#
#if [ "$RDNA3_CARD" ]; then
#    if ! python -s -m pip install -U git+https://github.com/ROCm/flash-attention@howiejay/navi_support; then
#		echo "Tried to install updated flash attention and failed!"
#	else
#		echo "Installed updated flash attn for RDNA 3 cards."
#    fi
#else
#	if ! python -s -m pip install -U git+https://github.com/ROCm/flash-attention; then
#		echo "Tried to install flash attention and failed!"
#	else
#		echo "Installed updated flash attn."
#    fi
#fi


if ! python -s -m pip install -U git+https://github.com/ROCm/flash-attention@howiejay/navi_support; then
	echo "Tried to install updated flash attention and failed!"
else
	echo "Installed updated flash attn."
fi

PY_SITE_DIR=$(python -c "import sysconfig; print(sysconfig.get_path('purelib'))")
if ! cp horde_worker_regen/amd_go_fast/amd_go_fast.py "${PY_SITE_DIR}"/hordelib/nodes/; then
	echo "Failed to install AMD GO FAST."
else
	echo "Installed AMD GO FAST."
fi
