#!/bin/bash

if [ "${FLASH_ATTENTION_USE_TRITON_ROCM^^}" == "TRUE" ]; then
	if ! pip install -U pytest git+https://github.com/ROCm/flash-attention@micmelesse/upstream_pr_rebase; then
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
	echo "Did not detect support for AMD GO FAST. Cleaning up."
 	pip uninstall flash_attn
	rm -f "${PY_SITE_DIR}"/hordelib/nodes/amd_go_fast.py
fi
