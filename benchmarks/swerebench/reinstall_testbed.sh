#!/usr/bin/env bash

# Keep this sequence aligned with _testbed_reinstall_script in apptainer_build.py.
set -uxo pipefail

if [[ ! -d /testbed ]]; then
    exit 0
fi

__swebench_restore_nounset=0
case $- in
    *u*) __swebench_restore_nounset=1; set +u ;;
esac

if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
    source /opt/conda/etc/profile.d/conda.sh
elif [[ -f /opt/miniconda3/etc/profile.d/conda.sh ]]; then
    source /opt/miniconda3/etc/profile.d/conda.sh
elif [[ -f /opt/miniconda3/bin/activate ]]; then
    source /opt/miniconda3/bin/activate
else
    echo "Conda activation script not found" >&2
    exit 1
fi

conda activate testbed
if [[ "${__swebench_restore_nounset:-0}" == 1 ]]; then
    set -u
fi

cd /testbed
git config --global --add safe.directory /testbed

install_command="$(printf '%s' "${1:?missing encoded install command}" | base64 --decode)"
eval "${install_command}"
