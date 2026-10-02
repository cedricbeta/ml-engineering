#!/usr/bin/env bash

# Report the GPU and RDMA driver stack - kernel modules, driver user-space libraries, CUDA versions,
# PyTorch's view and the RDMA devices - so that mismatches between the layers are easy to spot.
#
# Works on the host and inside a container/pod (where it shows what the container actually got).
# Needs no root and no extra packages; whatever isn't available is reported as such.
#
# usage:
#   bash driver-report.sh
#   kubectl exec -i POD -- bash -s < driver-report.sh
#   docker run --rm -i --gpus all IMAGE bash -s < driver-report.sh
#
# The meaning of each section is explained in drivers.md

section() { printf "\n=== %s ===\n" "$1"; }
have() { command -v "$1" > /dev/null 2>&1; }

# list loaded kernel modules matching a regex - /proc/modules is the host's even inside a container
loaded_modules() {
    if [ -r /proc/modules ]; then
        awk '{print $1}' /proc/modules | grep -E "$1" | sort | tr '\n' ' '
        echo
    else
        echo "/proc/modules is not readable"
    fi
}

# resolve a shared library's soname to the real file, which carries the full version in its name
resolve_lib() {
    local path
    path=$(ldconfig -p 2>/dev/null | awk -v l="$1" '$1==l {print $NF; exit}')
    if [ -n "$path" ]; then
        echo "$1 -> $(readlink -f "$path")"
    else
        echo "$1: not found"
    fi
}

section "Kernel"
echo "kernel: $(uname -r) ($(uname -m))"
if [ -f /.dockerenv ] || [ -f /run/.containerenv ] || [ -n "$KUBERNETES_SERVICE_HOST" ]; then
    echo "running inside a container - the kernel and the kernel modules below are the host's"
fi

section "NVIDIA kernel driver"
if [ -r /proc/driver/nvidia/version ]; then
    # "Open Kernel Module" in this line means the open modules, otherwise the proprietary ones
    head -1 /proc/driver/nvidia/version
else
    echo "not loaded (no /proc/driver/nvidia/version)"
fi
echo -n "modules: "; loaded_modules '^(nvidia|nvidia_uvm|nvidia_modeset|nvidia_drm|nvidia_peermem|gdrdrv)$'
echo -n "devices: "; ls /dev/nvidia* 2>/dev/null | tr '\n' ' '; echo

section "NVIDIA driver user-space (must match the kernel driver version)"
resolve_lib libcuda.so.1
resolve_lib libnvidia-ml.so.1
for d in /usr/local/cuda/compat /usr/local/cuda-*/compat; do
    [ -d "$d" ] && echo "CUDA forward compatibility libraries present in $d: $(ls "$d" | grep -m1 'libcuda.so.')"
done
if have nvidia-smi; then
    nvidia-smi --query-gpu=index,name,compute_cap,driver_version,persistence_mode,pci.bus_id --format=csv
    nvidia-smi | grep -o "CUDA Version: [0-9.]*" | sed 's/^/max supported by the driver: /'
    nvidia-smi -q | grep -m1 "GSP Firmware Version"
    # NVSwitch systems: "Completed" + "Success" means Fabric Manager has set up the NVLink fabric
    nvidia-smi -q | grep -A2 -E "^\s+Fabric$" | grep -E "State|Status" | sort | uniq -c
else
    echo "nvidia-smi not found (inside a container: NVIDIA_DRIVER_CAPABILITIES may lack 'utility')"
fi
echo "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES-<unset>}"
echo "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES-<unset>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>}"

section "Host services"
if have systemctl && systemctl list-units > /dev/null 2>&1; then
    for s in nvidia-persistenced nvidia-fabricmanager nvidia-imex nvidia-dcgm; do
        echo "$s: $(systemctl is-active $s 2>/dev/null)"
    done
else
    echo "systemctl not available (normal inside a container)"
fi

section "CUDA in this environment"
if have nvcc; then
    nvcc --version | grep release
else
    echo "nvcc: not installed (fine unless you compile CUDA code)"
fi
PYTHON=$(command -v python || command -v python3)
if [ -n "$PYTHON" ]; then
    "$PYTHON" - <<'EOF'
try:
    import torch
except ImportError:
    print("torch: not installed")
    raise SystemExit
print(f"torch {torch.__version__}, built with CUDA {torch.version.cuda}, HIP {torch.version.hip}")
available = torch.cuda.is_available()
print(f"torch.cuda.is_available(): {available}, device count: {torch.cuda.device_count()}")
if available:
    print(f"NCCL: {'.'.join(map(str, torch.cuda.nccl.version()))}")
    major, minor = torch.cuda.get_device_capability(0)
    print(f"GPU 0: {torch.cuda.get_device_name(0)}, arch sm_{major}{minor}")
    print(f"archs compiled into this torch build: {' '.join(torch.cuda.get_arch_list())}")
EOF
else
    echo "python: not found"
fi

section "RDMA / InfiniBand / RoCE / EFA"
echo -n "modules: "; loaded_modules '^(ib_core|ib_uverbs|ib_umad|rdma_ucm|mlx5_core|mlx5_ib|efa)$'
echo -n "devices: "; ls /dev/infiniband 2>/dev/null | tr '\n' ' '; echo
if [ -d /sys/class/infiniband ] && [ -n "$(ls /sys/class/infiniband 2>/dev/null)" ]; then
    for dev in /sys/class/infiniband/*; do
        fw=$(cat "$dev/fw_ver" 2>/dev/null)
        for port in "$dev"/ports/*; do
            echo "$(basename "$dev") port $(basename "$port"): $(cat "$port/state" 2>/dev/null), $(cat "$port/rate" 2>/dev/null), link_layer=$(cat "$port/link_layer" 2>/dev/null), fw=$fw"
        done
    done
else
    echo "no RDMA devices in /sys/class/infiniband"
fi
echo "memlock limit (needs to be unlimited for RDMA): $(ulimit -l)"
ldconfig -p 2>/dev/null | grep -E "libibverbs\.so\.|libnccl-net|libfabric\.so\." | awk '{print $1 " -> " $NF}'

if [ -e /dev/kfd ]; then
    section "AMD"
    echo "amdgpu kernel driver: $(cat /sys/module/amdgpu/version 2>/dev/null || echo 'version unknown (inbox driver?)')"
    have amd-smi && amd-smi version
fi
