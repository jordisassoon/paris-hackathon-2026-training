#!/bin/bash
# Helper script to setup pytorch
# Run as the user you want to use it as

set -e

get_cuda_version() {
  if command -v nvcc &> /dev/null; then
    local cuda_version
    cuda_version=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
    if [[ -z "${cuda_version}" ]]; then
      echo "Warning: CUDA version could not be determined. Defaulting to CPU." >&2
      echo "cpu"
    fi
  else
    echo "Warning: nvcc not found. Assuming CPU-only installation." >&2
    echo "cpu"
  fi

  echo "${cuda_version}"
}

install_pytorch() {
  local user="$1"
  local install_path="$2"

  if [[ -z "$user" || -z "$install_path" ]]; then
    echo "Usage: install_pytorch <user> <install_path>"
    return 1
  fi

  cd

  if [[ "$user" != "${USER}" ]]; then
    echo "Run as the user you want to set it up as"
    return 2
  fi

  if ! mkdir -vp "$install_path"; then
    echo "Error: Failed to create directory $install_path for user $user. Trying as root instead"
    sudo mkdir -vp "${install_path}"
    sudo chown -v "${user}:${user}" "${install_path}"
  fi

  if ! which uv 2>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for this session
    export PATH="$HOME/.local/bin:$PATH"
  else
    # Ensure uv is in PATH
    export PATH="$HOME/.local/bin:$PATH"
  fi

  if [ -f "${install_path}/bin/activate" ]; then
    # shellcheck disable=SC1091
    . "${install_path}/bin/activate"
  else
    # --seed adds pip etc
    uv venv "${install_path}" --seed
    # shellcheck disable=SC1091
    . "${install_path}/bin/activate"
  fi
  if [[ "${cuda_version_without_dots}" != "" ]]; then
    if ! bash -c "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu${cuda_version_without_dots}"; then
      echo "Error: PyTorch installation failed for user $user with CUDA ${cuda_version_without_dots}"
      return 1
    fi
    echo "OK: PyTorch has been installed for user $user. To activate, run: '. $install_path/bin/activate'"
  else
    echo "No cuda_version found: ${cuda_version_without_dots}, not installing pytorch"
    return 3
  fi

  cd -
}

cuda_version=$(get_cuda_version)
cuda_version_without_dots=$(echo "${cuda_version}"|tr -d '\.')

install_pytorch "${USER}" "/home/venv_${USER}_cu${cuda_version_without_dots}"
