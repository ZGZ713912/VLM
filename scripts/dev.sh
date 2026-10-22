#!/usr/bin/env bash
# One entry point for Linux CLI and Dev Containers initialization.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/dev.sh cpu|gpu COMMAND [ARGS...]

  prepare     Create .env if needed and synchronize host UID/GID (no container start)
  up -d       Start the development container
  build       Build the selected CPU or CUDA image locally
  shell       Open zsh in the running development container
  check       Run a one-off environment/permissions/device check
  test        Run the project pytest suite in a one-off container
  config      Show the merged Docker Compose configuration
  down        Stop this project's container, preserving home/cache volumes

Other commands/arguments are forwarded to Docker Compose.
Optional personal mounts: VLM_HOST_TOOLS=1 ./scripts/dev.sh cpu|gpu COMMAND
EOF
}

mode="${1:-help}"
action="${2:-help}"
case "${mode}" in
  -h|--help|help) usage; exit 0 ;;
  cpu|gpu) ;;
  *) usage >&2; exit 2 ;;
esac
case "${action}" in
  -h|--help|help) usage; exit 0 ;;
esac
shift 2

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_dir}"
export LOCAL_UID="${LOCAL_UID:-$(id -u)}"
export LOCAL_GID="${LOCAL_GID:-$(id -g)}"
if [[ ! "${LOCAL_UID}" =~ ^[0-9]+$ || ! "${LOCAL_GID}" =~ ^[0-9]+$ ]]; then
  echo 'LOCAL_UID and LOCAL_GID must be numeric.' >&2
  exit 2
fi

if [[ "${mode}" == gpu ]]; then
  case "${action}" in
    up|build|run|check|test|prepare)
      case "$(uname -m)" in
        x86_64|amd64) ;;
        *) echo 'The pinned CUDA image requires x86-64; use cpu on ARM64 (Jetson needs a separate L4T image).' >&2; exit 2 ;;
      esac
      ;;
  esac
fi

# Never source .env as shell code. Preserve all user settings except these IDs.
umask 077
if [[ ! -e .env ]]; then
  cp .env.example .env
fi
env_tmp="$(mktemp "${project_dir}/.env.tmp.XXXXXX")"
trap 'rm -f -- "${env_tmp}"' EXIT
awk -v uid="${LOCAL_UID}" -v gid="${LOCAL_GID}" '
  /^[[:space:]]*(export[[:space:]]+)?LOCAL_UID[[:space:]]*=/ { print "LOCAL_UID=" uid; u=1; next }
  /^[[:space:]]*(export[[:space:]]+)?LOCAL_GID[[:space:]]*=/ { print "LOCAL_GID=" gid; g=1; next }
  { print }
  END { if (!u) print "LOCAL_UID=" uid; if (!g) print "LOCAL_GID=" gid }
' .env > "${env_tmp}"
# Keep the existing file's permissions and any symlink target intact.
cat "${env_tmp}" > .env

if [[ "${action}" == prepare ]]; then
  echo "Prepared .env for UID:GID ${LOCAL_UID}:${LOCAL_GID} (${mode})."
  exit 0
fi

compose=(docker compose --project-directory "${project_dir}" -f "${project_dir}/docker-compose.yml")
expect=cpu
if [[ "${mode}" == gpu ]]; then
  compose+=(-f "${project_dir}/docker-compose.gpu.yml")
  expect=cuda
fi
if [[ "${VLM_HOST_TOOLS:-0}" == 1 ]]; then
  compose+=(-f "${project_dir}/docker-compose.host-tools.yml")
fi

# Let Compose parse dotenv interpolation instead of executing the file as shell.
compose_setting() {
  "${compose[@]}" config --environment | awk -v key="$1" '
    index($0, key "=") == 1 { value=substr($0, length(key)+2) }
    END { print value }
  '
}

case "${action}" in
  build)
    base_image="$(compose_setting CPU_BASE_IMAGE)"
    base_image="${base_image:-python:3.10-slim-bookworm}"
    torch_index=https://download.pytorch.org/whl/cpu
    if [[ "${mode}" == gpu ]]; then
      base_image="$(compose_setting CUDA_BASE_IMAGE)"
      base_image="${base_image:-pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime}"
      torch_index=https://download.pytorch.org/whl/cu121
    fi
    image="$("${compose[@]}" config --images)"
    shell_tools="$(compose_setting INSTALL_OPTIONAL_SHELL_TOOLS)"
    # Runtime/VS Code only use images; keep local builds in this explicit command.
    docker build --tag "${image}" \
      --build-arg "BASE_IMAGE=${base_image}" \
      --build-arg "TORCH_INDEX_URL=${torch_index}" \
      --build-arg "INSTALL_OPTIONAL_SHELL_TOOLS=${shell_tools:-0}" \
      "$@" "${project_dir}"
    ;;
  shell) "${compose[@]}" exec vlm-dev zsh "$@" ;;
  check) "${compose[@]}" run --rm --no-deps vlm-dev python scripts/docker/check_environment.py --expect "${expect}" "$@" ;;
  test) "${compose[@]}" run --rm --no-deps vlm-dev python -m pytest -q "$@" ;;
  *) "${compose[@]}" "${action}" "$@" ;;
esac
