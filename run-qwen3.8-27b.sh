#!/bin/bash

set -euo pipefail

# tensor split mode produced 40 tok/s but after a little while 1 GPU hung
# -dev is not required as llama-server takes by default all GPU podman provides (-dev /dev/dri)

# VALID_SPLITS=(layer tensor row none)
# CONTAINER_NAME="llama-cockpit-server-72cb15ef"

# is_valid_split() {
#     local p
#     for p in "${VALID_SPLITS[@]}"; do
#         [[ "$1" == "$p" ]] && return 0
#     done
#     return 1
# }

usage() {
    echo "Usage: $0 [options]" >&2
    echo ""
    echo "Options:"
    #echo "  -split <splitmode>  Split mode (layer | tensor | row | none - default: layer)"
    echo "  -port <number>       Port number (default: 8080)"
    echo "  -host <address>      Host address (default: 0.0.0.0)"
    #echo "  -8bit              Choose Unsloth quantization Q8_K_XL instead of the default Q4_K_XL"
    echo "  -no-mtp              Disable multi token prediction (default: enabled)"
    echo "  -ctx <context>       Context size in tokens (integer, default = 262144)"
    echo "  -log <file>          Log file for stdout+stderr"
    echo "                       (default: /tmp/llama-server-<timestamp>.log)"
    echo "  -llama-server <path> path to llama-server engine (default: /opt/llama.cpp/llama-server)"
    echo "                       "
    exit 1
}

split="layer"
mtp="--spec-type draft-mtp --spec-draft-n-max 5"
host="0.0.0.0"
port="8080"
quant="4"
context="262144"
log="./llama-server-$(date +%Y%m%d-%H%M%S).log"
llama="/opt/llama.cpp/llama-server"

while [[ $# -gt 0 ]]; do
    case "$1" in
        # -split)
        #     split="$2"
        #     shift 2
        #     ;;
        -host)
            host="$2"
            shift 2
            ;;
        -port)
            port="$2"
            shift 2
            ;;
        # -8bit)
        #     quant="8"
        #     shift
        #     ;;
        -ctx)
            context="$2"
            shift 2
            ;;
        -log)
            log="$2"
            shift 2
            ;;
        -no-mtp)
            # if [[ "$quant" == "4" ]]; then
            #     mtp="-md /models/mtp-Qwen3.8-27B-Q4_0.gguf --spec-type draft-mtp --spec-draft-n-max 5"
            # else
            #     mtp="--spec-type draft-mtp --spec-draft-n-max 5"
            # fi
            mtp=""
            shift
            ;;
        -help)
            usage
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            ;;
    esac
done

# if ! is_valid_split "$split"; then
#     echo "Error: Invalid split mode: $split" >&2
#     usage
# fi

# if [[ -z "$log" ]]; then
#     log="/tmp/llama-server-$(date +%Y%m%d-%H%M%S).log"
# fi

# il file di log deve essere creabile prima di lanciare in background,
# altrimenti l'errore si perderebbe
if ! : >>"$log"; then
    echo "Error: cannot write log file: $log" >&2
    exit 1
fi

# if podman container exists "$CONTAINER_NAME"; then
#     echo "Error: container '$CONTAINER_NAME' already exists. Stop it first:" >&2
#     echo "       podman stop $CONTAINER_NAME" >&2
#     exit 1
# fi

# echo "Chosen quantization : $quant"
echo "Chosen model        : Qwen3.8-27B-UD-Q8_K_XL.gguf"
echo "Context size        : $context"
echo "Speculative decode  : [$mtp]"
# echo "Split mode          : $split"
echo "Endpoint            : ${host}:${port}"
echo "Log file            : $log"

{
    echo "=== $(date -Is) ==="
    echo "model=Qwen3.8-27B-UD-Q8_K_XL.gguf ctx=$context endpoint=${host}:${port}"
    echo "mtp=[$mtp]"
    echo "==="
} >>"$log"

# --rm resta valido; -it e' stato rimosso perche' in background non c'e' TTY.
# nohup + & : il container sopravvive alla chiusura del terminale.
#ONEAPI_DEVICE_SELECTOR=level_zero:*
# nohup podman run --rm --name "$CONTAINER_NAME" \
#   --device /dev/dri \
#   --group-add keep-groups \
#   --security-opt seccomp=unconfined \
#   --security-opt label=disable \
#   --userns=keep-id \
#   -e XDG_CACHE_HOME=/tmp \
#   -v /nvme/gguf:/models:ro \
#   -p "${host}:${port}:${port}" \
#   docker.io/kyuz0/intel-b70-ai-toolboxes:sycl \
#   llama-server \
#   -m /models/Qwen3.8-27B-UD-Q${quant}_K_XL.gguf $mtp \
#   -c "$context" \
#   -ngl 999 \
#   --host 0.0.0.0 --port "$port" \
#   --load-mode none \
#   -fa 1 \
#   --cache-type-k f16 --cache-type-v f16 \
#   -b 2048 -ub 512 \
#   -np 1 \
#   --jinja \
#   --reasoning on \
#   --chat-template-kwargs '{"reasoning_effort":"medium"}' \
#   --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 \
#   --fit off \
#   --cache-ram 0 \
#   --chat-template-file /models/qwen3.8-27b.jinja \
#   -sm "$split" \
#   --metrics \
#   >>"$log" 2>&1 &

source /opt/intel/oneapi/setvars.sh

ONEAPI_DEVICE_SELECTOR=level_zero:* $llama \
   -m /nvme/gguf/Qwen3.8-27B-UD-Q8_K_XL.gguf $mtp \
   -c "$context" \
   -ngl 999 \
   --host ${host} --port "$port" \
   --load-mode none \
   -fa 1 \
   --cache-type-k f16 --cache-type-v f16 \
   -b 2048 -ub 512 \
   -np 1 \
   --jinja \
   --reasoning on \
   --reasoning-effort medium \
   #--chat-template-kwargs '{"reasoning_effort":"medium"}' \
   --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 \
   --fit off \
   --cache-ram 0 \
   --chat-template-file /nvme/gguf/qwen3.8-27b.jinja \
   -sm layer \
   --metrics \
   >"$log" 2>&1 &


# pid=$!
# disown "$pid" 2>/dev/null || true

# # verifica che non sia morto subito (errori podman/OCI, porta occupata, modello assente)
# sleep 3
# if ! kill -0 "$pid" 2>/dev/null; then
#     echo "" >&2
#     echo "Error: podman terminated immediately. Last lines of $log:" >&2
#     tail -n 25 "$log" >&2
#     exit 1
# fi

# echo ""
# echo "Started. podman PID: $pid"
# echo "Follow the log with : tail -f $log"
echo "Stop the server with: podman stop $CONTAINER_NAME"
