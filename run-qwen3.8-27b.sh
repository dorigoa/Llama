#!/bin/bash

set -euo pipefail

VALID_SPLITS=(layer tensor row none)

is_valid_split() {
    local p
    for p in "${VALID_SPLITS[@]}"; do
        [[ "$1" == "$p" ]] && return 0
    done
    return 1
}

usage() {
    echo "Usage: $0 [options]" >&2
    echo ""
    echo "Options:"
    echo "  -split <splitmode>   Split mode (layer | tensor | row | none - default: layer)"
    echo "  -port <number>       Port number (default: 8080)"
    echo "  -host <address>      Host address (default: 0.0.0.0)"
    echo "  -4bit                Choose Unsloth quantization Q4_K_XL instead of the default Q8_K_XL"
    echo "  -kv8                 Chose 8 bits quantization for KV cache"
    echo "  -no-mtp              Disable multi token prediction (default: enabled)"
    echo "  -ctx <context>       Context size in tokens (integer, default = 262144)"
    echo "  -batch <batch>       Size of batch higher is better for large prompt processing (default = 512)"
    echo "  -log <file>          Log file for stdout+stderr"
    echo "                       (default: /tmp/llama-server-<timestamp>.log)"
    echo "  -llama-server <path> path to llama-server engine (default: /opt/llama.cpp/llama-server)"
    echo "                       "
    exit 1
}

split="layer"
mtp="--spec-type draft-mtp --spec-draft-n-max 3"
host="0.0.0.0"
port="8080"
quant="8"
context="262144"
log="./llama-server-$(date +%Y%m%d-%H%M%S).log"
llama="/opt/llama.cpp/llama-server"
gguf_prefix="/nvme/gguf"
kq="f16"
vq="f16"
batch="512"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -split)
            split="$2"
            shift 2
            ;;
        -host)
            host="$2"
            shift 2
            ;;
        -port)
            port="$2"
            shift 2
            ;;
        -4bit)
            quant="4"
            mtp="--spec-type draft-mtp --spec-draft-n-max 3 -md ${gguf_prefix}/qwen3.8/mtp-Qwen3.8-27B-Q4_0.gguf"
            shift 1
            ;;
        -kv8)
            kq="q8_0"
            vq="q8_0"
            shift 1
            ;;
        -ctx)
            context="$2"
            shift 2
            ;;
        -batch)
            batch="$2"
            shift 2
            ;;
        -log)
            log="$2"
            shift 2
            ;;
        -no-mtp)
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

if ! is_valid_split "$split"; then
    echo "Error: Invalid split mode: $split" >&2
    usage
fi

if ! : >>"$log"; then
    echo "Error: cannot write log file: $log" >&2
    exit 1
fi

model="${gguf_prefix}/qwen3.8/Qwen3.8-27B-UD-Q${quant}_K_XL.gguf"

echo "Chosen model        : Qwen3.8-27B-UD-Q${quant}_K_XL.gguf"
echo "Context size        : $context"
echo "Speculative decode  : [$mtp]"
echo "Split mode          : $split"
echo "Endpoint            : ${host}:${port}"
echo "Log file            : $log"

set +eu
source /opt/intel/oneapi/setvars.sh --force
set -eu

ONEAPI_DEVICE_SELECTOR=level_zero:* GGML_SYCL_DEV2DEV_MEMCPY=2 $llama \
   -dev SYCL0,SYCL1 \
   -m $model $mtp \
   -c "$context" \
   -ngl 999 \
   --host $host --port "$port" \
   --load-mode none \
   -fa 1 \
   --cache-type-k $kq --cache-type-v $vq \
   -b 2048 -ub $batch \
   -np 1 \
   --jinja \
   --reasoning on \
   --reasoning-effort medium \
   --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 \
   --fit off \
   --cache-ram 0 \
   --chat-template-file ${gguf_prefix}/qwen3.8/qwen3.8-27b.jinja \
   -sm ${split} \
   --metrics \
   >>"$log" 2>&1 &
