#!/bin/bash

set -euo pipefail

VALID_SPLITS=(layer tensor row none)
VALID_MODELS=(gemma qwen)

is_valid_split() {
    local p
    for p in "${VALID_SPLITS[@]}"; do
        [[ "$1" == "$p" ]] && return 0
    done
    return 1
}

is_valid_model() {
    local p
    for p in "${VALID_MODELS[@]}"; do
        [[ "$1" == "$p" ]] && return 0
    done
    return 1
}

usage() {
    echo "Usage: $0 -model <gemma|qwen> [options]" >&2
    echo ""
    echo "Options:"
    echo "  -model <name>        Model to run: gemma (gemma-4-31B-it) | qwen (Qwen3.8-27B)"
    echo "  -split <splitmode>   Split mode (layer | tensor | row | none - default: layer)"
    echo "  -port <number>       Port number (default: 8080)"
    echo "  -host <address>      Host address (default: 0.0.0.0)"
    echo "  -4bit                Choose 4bit Unsloth quantization instead of the default Q8_K_XL"
    echo "                       (gemma: qat-UD-Q4_K_XL, alias -qat4bit; qwen: UD-Q4_K_XL)"
    echo "  -kv8                 Chose 8 bits quantization for KV cache"
    echo "  -no-mtp              Disable multi token prediction (default: enabled)"
    echo "  -ctx <context>       Context size in tokens (integer, default = 262144)"
    echo "  -batch <batch>       Size of batch higher is better for large prompt processing (default = 512)"
    echo "  -log <file>          Log file for stdout+stderr"
    echo "                       (default: ./llama-server-<timestamp>.log)"
    echo "  -llama-server <path> path to llama-server engine (default: /opt/llama.cpp/llama-server)"
    echo "                       "
    exit 1
}

model=""
split="layer"
no_mtp="no"
host="0.0.0.0"
port="8080"
quant4="no"
context="262144"
log="./llama-server-$(date +%Y%m%d-%H%M%S).log"
llama="/opt/llama.cpp/llama-server"
gguf_prefix="/nvme/gguf"
kq="f16"
vq="f16"
batch="512"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -model)
            model="$2"
            shift 2
            ;;
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
        -4bit|-qat4bit)
            quant4="yes"
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
        -llama-server)
            llama="$2"
            shift 2
            ;;
        -no-mtp)
            no_mtp="yes"
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

if [[ -z "$model" ]]; then
    echo "Error: -model is required (gemma | qwen)" >&2
    usage
fi

if ! is_valid_model "$model"; then
    echo "Error: Invalid model: $model (expected: gemma | qwen)" >&2
    usage
fi

if ! is_valid_split "$split"; then
    echo "Error: Invalid split mode: $split" >&2
    usage
fi

if ! : >>"$log"; then
    echo "Error: cannot write log file: $log" >&2
    exit 1
fi

# ---- Model-specific settings -------------------------------------------------
# Variables set here:
#   model_name   display name
#   model_dir    subdirectory under $gguf_prefix
#   quant        quantization suffix embedded in the GGUF filename
#   mtp          speculative decoding flags (empty if -no-mtp)
#   jinja        chat template file

case "$model" in
    gemma)
        model_name="gemma-4-31B-it"
        model_dir="${gguf_prefix}/gemma-4"
        if [[ "$quant4" == "yes" ]]; then
            quant="qat-UD-Q4_K_XL"
            mtp_md="${model_dir}/mtp-gemma-4-31B-it-Q4_0.gguf"
        else
            quant="UD-Q8_K_XL"
            mtp_md="${model_dir}/mtp-gemma-4-31B-it-Q8_0.gguf"
        fi
        model="${model_dir}/${model_name}-${quant}.gguf"
        # Gemma always needs an explicit MTP draft model
        mtp="--spec-type draft-mtp --spec-draft-n-max 3 -md ${mtp_md}"
        jinja="${model_dir}/gemma-4-31b.jinja"
        ;;
    qwen)
        model_name="Qwen3.8-27B"
        model_dir="${gguf_prefix}/qwen3.8"
        if [[ "$quant4" == "yes" ]]; then
            quant="UD-Q4_K_XL"
            mtp_md="${model_dir}/mtp-Qwen3.8-27B-Q4_0.gguf"
        else
            quant="UD-Q8_K_XL"
            mtp_md=""
        fi
        model="${model_dir}/${model_name}-${quant}.gguf"
        # Qwen needs an explicit draft model only in 4bit mode
        mtp="--spec-type draft-mtp --spec-draft-n-max 3"
        [[ -n "$mtp_md" ]] && mtp+=" -md ${mtp_md}"
        jinja="${model_dir}/qwen3.8-27b.jinja"
        ;;
esac

if [[ "$no_mtp" == "yes" ]]; then
    mtp=""
fi

echo "Chosen model        : $(basename "$model")"
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
   --chat-template-file "$jinja" \
   -sm ${split} \
   --metrics \
   >>"$log" 2>&1 &
