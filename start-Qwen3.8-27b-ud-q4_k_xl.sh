# ONEAPI_DEVICE_SELECTOR='level_zero:*' /opt/llama.cpp/llama-server \
#   -a "Qwen3.8-27b-ud-q4_k_xl" \
#   -m /nvme/gguf/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf \
#   -md /nvme/gguf/qwen3.8-27b/mtp-Qwen3.8-27B-Q4_0.gguf \
# 	--load-mode none \
#   -ngl 999 -ts 1,1 \
#   -c 262144 -ctk q8_0 -ctv q8_0 \
#   -fa on \
#   --jinja \
#   -lv 4 --log-file ~/llama-server.out \
#   --spec-type draft-mtp --spec-draft-n-max 3 -np 1 \
#   -sm tensor \
#   --no-mmproj-offload \
#   --chat-template-file ~/Llama/chat-templates/qwen3.8-27b.jinja \
#   --metrics \
#   --host 192.168.1.200 --port 8080 --no-warmup

/opt/llama-b12246-sycl/llama-server \
	-m /nvme/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q4_K_XL.gguf \
	--alias Qwen3.8-27B-GGUF-UD-Q4_K_XL \
	--parallel 1 \
	--flash-attn on \
	--no-context-shift -c 262144 \
	-ngl -1 \
	--fit on \
	--slot-save-path /nvme/llama-slots \
	--kv-unified \
	--jinja \
	--verbosity 4 \
	--reasoning on \
	--chat-template-kwargs '{"preserve_thinking": true}' \
	--ctx-checkpoints 12 \
	--device SYCL0,SYCL1 \
	--chat-template-file /home/alvise/Llama/chat-templates/qwen3.8-27b.jinja \
	--metrics \
	--port 8080 --host 0.0.0.0
