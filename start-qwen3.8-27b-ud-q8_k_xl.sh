/opt/llama.cpp/llama-server \
	-m /nvme/gguf/qwen3.8-27b/Qwen3.8-27B-UD-Q8_K_XL.gguf \
	-ngl 59 -ts 1,1 \
	-c 81920 -ctk q8_0 -ctv q8_0 -ub 256 \
	-fa on \
	--jinja \
	-lv 4 --log-file llama-server.out \
	--spec-type draft-mtp --spec-draft-n-max 5 -np 1 \
	-sm tensor \
	--no-mmproj-offload
