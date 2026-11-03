/opt/llama.cpp/llama-server \
  -hf unsloth/Qwen3.8-27B-GGUF:Q8_0 \
  -ngl 66 -ts 1,1 \
  -c 81920 -ctk q8_0 -ctv q8_0 -ub 256 \
  -fa on \
  --jinja \
  -lv 4 --log-file llama-server.out \
  --spec-type draft-mtp --spec-draft-n-max 3 -np 1 \
  -sm tensor \
  --no-mmproj-offload
