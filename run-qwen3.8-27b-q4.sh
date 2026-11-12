#podman run --rm -it --name llama-cockpit-server-72cb15ef --device /dev/dri --group-add keep-groups --security-opt seccomp=unconfined --security-opt label=disable --userns=keep-id -e XDG_CACHE_HOME=/tmp -v /nvme/gguf:/models:ro -p 192.168.1.200:8080:8080 docker.io/kyuz0/intel-b70-ai-toolboxes:sycl llama-server -m /models/Qwen3.8-27B-UD-Q4_K_XL.gguf -c 67840 -ngl 999 --host 0.0.0.0 --port 8080 --load-mode none -fa 1 --cache-type-k f16 --cache-type-v f16 -b 2048 -ub 512 -np 1 --jinja --reasoning on --chat-template-kwargs {"reasoning_effort":"medium"} --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --jinja --fit off --cache-ram 0 --spec-type draft-mtp --spec-draft-n-max 3 -dev SYCL0,SYCL1


#-dev is not required as llama-server takes by default all GPU podman provides (-dev /dev/dri)


#podman run --rm -it --name llama-cockpit-server-72cb15ef --device /dev/dri --group-add keep-groups --security-opt seccomp=unconfined --security-opt label=disable --userns=keep-id -e XDG_CACHE_HOME=/tmp -v /nvme/gguf:/models:ro -p 192.168.1.200:8080:8080 docker.io/kyuz0/intel-b70-ai-toolboxes:sycl llama-server -m /models/Qwen3.8-27B-UD-Q4_K_XL.gguf -c 262144 -ngl 999 --host 0.0.0.0 --port 8080 --load-mode none -fa 1 --cache-type-k f16 --cache-type-v f16 -b 2048 -ub 512 -np 1 --jinja --reasoning on --chat-template-kwargs '{"reasoning_effort":"medium"}' --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --jinja --fit off --cache-ram 0 --spec-type draft-mtp --spec-draft-n-max 5 -sm tensor
  
# tensor split mode produced 40 tok/s but after a little while 1 GPU hung
   

podman run --rm -it --name llama-cockpit-server-72cb15ef \
  --device /dev/dri \
  --group-add keep-groups \
  --security-opt seccomp=unconfined \
  --security-opt label=disable \
  --userns=keep-id \
  -e XDG_CACHE_HOME=/tmp \
  -v /nvme/gguf:/models:ro \
  -p 192.168.1.200:8080:8080 \
  docker.io/kyuz0/intel-b70-ai-toolboxes:sycl \
  llama-server \
  -m /models/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  -c 262144 -ngl 999 \
  --host 0.0.0.0 --port 8080 \
  --load-mode none \
  -fa 1 \
  --cache-type-k f16 --cache-type-v f16 \
  -b 2048 -ub 512 \
  -np 1 \
  --jinja \
  --reasoning on \
  --chat-template-kwargs '{"reasoning_effort":"medium"}' \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 \
  --jinja \
  --fit off \
  --cache-ram 0 \
  --spec-type draft-mtp --spec-draft-n-max 5 \
  --chat-template-file /models/qwen3.8-27b.jinja \
  -sm layer
