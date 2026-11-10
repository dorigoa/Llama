# llama.cpp SYCL on 2× Intel Arc Pro B70 (Ubuntu 26) — runtime setup and troubleshooting

Date: 2026-11-10 — llama.cpp build b12246 (prebuilt release "Ubuntu x64 SYCL FP16")
GPUs: 2× Arc Pro B70 32 GB (PCI ID 0xe223, BMG-G31) — host 192.168.1.200

## Backend choice

- SYCL FP16: Intel's native backend, usually faster than Vulkan on the B70 (FP16 speeds up prompt processing, not token generation).
- Vulkan: fallback; simpler to set up and more complete (historically more reliable on hybrid models such as Qwen3.5/3.6/3.8).
- SYCL FP32: only if FP16 shows numerical problems. OpenVINO: still immature for discrete GPUs.

## Runtime dependencies (Intel oneAPI APT repository)

```bash
wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
  | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg >/dev/null
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
  | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update
sudo apt install intel-oneapi-runtime-dpcpp-cpp intel-oneapi-runtime-mkl \
                 intel-oneapi-runtime-openmp intel-oneapi-runtime-dnnl intel-oneapi-umf
# libumf is installed outside the system library paths → register it
echo /opt/intel/oneapi/umf/latest/lib | sudo tee /etc/ld.so.conf.d/intel-oneapi-umf.conf   # or .../umf/1.2/lib
sudo ldconfig
```

Level Zero driver (from the Ubuntu repositories): `libze1` 1.28.2, `libze-intel-gpu1` 26.05.37020.3, plus `intel-opencl-icd`.

## Root cause: silent fallback to OpenCL

- The Unified Runtime adapters `libur_adapter_level_zero*.so` (in `/opt/intel/oneapi/redist/lib`) are loaded at runtime (dlopen) and depend on `libumf.so.1`. Without UMF the Level Zero adapter fails to load and SYCL silently uses **OpenCL** instead.
- Symptoms seen while running on OpenCL:
  - `-sm tensor`: the server starts but the output is garbage (chat: empty reply with `finish_reason: stop` after 1 token; `/completion` "La capitale della Francia è" → " ha, ha…").
  - `-sm layer`: crash during warmup in `dev2dev_memcpy` → `ext_oneapi_can_access_peer` → `ur::opencl::urUsmP2PPeerAccessGetInfoExp` → `die()`.
- `GGML_SYCL_DISABLE_OPT=1` was unrelated (no effect, only costs performance).
- With Level Zero active, both `-sm layer` and `-sm tensor` work.

## Useful diagnostics

```bash
ldd /opt/llama.cpp/llama-server | grep "not found" | sort -u            # direct dependencies
cd /opt/intel/oneapi/redist/lib
ldd libur_adapter_level_zero.so.0 libur_adapter_level_zero_v2.so.0 | grep -E ':$|not found'   # dependencies of the runtime-loaded adapters
ONEAPI_DEVICE_SELECTOR='level_zero:*' /opt/llama.cpp/llama-server --list-devices   # empty/error = Level Zero not working
curl -s http://HOST:8080/completion -H 'Content-Type: application/json' \
  -d '{"prompt":"The capital of France is","n_predict":10,"temperature":0,"n_probs":5}' | jq   # numerical sanity check, bypasses the chat template
```

## Working command (Qwen3.8-27B UD-Q8_K_XL, 256K context)

```bash
ONEAPI_DEVICE_SELECTOR='level_zero:*' /opt/llama.cpp/llama-server --device SYCL0,SYCL1 \
  -a Qwen3.8-27b-ud-q8_k_xl -m /nvme/gguf/qwen3.8-27b/Qwen3.8-27B-UD-Q8_K_XL.gguf \
  -ngl 999 -ts 1,1 -c 262144 -ctk q8_0 -ctv q8_0 -ub 512 -fa on --jinja \
  -lv 4 --log-file /var/log/llama-server.out -sm tensor --no-mmproj-offload \
  --host 192.168.1.200 --port 8080 --temp 0.0
```

`-sm layer` works too. Always keep `ONEAPI_DEVICE_SELECTOR='level_zero:*'`: it forces Level Zero, so a future breakage fails loudly instead of silently falling back to OpenCL.

## To do

- Benchmark prompt processing / token generation with `-sm tensor` vs `-sm layer`; evaluate MTP (`--spec-type draft-mtp`) as on the CUDA PC.
- Drop `-lv 4` / `--log-file` for normal use.
