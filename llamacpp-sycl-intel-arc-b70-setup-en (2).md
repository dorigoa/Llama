# llama.cpp SYCL on 2× Intel Arc Pro B70 (Ubuntu 26) — runtime setup and troubleshooting

Date: 2026-11-10 — llama.cpp build b12246 (prebuilt release "Ubuntu x64 SYCL FP16")
GPUs: 2× Arc Pro B70 32 GB (PCI ID 0xe223, BMG-G31) — host 192.168.1.200 — kernel 7.0.0-43-generic

## Backend choice

- SYCL FP16: Intel's native backend, usually faster than Vulkan on the B70 (FP16 speeds up prompt processing, not token generation).
- Vulkan: fallback; simpler to set up and more complete (historically more reliable on hybrid models such as Qwen3.5/3.6/3.8).
- SYCL FP32: only if FP16 shows numerical issues. OpenVINO: still immature for discrete GPUs.

## Runtime dependencies (Intel oneAPI APT repository)

```bash
wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
  | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg >/dev/null
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
  | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update
sudo apt install intel-oneapi-runtime-dpcpp-cpp intel-oneapi-runtime-mkl \
                 intel-oneapi-runtime-openmp intel-oneapi-runtime-dnnl intel-oneapi-umf
# libumf is installed outside the system library path → register it
echo /opt/intel/oneapi/umf/latest/lib | sudo tee /etc/ld.so.conf.d/intel-oneapi-umf.conf   # or .../umf/1.2/lib
sudo ldconfig
```

Level Zero loader: `libze1` 1.28.2 (Ubuntu). GPU compute runtime: **updated 2026-11-10** from Ubuntu's 26.05.37020.3 / IGC 2.28.4 to Intel GitHub release **26.35.39758.10 / IGC 2.41.5**:

```bash
mkdir -p ~/neo-26.35 && cd ~/neo-26.35
R=https://github.com/intel/compute-runtime/releases/download/26.35.39758.10
I=https://github.com/intel/intel-graphics-compiler/releases/download/v2.41.5
wget $I/intel-igc-core-2_2.41.5+22716_amd64.deb $I/intel-igc-opencl-2_2.41.5+22716_amd64.deb \
     $R/intel-ocloc_26.35.39758.10-0_amd64.deb $R/intel-opencl-icd_26.35.39758.10-0_amd64.deb \
     $R/libigdgmm12_22.10.0_amd64.deb $R/libze-intel-gpu1_26.35.39758.10-0_amd64.deb $R/ww35.sum
sha256sum -c --ignore-missing ww35.sum
sudo dpkg -i *.deb      # ldconfig "is not a symbolic link" warnings on oneAPI libs are harmless
```

No reboot needed (userspace only). Check what a running server actually loads:
`grep -E 'libze_intel_gpu|libigc' /proc/$(pgrep -f llama-server | head -1)/maps | awk '{print $6}' | sort -u`
→ `libze_intel_gpu.so.1.17.39758` (1.17 = Level Zero spec, 39758 = build), `/usr/local/lib/libigc.so.2.41.5…`.

## Root cause 1: silent fallback to OpenCL

- The Unified Runtime adapters `libur_adapter_level_zero*.so` (in `/opt/intel/oneapi/redist/lib`) are loaded at runtime (dlopen) and depend on `libumf.so.1`. Without UMF the Level Zero adapter fails to load and SYCL silently uses **OpenCL** instead.
- Symptoms observed on OpenCL:
  - `-sm tensor`: the server starts but output is garbage (chat: empty reply with `finish_reason: stop` after 1 token; `/completion` "La capitale della Francia è" → " ha, ha…").
  - `-sm layer`: crash during warmup in `dev2dev_memcpy` → `ext_oneapi_can_access_peer` → `ur::opencl::urUsmP2PPeerAccessGetInfoExp` → `die()`.
- `GGML_SYCL_DISABLE_OPT=1` was unrelated (no effect, only costs performance).
- With Level Zero active, both `-sm layer` and `-sm tensor` work.

## Root cause 2: GPU hangs during generation with MTP speculative decoding

- Symptom: llama-server stops mid-generation, GPU utilization drops to 0% with VRAM still full; the server must be restarted. Recurs every few minutes.
- `dmesg` (xe driver):
  - `Engine reset: engine_class=ccs … Timedout job … in llama-server` (CCS = compute engine);
  - with the old runtime 26.05 also `Fault response: Unsuccessful -EINVAL` + `Engine memory CAT error … class=bcs` (BCS = copy engine, invalid memory access) on both cards. These BCS faults **disappeared after updating to 26.35**; the CCS timeouts did not.
- Isolation (one variable at a time):
  - happens on a single card too (`ONEAPI_DEVICE_SELECTOR='level_zero:0'`) → not multi-GPU / P2P;
  - persists after the runtime update;
  - **removing `--spec-type draft-mtp -md …` → stable** (> 50 min of continuous generation, empty dmesg).
- Conclusion: the trigger is the MTP path on SYCL (Qwen3.8-27B, hybrid attention/GATED_DELTA_NET). Related upstream issue: ggml-org/llama.cpp#23203 (SYCL + MTP, root cause not identified there either).
- Baseline without MTP, 1× B70, Qwen3.8-27B UD-Q4_K_XL, 256K ctx, KV q8_0: **19.2 tok/s** generation, ~550 tok/s prompt processing.

## Useful diagnostics

```bash
ldd /opt/llama.cpp/llama-server | grep "not found" | sort -u            # direct dependencies
cd /opt/intel/oneapi/redist/lib
ldd libur_adapter_level_zero.so.0 libur_adapter_level_zero_v2.so.0 | grep -E ':$|not found'   # dependencies of runtime-loaded adapters
ONEAPI_DEVICE_SELECTOR='level_zero:*' /opt/llama.cpp/llama-server --list-devices   # empty/error = Level Zero not working
curl -s http://HOST:8080/completion -H 'Content-Type: application/json' \
  -d '{"prompt":"The capital of France is","n_predict":10,"temperature":0,"n_probs":5}' | jq   # numerical sanity check, bypasses the chat template
sudo dmesg -wT | grep -iE 'xe .*(reset|timedout|fault|CAT)'                 # live watch for GPU hangs/resets
grep -E 'eval time' ~/llama-server.out | tail -3                            # tok/s from the log
```

## Working command (Qwen3.8-27B UD-Q8_K_XL, 256K context)

```bash
ONEAPI_DEVICE_SELECTOR='level_zero:*' /opt/llama.cpp/llama-server --device SYCL0,SYCL1 \
  -a Qwen3.8-27b-ud-q8_k_xl -m /nvme/gguf/qwen3.8-27b/Qwen3.8-27B-UD-Q8_K_XL.gguf \
  -ngl 999 -ts 1,1 -c 262144 -ctk q8_0 -ctv q8_0 -ub 512 -fa on --jinja \
  -lv 4 --log-file /var/log/llama-server.out -sm tensor --no-mmproj-offload \
  --host 192.168.1.200 --port 8080 --temp 0.0
```

`-sm layer` works too. Always keep `ONEAPI_DEVICE_SELECTOR='level_zero:*'`: it forces Level Zero, so a future breakage fails loudly instead of silently falling back to OpenCL. **Do not use `--spec-type draft-mtp` on SYCL for now** (see root cause 2).

## To do

- Benchmark prompt processing / generation, `-sm tensor` vs `-sm layer`, 1 vs 2 cards.
- MTP on SYCL: retest on newer llama.cpp builds (and/or with `--spec-draft-n-max 1`); compare with the Vulkan backend + MTP.
- Drop `-lv 4` / `--log-file` for normal use.
