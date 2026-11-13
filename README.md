## Notes on installation for SYCL (Intel ARC B70) on Ubuntu 26.10
For llama.cpp (1.0.0) download as pre-built binaries, ubuntu 26.10, SVML is required; we will install the openAPI stack:
```
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
  | sudo gpg --dearmor -o /usr/share/keyrings/oneapi-archive-keyring.gpg

echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
  | sudo tee /etc/apt/sources.list.d/oneAPI.list

sudo apt update

sudo apt install intel-oneapi-runtime-libs

ldconfig -p | grep libsvml

ldd <path>/llama-server -h
```

Checked that after installing libraries system-wide, strata still works.

## Run Qwen3.8-Flash-Next Q4
Downloaded UD-IQ4_XS (v3.1) from `https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF?show_file_info=UD-IQ4_XS%2FQwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf` (one of them is the quantised n-gram dictionary). MTP is embedded, but can also use an external one (see below).
Options:
```
-ot "ple_ngram_embd=CPU,per_layer_token_embd=CPU"

--n-cpu-moe N 

--load-mode none --fit off # on mixed env it makes mistakes istimating the distribution

-ngl 99 --split-mode layer --tensor-split 1,1

-fa on -ctk q8_0 -ctv q8_0

-c 65536 per iniziare --no-context-shift, --parallel 1

-b 4096 -ub 2048 # Costa VRAM

-md mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 2

--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --jinja --metrics
```

## Test to get stats

```bash
curl -s http://localhost:8080/v1/chat/completions   -H 'Content-Type: application/json'   -d '{"messages":[{"role":"user","content":"Spiegami la relatività ristretta in 600 parole"}],"max_tokens":1000}'   | jq .timings
```
## Stress test
```bash
seq 1 500 | xargs -P 4 -I{} bash -c 'n=$((RANDOM%800+100)); p=$(shuf -n $((RANDOM%40+5)) /usr/share/dict/words 2>/dev/null | tr "\n" " " || echo "spiega la meccanica statistica"); curl -s --max-time 180 -o /dev/null -w "[%{http_code}] %{time_total}s\n" http://192.168.1.200:8080/v1/chat/completions -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Scrivi un testo dettagliato su: $p\"}],\"max_tokens\":$n}"'
```
