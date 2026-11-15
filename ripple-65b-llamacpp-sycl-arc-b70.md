# Ripple-65B Preview su 2× Intel Arc Pro B70 — llama.cpp SYCL

Data: 2026-11-14 — llama.cpp build b12246 (release precompilata "Ubuntu x64 SYCL FP16")
GPU: 2× Arc Pro B70 32 GB — host 192.168.1.200 — RAM 112 GB (109,7 usabili), DDR4 dual-channel

## Configurazione finale (15,7 tok/s in generazione, contesto 256K)

```bash
ONEAPI_DEVICE_SELECTOR='level_zero:*' /opt/llama.cpp/llama-server \
  --device SYCL0,SYCL1 -sm tensor -ts 1,1 \
  -m /nvme/gguf/ripple-65b-ac/Ripple-65B-Preview-AD-Q4_K_M.gguf \
  -ngl 999 -c 262144 -ctk f16 -ctv f16 -fa on \
  -b 2048 -ub 512 -np 1 --jinja --reasoning on \
  --host 0.0.0.0 --port 8080
```

| Configurazione | tok/s |
|---|---|
| `-sm layer`, 256K, KV f16, `-fa on` | 12,55 |
| `-sm tensor`, 256K, KV f16, `-fa on` | **15,67** |

KV in `f16` e non `q8_0`: a questa taglia di pesi ci sta, e togliere la KV quantizzata elimina una variabile nota come fragile su SYCL.

## Causa radice del primo fallimento: GGUF pre-merge

**Sintomo**: il modello carica senza un solo avviso, la velocità è quella attesa, ma l'output è solo newline (`"\n\n\n\n..."`), sia da `/completion` che da `/v1/chat/completions`.

**Causa**: il GGUF proveniva da un repo che aveva scritto una patch propria per l'architettura `ricursive` contro il commit upstream 6f1e654, **prima** del merge. Il supporto merged (PR #31692, prima release **b12167**) legge male qualche parametro di quei file. Nessun errore, nessun warning: degenerazione silenziosa.

**Rimedio**: usare i GGUF di `AtomicChat/Ripple-65B-Preview-GGUF` — sono gli autori del port upstream e dichiarano i file caricabili su build stock b12167+ senza patch, verificati contro il codice di riferimento di Ricursive.

File che **non** funziona: `mrfakename/Ripple-65B-Preview-GGUF` Q5_K_M (e presumibilmente gli altri dello stesso repo e di `prometheusAIR`, costruiti con la stessa patch).

**Regola generale**: per un'architettura appena aggiunta, controllare sempre se il GGUF è stato prodotto prima o dopo il merge upstream. Un mismatch di chiavi GGUF non genera diagnostica.

## Catena diagnostica (cosa è stato escluso, e come)

| Ipotesi | Test | Esito |
|---|---|---|
| Server bloccato | `llama-server` non termina mai, e `grep` su pipe bufferizza | Falso allarme: era in ascolto |
| KV cache `q8_0` | rilancio con `-ctk f16 -ctv f16` | scagionata |
| Flash attention su SYCL | rilancio con `-fa off` | scagionata |
| Backend SYCL / le B70 | `-ngl 0`, CPU pura, `--no-repack` | **scagionato** — stesso output degenere |
| Chiave GGUF letta male | `grep -iE 'warn\|missing\|unknown'` sul log | nessun avviso (il difetto è silenzioso) |
| Il file | stesso comando, GGUF di AtomicChat | **colpevole** |

Il test su CPU è stato il discriminante: sposta il sospetto a monte del backend in un colpo, costa ~1 tok/s ma con `n_predict 5` sono secondi.

Controllo di sanità che bypassa il template (lo stesso che smascherò il fallback a OpenCL):

```bash
curl -s http://127.0.0.1:8080/completion -H 'Content-Type: application/json' \
  -d '{"prompt":"The capital of France is","n_predict":5,"temperature":0}' | jq '.content'
```

Usare `jq '.content'` **senza** `-r`: i newline restano come `\n` letterali e si distinguono da caratteri corrotti.

## Memoria: numeri misurati, non stimati

VRAM totale: 2 × 32.656 MiB = **65.312 MiB (63,8 GiB)**.

Dalla tabella `common_memory_breakdown_print`:

| Voce | Valore |
|---|---|
| Pesi Q8_0 (64,11 GiB) | 65.647 MiB → **335 MiB più della VRAM totale** |
| Pesi Q5_K_M | 43.297 MiB (21.970 + 21.327) |
| Pesi AD-Q4_K_M | ~38.500 MiB |
| KV cache 256K `f16` | 16.768 MiB |
| KV cache 256K `q8_0` | 8.913 MiB |
| Compute buffer, `-ub 512`, `-fa off` | 2.264 + 2.214 = 4.478 MiB |

Il Q8_0 non entra **a nessun contesto**, nemmeno con KV nulla: i pesi da soli superano la VRAM totale di 335 MiB. Con `-ngl 20` finivano ~48 GiB in RAM DDR4 (~40 GB/s reali) → ordine di 1 tok/s, con due terzi della VRAM inutilizzati.

Il compute buffer è 2,2 GiB per scheda, non ~1,5: questo esclude anche il Q6_K (50.688 + 8.913 + 4.478 = 64.079 MiB, cioè ~616 MiB liberi per scheda, sotto il margine di 1.024 MiB/scheda che `--fit` tiene per default).

Budget pesi, partendo da 65.312 MiB e togliendo KV + compute + margine `--fit`:
- con KV `q8_0` a 256K: ~51 GiB
- con KV `f16` a 256K: ~42,5 GiB

| Quant | Dimensione | A 256K |
|---|---|---|
| Q8_0 | 64,1 GiB | impossibile |
| AD-Q5_K_M | 44,6 GiB | solo con KV `q8_0` |
| AD-Q4_K_M | 37,6 GiB | sì, anche con KV `f16` |
| Q4_K_M (non-AD) | 36,1 GiB | sì |

Perplessità misurata dai repo: Q8_0 8,81, Q6_K 9,14, Q5_K_M 8,74, Q4_K_M 9,13, BF16 8,82 — con barre d'errore ±0,4 i riordinamenti fra queste taglie sono rumore. Non vale pagare VRAM per salire di quant.

## Architettura (dai metadati GGUF)

```
general.architecture              = ricursive
ricursive.block_count             = 80
ricursive.context_length          = 2097152
ricursive.embedding_length        = 5120
ricursive.feed_forward_length     = 40960
ricursive.attention.head_count    = 80
ricursive.attention.head_count_kv = 8
ricursive.attention.key_length    = 128
ricursive.attention.value_length  = 128
ricursive.attention.sliding_window         = 1024
ricursive.attention.sliding_window_pattern = [true,true,true,true,false,...]  (80)
ricursive.rope.freq_base          = 10000
ricursive.embedding_scale         = 71.554176   (= √5120)
ricursive.swiglu_clamp            = 20.0        (scalare, non array)
ricursive.vocab_size              = 100352
tokenizer.ggml.model / pre        = gpt2 / dbrx
bos 100257 · eos/eot 100265 · pad 100277
```

Attenzione **ibrida**: 4 layer a finestra scorrevole (1.024 token) + 1 globale, quindi 64 locali e 16 globali. RoPE applicata solo ai layer a finestra, omessa sui globali — per questo `freq_base = 10000` basta anche a 1M di contesto, senza scaling.

La KV cache a 256K torna esattamente:
- 16 layer globali × 262.144 token × 4 KiB = 16.384 MiB
- 64 layer a finestra × (1.024 + 512) token × 4 KiB = 384 MiB
- somma 16.768 MiB = identica al misurato

4 KiB per layer per token = 2 (K e V) × 8 head × 128 dim × 2 byte. Senza l'ottimizzazione a finestra la cache sarebbe 5× più grande (82 GiB) e nulla starebbe in VRAM: llama.cpp la implementa correttamente.

## Flag: cosa serve e cosa no

Da `/props` → `chat_template_caps`:

```json
"supports_reasoning_effort": false,
"reasoning_effort_levels": [],
"supports_preserve_reasoning": true,
"supports_tool_calls": true,
"supports_parallel_tool_calls": true
```

- `--reasoning-effort` è **inutile** con Ripple: il template non distingue i livelli. Il server accetta il flag e non fa nulla. Per accorciare il ragionamento serve `--reasoning-budget N` (budget in token), indipendente dal template.
- `--reasoning on` serve: separa il ragionamento in `reasoning_content`.
- Tool calling nativo supportato, anche in parallelo.
- `--fit on` è già il default e regola solo gli argomenti **non** impostati: con `-c`, `-ngl`, `-b`, `-ub` espliciti non ha nulla da fare.
- `-fa` prende un valore da b6325 (`on|off|auto`): un `-fa` nudo consuma la parola successiva. `-fa 1` è da evitare, usare `-fa on`.
- `--no-mmap` non esiste più, è `-lm none` / `--load-mode none`. Da **non** usare qui: esiste la issue #26110 sullo swapping dei pesi residenti in CPU, e il default (`auto` = mmap) è preferibile.
- `-ctk f16 -ctv f16` sono già i default.
- Senza `-np 1` il server parte con 4 slot; con `kv_unified = 'true'` la cache è condivisa e non si moltiplica per 4, ma 1 slot resta più pulito.
- Su SYCL `-sm tensor` **richiede** `-fa on`: con `--flash-attn off` viene rifiutato alla creazione del contesto. Il parallelismo tensoriale è ottimizzato per 2 GPU.

## Da fare

- Misurare il prefill su prompt lunghi (decine di migliaia di token), `-ub 512` vs 256 vs 1024.
- Verificare se il problema dei blocchi in generazione (engine reset CCS, vedi scheda SYCL/B70) si manifesta anche con questo modello su generazioni lunghe.
- Drafter DFlash2: esiste in GGUF (`ashbash/Ripple-65B-Preview-DFlash2-GGUF`, Q8_0 4,3 GB) ma è documentato per `gmlx`, non per llama.cpp. In più la issue #29419 riporta un abort SYCL "Not support Flash-Attention" in `fattn.cpp` proprio durante il draft decoding **con tensor split**, che è la configurazione attuale. Da riprovare solo dopo che quel percorso è sistemato.
- Valutare AD-Q5_K_M + KV `q8_0` a 256K: entra con ~6 GiB di margine, ma la perplessità dice che non cambia nulla.
