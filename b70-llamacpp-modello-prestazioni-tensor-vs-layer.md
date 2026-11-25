# 2× Arc Pro B70 — modello prestazionale llama.cpp SYCL: tensor vs layer split, taratura MTP

2026-11-25. Chiude il to-do "Benchmark prompt processing / generation, `-sm tensor` vs
`-sm layer`, 1 vs 2 schede" di `llamacpp-sycl-intel-arc-b70-setup.md`.

## Setup

Qwen3.8-27B **UD-Q8_K_XL**, file 30001 MiB = **31,46 GB**. 2× Intel Arc Pro B70 32 GB
(GDDR6, bus 256 bit, 8 canali). `-c 262144`, KV f16, `-fa 1`, `-np 1`,
`--spec-type draft-mtp`, `GGML_SYCL_DEV2DEV_MEMCPY=2`. VRAM a regime 25122 + 24477 MiB
su 32656 → ~7,6 GB liberi per scheda.

## Il modello

Con decodifica speculativa MTP il tempo di un ciclo è **lineare nella profondità del draft**:

```
ms/ciclo = V + n·d          tok/s = (1 + accettati_per_ciclo) / (ms/ciclo)
```

- **V** = verify forward pass, indipendente da n (legge i pesi del modello)
- **d** = costo di un passo di drafter (rilegge la testa di output, ~1,3 GB, vocabolario ~248K)

| split | V | d |
|---|---:|---:|
| `-sm tensor` | **51,6 ms** | **4,48 ms** |
| `-sm layer` | **59,8 ms** | **7,17 ms** |

## Misure (temp 0, stesso prompt, predicted_n = 2000)

A `temp 0` la speculazione è lossless: a parità di split il testo è identico, quindi i run
sono confrontabili senza ripetizioni. Pavimento di rumore misurato su due run identici:
**0,3%** (42,118 vs 41,995 tok/s).

| split | n-max | cicli | accettati/ciclo | token/forward | ms/ciclo | tok/s |
|---|---:|---:|---:|---:|---:|---:|
| tensor | 3 | 731 | 1,736 | 2,736 | 65,0 | **42,12** |
| tensor | 4 | 685 | 1,920 | 2,920 | 69,84 | 41,81 |
| tensor | 8 | 678 | 1,950 | 2,950 | 87,4 | 33,76 |
| layer | 4 | 707 | 1,829 | 2,829 | 88,48 | 31,97 |
| layer | 8 | 666 | 2,003 | 3,003 | 117,16 | 25,63 |

Cicli = `predicted_n − draft_n_accepted`; profondità effettiva = `draft_n / cicli`.

**Validazione:** V e d del tensor split tarati su n=3 e n=8 prevedono n=4 a 69,5 ms contro
69,84 misurati (**0,5% di errore**).

Attenzione: cambiando split i testi divergono anche a `temp 0` — l'ordine delle riduzioni in
virgola mobile cambia, i logit si discostano e la greedy prende prima o poi un token diverso.
Non inquina i tempi di ciclo (dipendono dai byte letti, non dal contenuto), ma l'acceptance
va letta per singolo run.

## Banda effettiva — il risultato principale

| configurazione | GB/s per scheda | % del nominale |
|---|---:|---:|
| `-sm layer` (15,73 GB in 29,9 ms) | **526** | **86–91%** |
| `-sm tensor` (15,73 GB in 51,6 ms) | **305** | ~50–58% |

Il nominale non è esposto da `xpu-smi discovery` (riporta solo GDDR6, bus 256 bit, 8 canali).
Si deduce: 256 bit = 32 byte/trasferimento, e i 526 GB/s misurati escludono memoria a
16 GT/s (darebbe 512 GB/s, cioè >100% di efficienza) → **18–19 GT/s, 576–608 GB/s**.

Conseguenze:

1. **Hardware e kernel SYCL sono sani**: 86–91% del nominale in layer split è nel range di un
   backend maturo su decode memory-bound.
2. **Il tensor split perde 1,72× di efficienza per scheda**, perché dimezza la larghezza di
   ogni GEMV (prodotto matrice-vettore, l'operazione dominante in decode).
3. **Tensor vince comunque, ma di poco**: raddoppia il parallelismo a 1/1,72 di efficienza →
   guadagno netto 2/1,72 = 1,16×, confermato da V_tensor 51,6 / V_layer 59,8 = 1,16.
   In tok/s: 41,81 vs 31,97 a n=4, **+31%** (il resto viene da d, 4,48 vs 7,17 ms).
4. Il formato di quantizzazione è **scagionato**: stesso file e stessi kernel nei due split,
   quindi il deficit non è nella dequantizzazione dei blocchi K-quant. Provare `Q8_0` liscio
   al posto di `UD-Q8_K_XL` non darebbe nulla.

Materiale per una issue upstream: *su Arc, il tensor split SYCL ottiene il 58% della banda per
scheda contro l'86–91% del layer split, a parità di modello e kernel.*

## Taratura di `--spec-draft-n-max`

La coda del drafter **si spegne** oltre la terza posizione, non decade: pos. 3 vale 0,184
token/ciclo, le pos. 4–7 **insieme** valgono 0,030. Coerente con `nextn_predict_layers = 1`
(una sola testa MTP riapplicata autoregressivamente sul proprio stato).

Ottimo in produzione (`temp 1.0`), calcolato con V e d tarati a temp 0 applicati ai contatori
`spec_decode_num_accepted_tokens_per_pos_total` di una sessione reale (5803 cicli;
contributi per posizione 0,669 / 0,415 / 0,258 / 0,176 / 0,139):

| n-max | tok/s previsti |
|---:|---:|
| 2 | 34,4 |
| 3 | 36,0 |
| **4** | **36,2** |
| 5 | 35,9 |
| 6 | ~35,3 |

**Validazione indipendente:** previsti 35,9 tok/s a n=5, misurati in sessione **35,7** (0,6%
di scarto), su un dataset diverso da quello di taratura (temp 1.0, mix di lavoro reale).

L'ottimo è **piatto fra 3 e 5 entro l'1%**: `--spec-draft-n-max 5` era già quasi ottimale,
passare a 4 vale +0,8%. Oltre, crolla: n=8 costa −20%.

**MTP vale 2,2×.** Senza drafter: 1 token per ciclo da 51,6 ms = **19,4 tok/s**. È l'argomento
quantitativo per tenerlo acceso malgrado i reset `ccs` di `b70-engine-reset-sintesi.md`.

## Parametri senza effetto, verificati

- **KV `q8_0`**: inutile all'uso reale misurato (`n_tokens_max` 4347, cioè ~4K di contesto).
  A quella lunghezza la KV cache è trascurabile contro i 31,46 GB di pesi letti a ogni forward
  pass. Tornerebbe utile solo a contesti di decine di migliaia di token.
- **`-ub` / `-b`**: agiscono **solo sul prefill**, non sulla generazione. Non misurati — la
  sessione di test aveva `prompt_tokens_total 47`. Con ~7,6 GB liberi per scheda c'è margine
  per alzarli, ma non è la leva sul decode.
- **`speculative.n_max` nel body JSON della richiesta: ignorato** da questa build. Per cambiare
  la profondità del draft serve il riavvio del server.

## Non misurato, ma con effetto atteso reale

`--cache-ram 0` disabilita la cache dei prompt lato host: nelle conversazioni multi-turno
costringe a riprocessare il prefill ogni volta. Riabilitarla non tocca il decode.

## Dove sta il fattore 1,7–2×

| leva | tok/s attesi | natura |
|---|---:|---|
| attuale (tensor, n=4, temp 1.0) | ~36 | — |
| tensor split all'efficienza del layer split | ~72 | limite backend SYCL, non tarabile |
| UD-Q4_K_XL (dimezza i byte letti) | ~84 | stima; costo in qualità |

Tetto con speculazione perfetta a n=3 e V attuale: ~61 tok/s. Il collo di bottiglia è la
banda, non i parametri.

## Metodologia

Conferma e rafforza le regole di `b70-engine-reset-sintesi.md`.

- **Benchmarkare a `temp 0`**, non a `temp 1.0`: la speculazione preserva la distribuzione, a
  temperatura zero il testo è identico e un run per lato basta. A `temp 1.0` due run della
  stessa configurazione differiscono fino al 20% per sola varianza di testo — durante questa
  indagine un "+19,7% da `n-max` 3" si è rivelato interamente varianza.
- **Leggere i tempi di ciclo, non i tok/s.** Ricavare `ms/ciclo = token_per_forward / tok_s` e
  `cicli = predicted_n − draft_n_accepted` separa l'effetto del parametro dall'effetto
  dell'acceptance.
- **Non confrontare run con testi diversi** per stimare V e d: è così che il modello è stato
  sbagliato due volte prima di convergere.
- A `temp 0` l'acceptance è più alta che in produzione, quindi i risultati **favoriscono**
  `n-max` grandi: l'ottimo operativo va calcolato sui contatori per posizione a `temp 1.0`,
  non trasferito direttamente.
- **Evitare query sysman (`xpu-smi stats`, telegraf) durante i benchmark e le sessioni lunghe**
  — vedi sotto.

## Nota sui reset `ccs` (da confermare)

Ipotesi dell'utente, 2026-11-25: i reset `ccs` sarebbero innescati dalla **lettura periodica
dello stato GPU da parte di telegraf** (% utilizzo, temperatura, potenza). Da quando telegraf
è spento, un watcher sul PC non ha più visto il problema. **Non ancora confermato con
precisione.**

L'ipotesi è coerente con *tutte* le osservazioni del dossier, cosa che nessuna precedente
spiegava: il polling è a intervallo fisso e indipendente dal backend, quindi rende conto della
stocasticità (MTBF ~26 min di busy), del fatto che accadesse anche su scheda singola (quindi
non P2P), dell'alternanza fra le due schede, del guasto anche su build **Vulkan**, e del ruolo
di MTP come amplificatore e non causa — più submission in volo, più probabilità di collidere
con una query di telemetria. Era l'unico elemento comune a tutti i run mai variato.

Se si conferma, MTP torna utilizzabile senza rischio: **+2,2× sul decode**, più di qualunque
parametro tarato qui.
