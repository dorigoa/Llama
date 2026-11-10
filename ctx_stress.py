#!/usr/bin/env python3
"""
ctx_stress.py - stress test di llama-server riempiendo il contesto.

Costruisce un prompt di lunghezza esatta in token (testo pseudo-casuale,
tokenizzato dal server stesso via /tokenize) e lo invia a /completion.

Modalita':
  single (default)  un prompt al FILL% di n_ctx, cache disattivata
  --ramp K          K richieste a lunghezza crescente sullo stesso testo
                    (prefisso comune, cache_prompt attiva): simula una chat
                    che cresce e stressa riuso cache / checkpoint
  --to-limit        genera fino a riempire ESATTAMENTE n_ctx (prompt + output)
  --overflow        prompt piu' lungo di n_ctx: verifica la gestione d'errore

Solo libreria standard. Exit code: 0 = tutte le richieste OK, 1 = errori,
2 = errore di configurazione/connessione, 130 = interrotto.
"""

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request

WORDS = (
    "the of and to in is that for it as with was on be by this are from or "
    "have an they which one you were all we her she there been their has "
    "would will if can more when what so about up out into time only new "
    "some could these two may first then do any like my now over such our "
    "man me even most made after also did many before must through back "
    "years where much your way well down should because each just those "
    "people how too little state good very make world still own see men "
    "work long get here between both life being under never day same "
    "another know while last might us great old year off come since against "
    "go came right used take three system kernel memory cluster storage "
    "network switch packet latency bandwidth throughput tensor matrix "
    "gradient photon electron quantum entropy galaxy nebula telescope "
    "exposure aperture sensor lens wildlife eagle heron fox mountain river "
    "il la di che e un una per con non sono come anche questo quello "
    "della delle nel nella tra fra molto sempre dopo prima ancora tempo "
    "calcolo sistema rete memoria disco processo energia luce stella"
).split()


def log(msg):
    print(msg, flush=True)


def http(method, url, payload=None, api_key=None, timeout=60):
    """Ritorna (status, json|dict). Solleva URLError/TimeoutError per problemi di rete."""
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
            return r.status, (json.loads(body) if body else {})
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, {"raw": raw[:2000]}


def get_n_ctx(base, key):
    st, j = http("GET", base + "/props", api_key=key)
    if st == 200 and isinstance(j, dict):
        n = (j.get("default_generation_settings") or {}).get("n_ctx")
        if n:
            return int(n), j.get("total_slots"), j.get("model_alias") or j.get("model_path")
    st, j = http("GET", base + "/slots", api_key=key)
    if st == 200 and isinstance(j, list) and j and "n_ctx" in j[0]:
        return int(j[0]["n_ctx"]), len(j), None
    raise RuntimeError(f"impossibile leggere n_ctx da /props o /slots (HTTP {st}): {j}")


def make_text(rng, n_words):
    out = []
    for _ in range(n_words):
        r = rng.random()
        if r < 0.03:
            w = str(rng.randint(0, 99999))
        else:
            w = rng.choice(WORDS)
        if r > 0.94:
            w += "."
        elif r > 0.90:
            w += ","
        out.append(w)
        if r > 0.997:
            out.append("\n\n")
    return " ".join(out)


def tokenize(base, key, text, add_special):
    st, j = http("POST", base + "/tokenize",
                 {"content": text, "add_special": add_special, "parse_special": False},
                 api_key=key, timeout=600)
    if st != 200 or "tokens" not in j:
        raise RuntimeError(f"/tokenize fallito (HTTP {st}): {j}")
    toks = j["tokens"]
    # con with_pieces=true sarebbero oggetti {id, piece}: normalizza
    return [t["id"] if isinstance(t, dict) else int(t) for t in toks]


def build_tokens(base, key, target, seed):
    """Lista di esattamente `target` token: [BOS se previsto] + testo casuale."""
    prefix = tokenize(base, key, "", add_special=True)  # [BOS] o []
    rng = random.Random(seed)
    tokens = list(prefix)
    while len(tokens) < target:
        need = target - len(tokens)
        n_words = min(max(int(need * 0.9) + 200, 500), 150_000)
        tokens.extend(tokenize(base, key, make_text(rng, n_words), add_special=False))
    return tokens[:target]


def run_one(base, key, tokens, n_predict, args, label):
    payload = {
        "prompt": tokens,
        "n_predict": n_predict,
        "cache_prompt": args.cache,
        "ignore_eos": not args.allow_eos,
        "temperature": args.temperature,
        "seed": args.seed,
        "stream": False,
    }
    log(f"[{label}] prompt={len(tokens)} tok, n_predict={n_predict}, "
        f"cache_prompt={args.cache} ... invio")
    t0 = time.monotonic()
    st, j = http("POST", base + "/completion", payload, api_key=key, timeout=args.timeout)
    wall = time.monotonic() - t0

    rec = {"label": label, "http": st, "wall_s": round(wall, 2),
           "prompt_sent": len(tokens), "n_predict": n_predict}
    if st != 200:
        err = j.get("error", j) if isinstance(j, dict) else j
        rec["error"] = err
        log(f"[{label}] ERRORE HTTP {st} dopo {wall:.1f}s: {json.dumps(err)[:600]}")
        return False, rec

    t = j.get("timings", {}) or {}
    content = j.get("content", "") or ""
    rec.update({
        "tokens_evaluated": j.get("tokens_evaluated"),
        "tokens_predicted": j.get("tokens_predicted"),
        "tokens_cached": j.get("tokens_cached"),
        "stop_type": j.get("stop_type"),
        "truncated": j.get("truncated"),
        "timings": t,
        "content_tail": content[-args.tail:],
    })
    log(f"[{label}] OK in {wall:.1f}s | eval={j.get('tokens_evaluated')} "
        f"cache_n={t.get('cache_n')} pred={j.get('tokens_predicted')} "
        f"stop={j.get('stop_type')} truncated={j.get('truncated')}")
    log(f"[{label}]   pp={t.get('prompt_per_second', 0):.1f} t/s "
        f"({t.get('prompt_n')} tok, {t.get('prompt_ms', 0)/1000:.1f}s) | "
        f"tg={t.get('predicted_per_second', 0):.2f} t/s ({t.get('predicted_n')} tok)")
    tail = content[-args.tail:].replace("\n", "\\n")
    log(f"[{label}]   coda output: ...{tail}")
    return True, rec


def main():
    ap = argparse.ArgumentParser(description="Stress test llama-server riempiendo il contesto.")
    ap.add_argument("--url", default="http://127.0.0.1:8080", help="base URL del server")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--fill", type=float, default=0.95,
                    help="frazione di n_ctx occupata dal prompt (default 0.95)")
    ap.add_argument("--prompt-tokens", type=int, default=None,
                    help="lunghezza prompt esplicita in token (ignora --fill)")
    ap.add_argument("--n-predict", type=int, default=256)
    ap.add_argument("--to-limit", action="store_true",
                    help="n_predict = n_ctx - prompt: riempie il contesto fino all'ultimo token")
    ap.add_argument("--ramp", type=int, default=0,
                    help="K richieste a lunghezza crescente (prefisso comune), attiva cache_prompt")
    ap.add_argument("--overflow", action="store_true",
                    help="prompt = n_ctx + 64 token: deve dare errore pulito, non crash")
    ap.add_argument("--runs", type=int, default=1, help="ripetizioni dell'intero test")
    ap.add_argument("--cache", action="store_true", help="cache_prompt=true (default false)")
    ap.add_argument("--allow-eos", action="store_true",
                    help="permetti stop su EOS (default: ignore_eos per forzare la generazione)")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--timeout", type=float, default=7200, help="timeout per richiesta (s)")
    ap.add_argument("--tail", type=int, default=200, help="caratteri finali di output da mostrare")
    ap.add_argument("--out", default="ctx_stress.jsonl", help="log JSONL dei risultati")
    args = ap.parse_args()

    base = args.url.rstrip("/")
    if not 0.0 < args.fill <= 1.0:
        ap.error("--fill deve essere in (0, 1]")
    if args.ramp:
        args.cache = True

    try:
        st, h = http("GET", base + "/health", api_key=args.api_key, timeout=10)
        if st != 200:
            log(f"/health -> HTTP {st}: {h}")
            return 2
        n_ctx, n_slots, model = get_n_ctx(base, args.api_key)
    except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as e:
        log(f"Connessione/config fallita su {base}: {e}")
        return 2

    log(f"Server {base} | modello: {model} | n_ctx per slot: {n_ctx} | slot: {n_slots}")

    if args.overflow:
        p_len = n_ctx + 64
    elif args.prompt_tokens:
        p_len = args.prompt_tokens
    else:
        p_len = int(n_ctx * args.fill)
    if not args.overflow and p_len >= n_ctx:
        p_len = n_ctx - 1

    n_predict = max(1, n_ctx - p_len) if args.to_limit else args.n_predict
    if not args.overflow and p_len + n_predict > n_ctx:
        log(f"Nota: prompt+n_predict = {p_len + n_predict} > n_ctx = {n_ctx} "
            f"(il server fermera' la generazione al limite)")

    try:
        log(f"Costruisco {p_len} token via /tokenize ...")
        t0 = time.monotonic()
        tokens = build_tokens(base, args.api_key, p_len, args.seed)
        log(f"Prompt pronto in {time.monotonic() - t0:.1f}s")
    except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as e:
        log(f"Costruzione prompt fallita: {e}")
        return 2

    if args.ramp:
        steps = [max(1, round(p_len * (i + 1) / args.ramp)) for i in range(args.ramp)]
    else:
        steps = [p_len]

    failures = 0
    try:
        with open(args.out, "a", encoding="utf-8") as fout:
            for r in range(args.runs):
                for i, n in enumerate(steps):
                    npred = max(1, n_ctx - n) if args.to_limit else n_predict
                    label = f"run{r + 1}/{args.runs} step{i + 1}/{len(steps)}"
                    try:
                        ok, rec = run_one(base, args.api_key, tokens[:n], npred, args, label)
                    except (urllib.error.URLError, TimeoutError, OSError) as e:
                        ok, rec = False, {"label": label, "exception": repr(e)}
                        log(f"[{label}] ECCEZIONE di rete: {e!r}  (server crashato?)")
                    rec["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                    if args.overflow:
                        # qui l'errore HTTP 400 e' il comportamento ATTESO
                        ok = (rec.get("http") == 400)
                        log(f"[{label}] overflow: {'gestito correttamente' if ok else 'COMPORTAMENTO ANOMALO'}")
                    failures += 0 if ok else 1
                    # il server e' ancora vivo?
                    try:
                        st, _ = http("GET", base + "/health", api_key=args.api_key, timeout=10)
                        if st != 200:
                            log(f"[{label}] /health -> HTTP {st}")
                    except (urllib.error.URLError, TimeoutError, OSError) as e:
                        log(f"[{label}] server NON raggiungibile dopo la richiesta: {e}")
                        return 1
    except KeyboardInterrupt:
        log("Interrotto.")
        return 130

    log(f"Fine: {failures} richieste fallite. Log: {args.out}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
