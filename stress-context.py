#!/usr/bin/env python3
"""Riempie il contesto di llama-server e verifica che non vada in OOM."""
import argparse, json, sys, time, urllib.request, urllib.error

def call(url, path, payload=None, timeout=60):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url + path, data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code} su {path}: {e.read().decode(errors='replace')[:500]}")
    except urllib.error.URLError as e:
        sys.exit(f"Connessione fallita su {path} (server crashato?): {e.reason}")
    except (OSError, json.JSONDecodeError) as e:
        sys.exit(f"Errore su {path}: {e!r}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:11000")
    ap.add_argument("--fill", type=float, default=0.95, help="frazione del contesto da riempire")
    ap.add_argument("--timeout", type=int, default=3600, help="secondi massimi per la richiesta")
    a = ap.parse_args()

    props = call(a.url, "/props")
    n_ctx = props.get("default_generation_settings", {}).get("n_ctx") or props.get("n_ctx")
    if not n_ctx:
        sys.exit("n_ctx non trovato in /props")
    target = int(n_ctx * a.fill)

    block = " ".join(f"Frase {i}: il gatto dorme sul divano mentre fuori piove." for i in range(200))
    tb = len(call(a.url, "/tokenize", {"content": block})["tokens"])
    prompt = (block + " ") * max(1, target // tb)
    n_tok = len(call(a.url, "/tokenize", {"content": prompt}, timeout=300)["tokens"])
    print(f"n_ctx={n_ctx}  prompt={n_tok} token ({n_tok / n_ctx:.1%})", flush=True)
    if n_tok + 16 > n_ctx:
        sys.exit("Prompt troppo lungo: riduci --fill")

    t0 = time.time()
    r = call(a.url, "/completion",
             {"prompt": prompt, "n_predict": 16, "cache_prompt": False}, timeout=a.timeout)
    t = r.get("timings", {})
    print(f"OK in {time.time() - t0:.0f} s | prefill {t.get('prompt_n')} tok "
          f"@ {t.get('prompt_per_second', 0):.1f} t/s | generazione "
          f"@ {t.get('predicted_per_second', 0):.2f} t/s")

if __name__ == "__main__":
    main()
