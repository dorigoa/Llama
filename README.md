## Test to get stats

```bash
curl -s http://localhost:8080/v1/chat/completions   -H 'Content-Type: application/json'   -d '{"messages":[{"role":"user","content":"Spiegami la relatività ristretta in 600 parole"}],"max_tokens":1000}'   | jq .timings
```
## Stress test
```bash
seq 1 500 | xargs -P 4 -I{} bash -c 'n=$((RANDOM%800+100)); p=$(shuf -n $((RANDOM%40+5)) /usr/share/dict/words 2>/dev/null | tr "\n" " " || echo "spiega la meccanica statistica"); curl -s --max-time 180 -o /dev/null -w "[%{http_code}] %{time_total}s\n" http://192.168.1.200:8080/v1/chat/completions -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Scrivi un testo dettagliato su: $p\"}],\"max_tokens\":$n}"'
```
