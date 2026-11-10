# Test to get stats

```
curl -s http://localhost:8080/v1/chat/completions   -H 'Content-Type: application/json'   -d '{"messages":[{"role":"user","content":"Spiegami la relatività ristretta in 600 parole"}],"max_tokens":1000}'   | jq .timings
```
