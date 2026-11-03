# Test to get stats

```
url -s http://localhost:9931/v1/chat/completions   -H 'Content-Type: application/json'   -d '{"messages":[{"role":"user","content":"Spiegami la relatività ristretta in 600 parole"}],"max_tokens":1000}'   | jq .timings
```
