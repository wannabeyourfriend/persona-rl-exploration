#!/usr/bin/env bash
# Validate the local judge (llama.cpp OR vLLM): /v1/models, plain chat, and the
# nested AgentEval 7-dim json_schema (exactly what agents/sotopia uses).
PORT="${PORT:-8000}"
B="http://127.0.0.1:$PORT/v1"
echo "=== /v1/models ==="; curl -sS -m 30 "$B/models" | head -c 300; echo
echo "=== plain chat ==="; curl -sS -m 90 "$B/chat/completions" -H "Content-Type: application/json" \
  -d '{"model":"qwen-judge","messages":[{"role":"user","content":"reply with exactly: ok"}],"max_tokens":16}' | head -c 400; echo
echo "=== json_schema AgentEval (7 dims) ==="
curl -sS -m 180 "$B/chat/completions" -H "Content-Type: application/json" -d '{
 "model":"qwen-judge",
 "messages":[{"role":"system","content":"You are a Sotopia social evaluator. Score each of 7 dimensions with short reasoning and an integer score."},
             {"role":"user","content":"Dialogue: Alice: Hi, can I borrow $10? Bob: Sure, here you go. Evaluate Bob on the 7 dimensions."}],
 "response_format":{"type":"json_schema","json_schema":{"name":"AgentEval","strict":false,"schema":{
   "$defs":{"DimScore":{"type":"object","properties":{"reasoning":{"type":"string"},"score":{"type":"integer"}},"required":["reasoning","score"]}},
   "type":"object",
   "properties":{"believability":{"$ref":"#/$defs/DimScore"},"relationship":{"$ref":"#/$defs/DimScore"},"knowledge":{"$ref":"#/$defs/DimScore"},"secret":{"$ref":"#/$defs/DimScore"},"social_rules":{"$ref":"#/$defs/DimScore"},"financial_and_material_benefits":{"$ref":"#/$defs/DimScore"},"goal":{"$ref":"#/$defs/DimScore"}},
   "required":["believability","relationship","knowledge","secret","social_rules","financial_and_material_benefits","goal"]
 }}}}' | head -c 1400; echo
