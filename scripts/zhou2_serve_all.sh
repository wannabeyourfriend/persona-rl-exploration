#!/usr/bin/env bash
# zhou-2: 4 llama.cpp judge instances (2 GPUs each, all 8x3090) on ports 8001-8004,
# behind an nginx least-conn load balancer on :8000. Qwen3-32B for partner+judge.
# NOTE: no NVLink -> llama.cpp layer-splits each instance across its 2 GPUs
# (sequential per token; one GPU of each pair sits idle). Throughput ceiling is
# ~4 GPUs of compute. For real tensor-parallel throughput, prefer vLLM
# (scripts/zhou2_vllm_noroot_setup.sh) or run the judge on a zhou-1 A100.
set -uo pipefail
cd "$HOME"
SERVE="$HOME/zhou2_serve_judge.sh"

echo "[$(date)] stopping any existing llama-server"
pkill -f "llamaenv/bin/llama-server" 2>/dev/null || true
sleep 3

echo "[$(date)] launching 4 instances (2 GPUs each)"
i=0
for gpus in 0,1 2,3 4,5 6,7; do
  port=$((8001+i)); i=$((i+1))
  PORT=$port GPUS=$gpus CTX=65536 PAR=8 bash "$SERVE"
done

NGINX="$HOME/nginxenv/bin/nginx"
if [ ! -x "$NGINX" ]; then
  echo "[$(date)] installing nginx (micromamba)"
  "$HOME/.local/bin/micromamba" create -y -p "$HOME/nginxenv" \
    -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ nginx 2>&1 | tail -5
fi
mkdir -p "$HOME/nginx_tmp/logs"
cat > "$HOME/nginx.conf" <<EOF
worker_processes 2;
pid $HOME/nginx_tmp/nginx.pid;
error_log $HOME/nginx_tmp/logs/error.log;
events { worker_connections 4096; }
http {
  access_log off;
  client_body_temp_path $HOME/nginx_tmp/cbt;
  proxy_temp_path $HOME/nginx_tmp/pt;
  fastcgi_temp_path $HOME/nginx_tmp/ft;
  uwsgi_temp_path $HOME/nginx_tmp/ut;
  scgi_temp_path $HOME/nginx_tmp/st;
  upstream judges { least_conn; server 127.0.0.1:8001; server 127.0.0.1:8002; server 127.0.0.1:8003; server 127.0.0.1:8004; }
  server {
    listen 0.0.0.0:8000;
    client_max_body_size 64m;
    location / { proxy_pass http://judges; proxy_read_timeout 900s; proxy_send_timeout 900s; }
  }
}
EOF
"$NGINX" -s stop -c "$HOME/nginx.conf" -p "$HOME/" 2>/dev/null || true
sleep 1
if "$NGINX" -c "$HOME/nginx.conf" -p "$HOME/"; then echo "[$(date)] NGINX_LB_UP on :8000"; else echo "NGINX_FAILED"; fi
echo "SERVE_ALL_DONE"
