#!/bin/bash
# Runs the demo in headless Chrome on SwiftShader (software WebGPU) and prints its reports.
#   ./run-headless.sh ['threads=8&sizes=64,128,256,512&reps=3']
# Serves this directory with COOP/COEP on $PORT (default 8795) with agent-tools/server.py unless
# something already listens there, and stops what it started. Needs ./build.sh first.
# The flags that work with Chrome 154 here: --enable-unsafe-webgpu --use-webgpu-adapter=swiftshader
# --enable-features=Vulkan (and not --disable-gpu).
cd "$(dirname "$0")"
query=${1:-}
PORT=${PORT:-8795}
SERVER=${SERVER:-$(cd ../../../.. && pwd)/agent-tools/server.py}
CHROME=${CHROME:-google-chrome}
LIMIT=${LIMIT:-900}

server_pid=
if ! ss -ltn "sport = :$PORT" | grep -q LISTEN; then
  python3 "$SERVER" "$PORT" > /dev/null 2>&1 &
  server_pid=$!
  sleep 1
fi

before=$(wc -l < results.log 2>/dev/null || echo 0)
profile=$(mktemp -d)
setsid "$CHROME" --headless=new --no-first-run --no-default-browser-check --user-data-dir="$profile" \
  --enable-unsafe-webgpu --use-webgpu-adapter=swiftshader --enable-features=Vulkan \
  "http://127.0.0.1:$PORT/?$query" > "$profile.log" 2>&1 &
chrome=$!
for _ in $(seq 1 "$LIMIT"); do
  sleep 1
  tail -n +$((before + 1)) results.log 2>/dev/null | grep -q '"event": *"end"' && break
done
kill -TERM -- -$chrome 2>/dev/null; sleep 1; kill -KILL -- -$chrome 2>/dev/null
[ -n "$server_pid" ] && kill "$server_pid" 2>/dev/null
tail -n +$((before + 1)) results.log
rm -rf "$profile" "$profile.log"
