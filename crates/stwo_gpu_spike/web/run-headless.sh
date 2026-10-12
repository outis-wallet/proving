#!/bin/bash
# Runs the demo in headless Chrome on SwiftShader (software WebGPU) and prints its reports.
#   ./run-headless.sh ['threads=8&sizes=64,128,256,512&reps=3']
#   PAGE=probe.html ./run-headless.sh          (the browser's capabilities only)
#   SAMPLE_RSS=1 ./run-headless.sh ...         (also the peak RSS of Chrome's processes, by type)
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
PAGE=${PAGE:-}

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
  "http://127.0.0.1:$PORT/$PAGE?$query" > "$profile.log" 2>&1 &
chrome=$!
peaks=$(mktemp)
for _ in $(seq 1 $((LIMIT * 2))); do
  sleep 0.5
  if [ -n "$SAMPLE_RSS" ]; then
    # RSS in KiB of each of Chrome's processes, keyed by --type (browser when none).
    ps -o rss=,args= -g "$chrome" | awk '{ t = "browser"; for (i = 2; i <= NF; i++) if ($i ~ /^--type=/) { t = substr($i, 8); if (t == "utility") for (j = 2; j <= NF; j++) if ($j ~ /^--utility-sub-type=/) t = t ":" substr($j, 20) } sum[t] += $1 } END { for (t in sum) print t, sum[t] }' >> "$peaks"
  fi
  tail -n +$((before + 1)) results.log 2>/dev/null | grep -Eq '"event": *"(end|probe)"' && break
done
kill -TERM -- -$chrome 2>/dev/null; sleep 1; kill -KILL -- -$chrome 2>/dev/null
[ -n "$server_pid" ] && kill "$server_pid" 2>/dev/null
tail -n +$((before + 1)) results.log
if [ -n "$SAMPLE_RSS" ]; then
  echo "peak RSS by process type (MiB):"
  awk '{ if ($2 > peak[$1]) peak[$1] = $2 } END { for (t in peak) printf "  %s %d\n", t, peak[t] / 1024 }' "$peaks"
fi
rm -rf "$profile" "$profile.log" "$peaks"
