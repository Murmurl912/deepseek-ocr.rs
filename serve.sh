cargo run -p deepseek-ocr-server --features metal --release -- \
  --host 0.0.0.0 --port 8000 \
  --device metal --dtype f16 --max-new-tokens 512