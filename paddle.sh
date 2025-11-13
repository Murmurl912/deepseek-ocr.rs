cargo run -p deepseek-ocr-cli --release --features metal -- \
  --model paddleocr-vl \
  --prompt "<image> OCR Recognition" \
  --image baselines/table_0.png \
  --device metal \
  --dtype f16 \
  --max-new-tokens 8196