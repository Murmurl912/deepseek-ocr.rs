cargo run -p deepseek-ocr-cli --release --features metal -- \
  --prompt "<image>\n<|grounding|>Convert this receipt to markdown." \
  --image baselines/sample/images/test.png \
  --device metal --max-new-tokens 512