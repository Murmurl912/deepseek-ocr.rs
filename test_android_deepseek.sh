cargo run -p deepseek-ocr-android --bin android-cli --release -- \
  --model-kind deepseek \
  --config-path ~/Library/Caches/deepseek-ocr/models/deepseek-ocr/config.json \
  --tokenizer-path ~/Library/Caches/deepseek-ocr/models/deepseek-ocr/tokenizer.json \
  --weights-path ~/Library/Caches/deepseek-ocr/models/deepseek-ocr/model.safetensors \
  --prompt "<image>\n<|grounding|>Convert this receipt to markdown." --image baselines/table_0.png