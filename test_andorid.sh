cargo run -p deepseek-ocr-android --bin android-cli --release -- \
  --model-kind paddle-ocr-vl \
  --config-path ~/Library/Caches/deepseek-ocr/models/paddleocr-vl/config.json \
  --tokenizer-path ~/Library/Caches/deepseek-ocr/models/paddleocr-vl/tokenizer.json \
  --weights-path ~/Library/Caches/deepseek-ocr/models/paddleocr-vl/model.safetensors \
  --prompt "Table Recognition: <image>" --image baselines/table_0.png