$env:KMP_DUPLICATE_LIB_OK = "TRUE"

python -B .\run_pipeline.py `
  --audio "D:\Desktop\音频识别项目\多人对话.wav" `
  --reference-file "D:\Desktop\音频识别项目\多人对话文本.txt" `
  --output .\outputs\medium_no_fix `
  --engine faster-whisper `
  --model "D:\Desktop\音频识别项目\asr_meeting_assistant\models\faster-whisper-medium" `
  --language zh `
  --speakers 5 `
  --diarizer cluster
