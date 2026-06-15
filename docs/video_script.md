# 10 Minute Video Script

## 0:00-1:00 Problem

Hello everyone. Our project is a meeting audio assistant for multi-speaker recordings. Raw recordings are hard to search, summarize, and review. This becomes more difficult when people speak quickly, interrupt each other, or when the audio contains background noise.

## 1:00-2:00 Project Goal

Our goal is to build a lightweight end-to-end pipeline:

```text
audio input -> ASR or transcript parsing -> speaker diarization -> text cleanup -> summary -> metrics -> report
```

The system generates a transcript with timestamps, anonymous speaker labels, a summary, keywords, action items, CER/WER metrics, and a markdown report.

## 2:00-4:00 Technical Pipeline

The project has two transcription modes. In demo mode, it uses a human transcript so the whole pipeline can run without downloading large ASR models. In real ASR mode, it uses faster-whisper to transcribe the audio automatically.

For speaker diarization, we do not identify real names. Instead, we assign anonymous labels such as SPEAKER_00 and SPEAKER_01. The main method is MFCC-like audio feature extraction followed by KMeans clustering. This connects the project to course topics such as feature extraction, standardization, and unsupervised learning.

The code is organized into modules: schemas, audio input, transcription, diarization, text processing, output writing, and the main pipeline controller.

## 4:00-6:00 Demo

We run the command line pipeline or the Streamlit GUI. The input is `多人对话.wav`, a real multi-speaker audio file, and `多人对话文本.txt`, the human reference transcript.

The output folder contains:

```text
raw_transcript.md
transcript.md
summary.md
report.md
segments.json
metrics.json
```

Show the transcript with timestamps and anonymous speaker labels. Then show the summary, keywords, and the metrics file.

## 6:00-8:00 Results and Analysis

For real ASR, we tested faster-whisper with different model sizes. The base model gives better Chinese character error rate than the tiny model, although it takes longer to run.

One representative result is:

```text
model: faster-whisper base
runtime: about 88 seconds
raw CER: 0.2863
raw ASR segments: 287
display segments after cleanup: 32
```

For Chinese, we focus more on CER than WER because Chinese text does not naturally separate words with spaces. We also compare raw ASR output and display output so that text cleanup does not hide the real ASR accuracy.

## 8:00-9:00 Limitations

This is a course-level prototype, not a commercial ASR or diarization system. The clustering method can group acoustically similar segments, but it cannot identify real names. If we need real identity recognition, we need reference voice samples or manual labels.

Overlapping speech, very short segments, background noise, and similar voices are still difficult. A stronger version could use pyannote.audio, SpeechBrain speaker embeddings, voice activity detection, and denoising.

## 9:00-10:00 Conclusion

This project demonstrates a complete ASR-centered workflow for multi-speaker audio. It is useful for meeting notes, classroom review, and searchable audio archives. More importantly, it shows practical machine learning trade-offs: model size versus accuracy, raw output versus post-processed output, and simple clustering versus stronger speaker diarization models.
