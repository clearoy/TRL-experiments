# SMP-Video benchmark for PolicyInduction

Popularity of short videos from the SMP Challenge (`smpchallenge/SMP-Video` on
Hugging Face, gated). Dataset details and the current numbers are on the
dashboard page `dashboard/smp.html`.

## Layout

```
smp/
├── pipeline/              current data pipeline, run in this order
│   ├── extract_clean.py     raw train files -> data/clean/ (US + English posts,
│   │                        grouped by account, user counts dropped, videos copied)
│   ├── describe_videos.py   data/clean/videos -> data/clean/video_content.jsonl
│   │                        (Gemini text record of each video, design V4; resumable)
│   ├── merge_data.py        descriptions + metadata -> data/clean/merged_data.parquet
│   │                        (drops posts without a description)
│   ├── make_ready.py        merged_data -> data/clean/ready_data.parquet (model-ready:
│   │                        ids, label, group, 15 content features, model_text)
│   └── prepare_data.py      ready_data -> data/split/{train,test}.parquet (by user,
│                            80/20; YES = top 20% within each of 5 user groups)
├── research/
│   └── video_extraction/  the prompt study that chose design V4 (prompts.py, run.py)
├── legacy/                the earlier random-split experiment (1,500 posts, top 10%).
│                          Kept for reference; it expects data/train.parquet with a
│                          `label` column, which the current pipeline no longer writes.
└── data/                  (gitignored except data/split_meta.json)
    ├── raw/                 Hugging Face download + videos (about 79 GB)
    ├── clean/               ready_data.parquet (model input), merged_data.parquet, metadata.parquet, videos/,
    │                        video_content.jsonl, clean_meta.json; see its README
    ├── research/            results of the video-extraction prompt study
    ├── split/               train.parquet (1,731 posts) and test.parquet (436 posts)
    └── split_meta.json      how the split was made, per-group cutoffs and counts; tracked
```

## Run

```bash
python smp/pipeline/extract_clean.py
python smp/pipeline/describe_videos.py
python smp/pipeline/merge_data.py
python smp/pipeline/make_ready.py
python smp/pipeline/prepare_data.py
```

Use the library's interpreter (`../think-reason-learn/.venv/bin/python`); see the
top-level README. API keys are read from the library repo's `.env`.
