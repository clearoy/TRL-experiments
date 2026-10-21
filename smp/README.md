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
│   ├── prepare_data.py      data/clean/ -> data/split/group_1..5/{train,test}.parquet
│   │                        (split by user, top 20% within each group = YES)
│   └── describe_videos.py   data/clean/videos -> data/clean/video_content.jsonl
│                            (Gemini describes each video; resumable)
├── legacy/                the earlier random-split experiment (1,500 posts, top 10%).
│                          Kept for reference; it expects data/train.parquet with a
│                          `label` column, which the current pipeline no longer writes.
└── data/                  (gitignored except data/split_meta.json)
    ├── raw/                 Hugging Face download + videos (about 79 GB)
    ├── clean/               metadata.parquet, videos/, clean_meta.json, video_content.jsonl
    ├── split/               group_1 ... group_5, each with train.parquet and test.parquet
    └── split_meta.json      how the split was made (tracked)
```

## Run

```bash
python smp/pipeline/extract_clean.py
python smp/pipeline/prepare_data.py
python smp/pipeline/describe_videos.py --dry-run
```

Use the library's interpreter (`../think-reason-learn/.venv/bin/python`); see the
top-level README. API keys are read from the library repo's `.env`.
