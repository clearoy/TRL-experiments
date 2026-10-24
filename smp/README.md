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
│   └── prepare_data.py      data/clean/ -> data/split/group_1..5/{train,test}.parquet
│                            (split by user, top 20% within each group = YES;
│                            not yet re-run on the merged data)
├── research/
│   └── video_extraction/  the prompt study that chose design V4 (prompts.py, run.py)
├── legacy/                the earlier random-split experiment (1,500 posts, top 10%).
│                          Kept for reference; it expects data/train.parquet with a
│                          `label` column, which the current pipeline no longer writes.
└── data/                  (gitignored except data/split_meta.json)
    ├── raw/                 Hugging Face download + videos (about 79 GB)
    ├── clean/               merged_data.parquet (main file), metadata.parquet, videos/,
    │                        video_content.jsonl, clean_meta.json; see its README
    ├── research/            results of the video-extraction prompt study
    ├── split/               (not built yet) group_1 ... group_5, train/test parquet
    └── split_meta.json      (not built yet) how the split was made; tracked
```

## Run

```bash
python smp/pipeline/extract_clean.py
python smp/pipeline/describe_videos.py
python smp/pipeline/merge_data.py
```

Use the library's interpreter (`../think-reason-learn/.venv/bin/python`); see the
top-level README. API keys are read from the library repo's `.env`.
