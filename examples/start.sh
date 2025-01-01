#!/bin/bash

#HF_ENDPOINT=https://hf-mirror.com

# Loop through prompt-len values from 2000 to 100000 with an increment of 2000
for prompt_len in {2000..100000..2000}
do
  # Run the Python script with the current prompt-len value
  python offline_profile.py --model /home/lvbo/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B/snapshots/91d2aff3f957f99e4c74c962f2f408dcc88a18d8/ --enforce-eager --max-num-batched-tokens 131072 --json datenlord_profile/qwen2_0.5b.json --prompt-len $prompt_len
done
