#!/bin/bash
curl -s  -H "Authorization: Bearer $LUCAS_KEY" -F "syntax=json" -F "content=<-" https://dpaste.com/api/ <results.json -F "title=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader)"
