#!/bin/bash

source .venv/bin/activate

CUDA_VISIBLE_DEVICES=7 vllm serve /public/llm_models/Qwen/Qwen3-30B-A3B-Thinking-2507 \
    --served-model-name model \
    --trust-remote-code \
    --data-parallel-size 1 \
    --tensor-parallel-size 1 \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --max-model-len 32768 \
    --max-num-seqs 16 \
    --no-async-scheduling \
    --attention-backend TREE_ATTN \
    --speculative-config '{"method":"mtp", "num_speculative_tokens":14, "model":"/workspace-dyb/hw-ding/qwen3-30b-a3b-thinking-mtp-1", "speculative_num_children_per_level": 2, "speculative_num_level": 3}' \
    --port 8081