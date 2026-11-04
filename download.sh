#!/bin/bash
# Download the raw data for the big run into ~/tiny-llm/data/raw.
set -u
R=~/tiny-llm/data/raw
mkdir -p $R/codeparrot $R/instruct $R/eval
HF=https://huggingface.co/datasets
get() { # url dest: resume partial files, retry on failure
  for i in 1 2 3 4 5; do curl -sSfL -C - --retry 5 -o "$2" "$1" && return 0; sleep 10; done; echo "FAILED $1"; }
get $HF/ise-uiuc/Magicoder-OSS-Instruct-75K/resolve/main/data-oss_instruct-decontaminated.jsonl $R/instruct/oss-instruct.jsonl
get $HF/ise-uiuc/Magicoder-Evol-Instruct-110K/resolve/main/data-evol_instruct-decontaminated.jsonl $R/instruct/evol-instruct.jsonl
get https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz $R/eval/HumanEval.jsonl.gz
get https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl $R/eval/mbpp.jsonl
files=$(curl -s "https://huggingface.co/api/datasets/codeparrot/codeparrot-clean/tree/main" | python3 -c "import sys,json; print(' '.join(f['path'] for f in json.load(sys.stdin) if f['path'].endswith('.json.gz')))")
for f in $files; do get $HF/codeparrot/codeparrot-clean/resolve/main/$f $R/codeparrot/$f; echo "got $f"; done
echo ALL DONE
