#!/bin/bash
cd ./lyt/czsl/EVA3/code
while ! (grep -q 'done!' logs/ut_train.log 2>/dev/null && grep -q 'done!' logs/mit_train.log 2>/dev/null); do
  sleep 60
done
{
  echo "===== summary generated at $(date) ====="
  for name in ut mit; do
    echo ""
    echo "########## $name ##########"
    echo "--- all val/test result lines (last line = final test closed-world) ---"
    grep -nE 'best_seen' logs/${name}_train.log
    echo "--- max AUC value seen ---"
    grep -oE 'AUC  [0-9.]+' logs/${name}_train.log | awk '{print $2}' | sort -g | tail -1
  done
} > logs/eva_summary.txt 2>&1
