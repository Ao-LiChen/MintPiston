#!/bin/bash
# 启动HDOCK decoy生成 - 全量运行

cd /home/chenaoli/piston/MintPiston

echo "=========================================="
echo "Starting HDOCK Decoy Generation - FULL RUN"
echo "=========================================="
echo ""
echo "Total PPIs: 9,265"
echo "Parallel workers: 8"
echo "Estimated time: ~10-15 hours"
echo ""
echo "Log file: pipeline_out/logs/step1_full_run.log"
echo ""

nohup python step1_generate_decoy_pdbs.py > pipeline_out/logs/step1_full_run.log 2>&1 &

PID=$!
echo "Started. PID: $PID"
echo ""
echo "Monitor progress:"
echo "  tail -f pipeline_out/logs/step1_full_run.log"
echo ""
echo "Check status:"
echo "  ps aux | grep step1_generate_decoy_pdbs"
echo ""
echo "Check generated decoys:"
echo "  find pipeline_out/data_preparation/00-raw_pdbs -name '*d1.pdb' | wc -l"
