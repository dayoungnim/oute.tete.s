#!/bin/bash

# CUDA 빌드 관련 환경변수 설정
export CMAKE_ARGS="-DGGML_CUDA=on -DCUDA_PATH=/usr/local/cuda-12.4 -DCUDAToolkit_ROOT=/usr/local/cuda-12.4 -DCUDAToolkit_INCLUDE_DIR=/usr/local/cuda-12/include -DCUDAToolkit_LIBRARY_DIR=/usr/local/cuda-12.4/lib64"
export CUDACXX=/usr/local/cuda-12.4/bin/nvcc

# 현재 날짜 및 시간
NOW=$(date "+%Y-%m-%d_%H-%M-%S")

# 가장 여유 메모리 많은 GPU 찾기
BEST_GPU=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
           | awk '{print NR-1, $1}' | sort -k2 -nr | head -n1 | cut -d' ' -f1)

echo "🚀 Using GPU $BEST_GPU"
export CUDA_VISIBLE_DEVICES=$BEST_GPU

# 로그 파일 경로
LOGFILE="log_$NOW.txt"

# Python 스크립트 실행 및 로그 저장
# python infer.py | tee "$LOGFILE"
script -q -c "python infer.py" "$LOGFILE"