#!/bin/bash

# 주피터 노트북 파일명을 인자로 받음
NOTEBOOK_FILE="$1"

# 파일명이 주어지지 않으면 에러 메시지를 출력하고 종료
if [ -z "$NOTEBOOK_FILE" ]; then
    echo "Usage: $0 <notebook-file>"
    exit 1
fi

# 주어진 파일이 .ipynb 확장자를 가지고 있는지 확인
if [[ "$NOTEBOOK_FILE" != *.ipynb ]]; then
    echo "Error: The input file must have a .ipynb extension"
    exit 1
fi

# jupyter nbconvert 명령어를 사용하여 노트북 파일을 Python 스크립트로 변환
python -m jupyter nbconvert --to script "$NOTEBOOK_FILE"

# 변환 결과 확인
if [ $? -eq 0 ]; then
    echo "Conversion successful: ${NOTEBOOK_FILE%.ipynb}.py"
else
    echo "Conversion failed"
    exit 1
fi