#!/bin/bash

# ASTRA 데이터셋 다운로드 스크립트
# README.md에 명시된 경로에 맞춰 필요한 데이터셋들을 다운로드합니다.

set -e  # 에러 발생 시 스크립트 중단

echo "=== ASTRA 데이터셋 다운로드 시작 ==="

# 기본 디렉토리 설정
BASE_DIR="/scratch2/pljh0906/ASTRA"
DATASETS_DIR="${BASE_DIR}/datasets"

cd $BASE_DIR

echo "현재 디렉토리: $(pwd)"

# 1. RealToxicityPrompts 데이터셋 다운로드
echo "=== 1. RealToxicityPrompts 다운로드 중... ==="
if [ ! -f "${DATASETS_DIR}/harmful_corpus/realtoxicityprompts.jsonl" ]; then
    mkdir -p ${DATASETS_DIR}/harmful_corpus
    cd ${DATASETS_DIR}/harmful_corpus
    
    # Hugging Face Hub에서 다운로드
    echo "RealToxicityPrompts를 Hugging Face에서 다운로드합니다..."
    wget https://huggingface.co/datasets/allenai/real-toxicity-prompts/resolve/main/prompts.jsonl -O realtoxicityprompts.jsonl
    
    echo "RealToxicityPrompts 다운로드 완료"
    
    # 데이터셋 분할 실행
    echo "데이터셋을 validation/test 세트로 분할합니다..."
    if [ -f "split_toxicity_set.py" ]; then
        python split_toxicity_set.py
        echo "Toxicity 데이터셋 분할 완료"
    else
        echo "경고: split_toxicity_set.py 파일이 없습니다."
    fi
else
    echo "RealToxicityPrompts가 이미 존재합니다. 건너뛰기..."
fi

# 2. AdvBench 데이터셋 다운로드  
echo "=== 2. AdvBench 데이터셋 다운로드 중... ==="
if [ ! -f "${DATASETS_DIR}/harmful_corpus/train.csv" ] || [ ! -f "${DATASETS_DIR}/harmful_corpus/eval.csv" ]; then
    cd ${DATASETS_DIR}/harmful_corpus
    
    # AdvBench repository에서 다운로드
    echo "AdvBench 데이터를 다운로드합니다..."
    
    # train.csv (steering vectors 구성용)
    wget https://raw.githubusercontent.com/RylanSchaeffer/AstraFellowship-When-Do-VLM-Image-Jailbreaks-Transfer/main/prompts_and_targets/AdvBench/train.csv -O train.csv
    
    # eval.csv (평가용)
    wget https://raw.githubusercontent.com/RylanSchaeffer/AstraFellowship-When-Do-VLM-Image-Jailbreaks-Transfer/main/prompts_and_targets/AdvBench/eval.csv -O eval.csv
    
    echo "AdvBench 데이터셋 다운로드 완료"
    
    # 데이터셋 분할 실행
    echo "AdvBench eval 데이터셋을 validation/test 세트로 분할합니다..."
    if [ -f "split_jb_set.py" ]; then
        python split_jb_set.py
        echo "Jailbreak 데이터셋 분할 완료"
    else
        echo "경고: split_jb_set.py 파일이 없습니다."
    fi
else
    echo "AdvBench 데이터셋이 이미 존재합니다. 건너뛰기..."
fi

# 3. MM-Vet 이미지 다운로드
echo "=== 3. MM-Vet 이미지 다운로드 중... ==="
if [ ! -d "${DATASETS_DIR}/mm-vet/images" ]; then
    mkdir -p ${DATASETS_DIR}/mm-vet
    cd ${DATASETS_DIR}/mm-vet
    
    echo "MM-Vet 이미지를 다운로드합니다..."
    
    # MM-Vet 이미지 다운로드 (GitHub releases에서)
    wget https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip
    unzip mm-vet.zip
    mv mm-vet/images ./
    rm -rf mm-vet mm-vet.zip
    
    echo "MM-Vet 이미지 다운로드 완료"
else
    echo "MM-Vet 이미지가 이미 존재합니다. 건너뛰기..."
fi

# 4. MMBench 이미지 다운로드
echo "=== 4. MMBench 이미지 다운로드 중... ==="
if [ ! -d "${DATASETS_DIR}/MMBench/images" ]; then
    mkdir -p ${DATASETS_DIR}/MMBench
    cd ${DATASETS_DIR}/MMBench
    
    echo "MMBench 데이터를 다운로드합니다..."
    
    # MMBench 데이터 다운로드 (Hugging Face에서)
    # TSV 파일들
    wget https://huggingface.co/datasets/open-compass/MMBench/resolve/main/mmbench_dev_20230712.tsv
    wget https://huggingface.co/datasets/open-compass/MMBench/resolve/main/mmbench_test_20230712.tsv
    
    # 이미지 압축 파일 다운로드
    echo "MMBench 이미지를 다운로드합니다 (시간이 걸릴 수 있습니다)..."
    wget https://huggingface.co/datasets/open-compass/MMBench/resolve/main/mmbench_dev_images.zip
    
    # 이미지 압축 해제
    unzip mmbench_dev_images.zip
    mv mmbench_dev_images images
    rm mmbench_dev_images.zip
    
    echo "MMBench 데이터 다운로드 완료"
    
    # 데이터셋 분할 실행
    echo "MMBench 데이터셋을 validation/test 세트로 분할합니다..."
    if [ -f "split_mmbench.py" ]; then
        python split_mmbench.py
        echo "MMBench 데이터셋 분할 완료"
    else
        echo "경고: split_mmbench.py 파일이 없습니다."
    fi
else
    echo "MMBench 이미지가 이미 존재합니다. 건너뛰기..."
fi

# 5. 추가 데이터셋: manual_harmful_instructions.csv (이미 있는지 확인)
echo "=== 5. manual_harmful_instructions.csv 확인 중... ==="
if [ ! -f "${DATASETS_DIR}/harmful_corpus/manual_harmful_instructions.csv" ]; then
    echo "경고: manual_harmful_instructions.csv가 없습니다."
    echo "다음 링크에서 수동으로 다운로드해주세요:"
    echo "https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/tree/main/harmful_corpus"
    echo "파일을 ${DATASETS_DIR}/harmful_corpus/ 에 저장해주세요."
else
    echo "manual_harmful_instructions.csv가 이미 존재합니다."
fi

# 다운로드 완료 상태 출력
echo ""
echo "=== 다운로드 완료 상태 ==="
cd $BASE_DIR

echo "📁 데이터셋 구조 확인:"
echo "harmful_corpus:"
ls -la ${DATASETS_DIR}/harmful_corpus/ 2>/dev/null || echo "  ❌ 디렉토리가 없습니다"

echo ""
echo "mm-vet:"
ls -la ${DATASETS_DIR}/mm-vet/ 2>/dev/null || echo "  ❌ 디렉토리가 없습니다"

echo ""  
echo "MMBench:"
ls -la ${DATASETS_DIR}/MMBench/ 2>/dev/null || echo "  ❌ 디렉토리가 없습니다"

echo ""
echo "=== 모든 데이터셋 다운로드 완료 ==="
echo "이제 다음 명령으로 실험을 시작할 수 있습니다:"
echo ""
echo "예제 (Qwen2-VL Jailbreak 설정):"
echo "  CUDA_VISIBLE_DEVICES=0 python ./extract_attr/extract_qwen_jb_attr.py"
echo "  CUDA_VISIBLE_DEVICES=0 python ./extract_act/extracting_activations_qwen_jb.py"
echo ""
echo "자세한 내용은 README.md를 참조하세요."
