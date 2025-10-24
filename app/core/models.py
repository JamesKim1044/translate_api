from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch
import os
# 🌟 수정된 부분: schemas 파일에서 코드 목록 임포트
from app.schema.translation import SUPPORTED_LANG_CODES

os.environ["TQDM_DISABLE"] = "1"

MODEL_NAME = "facebook/m2m100_418M"
global_resources = {} 

def load_translation_models():
    print("모델 및 토크나이저 로드 시작...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        global_resources["device"] = device
        
        model = M2M100ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
        tokenizer = M2M100Tokenizer.from_pretrained(MODEL_NAME)
        
        global_resources["model"] = model
        global_resources["tokenizer"] = tokenizer
        
        # M2M100 실제 지원 언어와 사용자 제공 목록의 교집합 사용
        m2m_supported_langs = set(tokenizer.lang_code_to_id.keys())
        global_resources["supported_langs"] = m2m_supported_langs.intersection(set(SUPPORTED_LANG_CODES))
        
        print(f"모델 로드 완료. 사용 장치: {device}")
        print(f"실제 로드된 지원 언어 코드 개수: {len(global_resources['supported_langs'])}")
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")

def get_translation_resources():
    return global_resources