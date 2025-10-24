# app/api/v1/translation.py (수정됨)

from fastapi import APIRouter, HTTPException, Depends
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from typing import Dict, Set
import torch

from app.schema.translation import TranslationRequest, TranslationResponse, LANG_CODE_TO_NAME
from app.core.models import get_translation_resources


# 1. 라우터 Description에 들어갈 상세 설명 문자열 생성
LANG_PAIRS = [f"{name} ({code})" for code, name in LANG_CODE_TO_NAME.items()]
ROUTER_DESCRIPTION = (
    "M2M100 모델을 사용하여 텍스트를 번역합니다.\n\n"
    "### 🌍 지원 언어 코드 (ISO 639-1 기반):\n\n"
    "  - " + "\n  - ".join(LANG_PAIRS)
)
router = APIRouter(
    prefix="/translate", 
    tags=["Translation"],
    
)
@router.post("", response_model=TranslationResponse, description=ROUTER_DESCRIPTION)
async def translate_text(
    request: TranslationRequest,
    resources: Dict = Depends(get_translation_resources)
):
    model: M2M100ForConditionalGeneration = resources.get("model")
    tokenizer: M2M100Tokenizer = resources.get("tokenizer")
    device: torch.device = resources.get("device")
    # 🌟 수정된 부분: 미리 로드된 지원 언어 목록 가져오기


    text = request.text
    src_lang = request.src_lang
    tgt_lang = request.tgt_lang



    try:
        # 1. 소스 언어 설정
        tokenizer.src_lang = src_lang
        
        # 2. 텍스트 인코딩 및 장치 할당
        encoded_input = tokenizer(text, return_tensors="pt")
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        
        # 3. 번역 생성
        with torch.no_grad():
            generated_tokens = model.generate(
                **encoded_input,
                # get_lang_id()는 여전히 유효한 메서드이므로 그대로 사용합니다.
                forced_bos_token_id=tokenizer.get_lang_id(tgt_lang) 
            )
        
        # 4. 토큰 디코딩
        translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        
        return TranslationResponse(
            original_text=text,
            translated_text=translated_text,
            src_lang=src_lang,
            tgt_lang=tgt_lang
        )

    except Exception as e:
        print(f"번역 중 오류 발생: {e}")
        # ... (오류 처리 로직 유지)
        raise HTTPException(status_code=500, detail=f"번역 처리 중 내부 서버 오류 발생: {e}")