# app/api/v1/translation.py (수정됨)

from fastapi import APIRouter, HTTPException, Depends
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from typing import Dict, Set, List # List 추가
import torch

from app.schema.translation import TranslationRequest, TranslationResponse, LANG_CODE_TO_NAME
from app.core.models import get_translation_resources

# 🌟 장문 처리를 위한 상수 및 헬퍼 함수 추가 🌟
MAX_TOKENS = 1024
CHUNK_MAX_LENGTH = 900 # 모델의 최대 길이보다 작게 설정하여 안전 마진 확보

def split_text_into_chunks(text: str, tokenizer: M2M100Tokenizer) -> List[str]:
    """
    텍스트를 토큰 길이 제한(CHUNK_MAX_LENGTH)에 맞춰 문장 단위로 분할합니다.
    (M2M100 토크나이저의 길이 체크를 기반으로 합니다.)
    """
    # 1. 텍스트를 문장 경계로 1차 분할 (간단한 구현)
    # 마침표, 물음표, 느낌표, 개행 문자를 기준으로 분할 후, 토큰 길이에 맞춰 다시 묶습니다.
    # Note: 이 분할 방식은 다국어 환경에서 완벽하지 않지만, 기본 동작 방식을 보여줍니다.
    sentences_raw = text.replace('\n', '[SNT_SEP]\n').replace('. ', '. [SNT_SEP] ').replace('? ', '? [SNT_SEP] ').replace('! ', '! [SNT_SEP] ')
    sentences = [s.strip() for s in sentences_raw.split('[SNT_SEP]') if s.strip()]

    # 2. 토큰 길이 제한에 맞춰 문장들을 청크로 다시 묶습니다.
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        
        # 현재 청크와 새 문장을 합쳤을 때의 텍스트
        test_text = (current_chunk + " " + sentence).strip()
        
        # 토큰 길이 체크 (특수 토큰 포함 길이)
        token_ids = tokenizer.encode(test_text, add_special_tokens=True)
        encoded_len = len(token_ids)

        if encoded_len <= CHUNK_MAX_LENGTH:
            # 길이를 초과하지 않으면 현재 청크에 추가
            current_chunk = test_text
        else:
            # 길이를 초과하면 현재 청크를 저장하고, 새 문장으로 새 청크 시작
            if current_chunk:
                chunks.append(current_chunk)
            
            # 새 문장 자체가 너무 길 경우 (MAX_TOKENS 초과), 그대로 한 청크로 만듦
            sentence_token_ids = tokenizer.encode(sentence, add_special_tokens=True)
            if len(sentence_token_ids) > MAX_TOKENS:
                print(f"경고: 단일 문장이 최대 토큰 길이({MAX_TOKENS})를 초과합니다. ({len(sentence_token_ids)}). 모델에서 잘림.")
            
            current_chunk = sentence

    # 마지막 남은 청크 저장
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks


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


    text = request.text
    src_lang = request.src_lang
    tgt_lang = request.tgt_lang

    try:
        # 1. 소스 언어 설정
        tokenizer.src_lang = src_lang
        
        # 🌟 2. 장문 번역을 위한 분할 로직 사용 🌟
        text_chunks = split_text_into_chunks(text, tokenizer)
        translated_chunks = []
        
        for chunk in text_chunks:
            
            # 3. 텍스트 인코딩 및 장치 할당
            encoded_input = tokenizer(
                chunk, 
                return_tensors="pt",
                padding="max_length", 
                truncation=True, # 안전하게 최대 토큰 수를 초과하지 않도록 설정
                max_length=MAX_TOKENS # 모델의 최대 입력 길이 (1024)
            )
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
            
            # 4. 번역 생성
            with torch.no_grad():
                generated_tokens = model.generate(
                    **encoded_input,
                    # get_lang_id()는 여전히 유효한 메서드이므로 그대로 사용합니다.
                    forced_bos_token_id=tokenizer.get_lang_id(tgt_lang),
                    max_length=MAX_TOKENS + 10 # 출력 길이도 충분히 확보 (예: 1034)
                )
            
            # 5. 토큰 디코딩
            translated_chunk = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            translated_chunks.append(translated_chunk)

        
        # 6. 분할된 번역 결과를 하나의 문자열로 결합
        translated_text = " ".join(translated_chunks) # 공백으로 결합
        
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
