# jameskim1044/translate_api/translate_api-3ccc27078703c49f092ace6650060d796848d85d/app/routers/translate.py

from fastapi import APIRouter, HTTPException, Depends
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from typing import Dict, Set, List, Any
import torch
import threading
import uuid
from langdetect import detect, LangDetectException
import re # 🌟 추가: 정규 표현식 모듈 임포트

from app.schema.translation import TranslationRequest, TranslationResponse, LANG_CODE_TO_NAME, AutoTranslationRequest
from app.core.models import get_translation_resources


# 🌟 전역 변수: 번역 작업 상태 및 결과 저장 (스레드 간 공유)
# 구조: {job_id: {"status": str, "result": TranslationResponse object or exception details}}
translation_jobs: Dict[str, Dict[str, Any]] = {}

# 🌟 장문 처리를 위한 상수 (500 토큰으로 유지) 🌟
MAX_TOKENS = 500
CHUNK_MAX_LENGTH = 480 # 모델의 최대 길이(500)보다 작게 설정하여 안전 마진 확보


def split_text_into_chunks(text: str, tokenizer: M2M100Tokenizer) -> List[str]:
    """
    텍스트를 토큰 길이 제한(CHUNK_MAX_LENGTH)에 맞춰 문장 단위로 분할합니다.
    🌟 수정됨: 정규 표현식을 사용하여 문장 부호 뒤에 공백이 없는 경우를 확실히 처리하도록 로직 개선.
    """
    
    # 1. 정규 표현식을 사용하여 문장 부호(., ?, !) 뒤에서 텍스트를 분할합니다.
    # (?<=[.?!])는 lookbehind assertion으로, 분할 시 구분자(문장 부호)를 유지합니다.
    # \s*는 문장 부호 뒤에 공백이 0개든 여러 개든 허용합니다.
    sentences_raw = re.split('(?<=[.?!])\s*', text)
    
    # 2. 결과 목록에서 빈 문자열 및 공백만 있는 문자열을 정리합니다.
    sentences = [s.strip() for s in sentences_raw if s.strip()]
    
    # 3. 토큰 길이 제한에 맞춰 문장들을 청크로 다시 묶습니다.
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        test_text = (current_chunk + " " + sentence).strip()
        token_ids = tokenizer.encode(test_text, add_special_tokens=True)
        encoded_len = len(token_ids)

        if encoded_len <= CHUNK_MAX_LENGTH:
            current_chunk = test_text
        else:
            if current_chunk:
                chunks.append(current_chunk)
            
            sentence_token_ids = tokenizer.encode(sentence, add_special_tokens=True)
            if len(sentence_token_ids) > MAX_TOKENS:
                print(f"경고: 단일 문장이 최대 토큰 길이({MAX_TOKENS})를 초과합니다. ({len(sentence_token_ids)}). 모델에서 잘림.")
            
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks

# 🌟 perform_translation_job 함수 (출력 토큰 제한 500으로 유지) 🌟
def perform_translation_job(
    job_id: str,
    text: str, 
    src_lang: str, 
    tgt_lang: str, 
    model: M2M100ForConditionalGeneration, 
    tokenizer: M2M100Tokenizer, 
    device: torch.device
):
    """
    백그라운드 스레드에서 실행되며, 번역 후 결과를 translation_jobs에 저장합니다.
    """
    print(f"[{job_id}] 번역 작업 시작...")
    translation_jobs[job_id]["status"] = "in_progress"
    
    try:
        tokenizer.src_lang = src_lang
        text_chunks = split_text_into_chunks(text, tokenizer)
        translated_chunks = []
        
        for chunk in text_chunks:
            
            encoded_input = tokenizer(
                chunk, 
                return_tensors="pt",
                padding="max_length", 
                truncation=True, 
                max_length=MAX_TOKENS # 500 토큰 제한 적용 (입력)
            )
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
            
            with torch.no_grad():
                generated_tokens = model.generate(
                    **encoded_input,
                    forced_bos_token_id=tokenizer.get_lang_id(tgt_lang),
                )
            
            translated_chunk = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            translated_chunks.append(translated_chunk)

        translated_text = " ".join(translated_chunks)
        
        # 성공 시 결과 저장 및 상태 업데이트
        result = TranslationResponse(
            original_text=text,
            translated_text=translated_text,
            src_lang=src_lang,
            tgt_lang=tgt_lang
        )
        translation_jobs[job_id]["result"] = result.dict() 
        translation_jobs[job_id]["status"] = "completed"
        
        src_lang_name = LANG_CODE_TO_NAME.get(src_lang, src_lang)
        tgt_lang_name = LANG_CODE_TO_NAME.get(tgt_lang, tgt_lang)
        print(f"[{job_id}] [{src_lang_name} -> {tgt_lang_name}] 번역 작업 완료.")

    except Exception as e:
        # 실패 시 오류 정보 저장 및 상태 업데이트
        print(f"[{job_id}] 번역 중 오류 발생: {e}")
        translation_jobs[job_id]["status"] = "failed"
        translation_jobs[job_id]["result"] = {"detail": f"번역 처리 중 내부 서버 오류 발생: {e}"}

# 1. 라우터 Description에 들어갈 상세 설명 문자열 생성 (유지)
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

# -------------------------------------------------------------
# 🎯 /auto 엔드포인트 전용 Description 정의 (유지)
# -------------------------------------------------------------
AUTO_DETECT_LANGS = [
    "af", "ar", "bg", "bn", "ca", "cs", "cy", "da", "de", "el", "en", "es", "et", "fa", 
    "fi", "fr", "gu", "he", "hi", "hr", "hu", "id", "it", "ja", "kn", "ko", "lt", "lv", 
    "mk", "ml", "mr", "ne", "nl", "no", "pa", "pl", "pt", "ro", "ru", "sk", "sl", "so", 
    "sq", "sv", "sw", "ta", "te", "th", "tl", "tr", "uk", "ur", "vi", "zh-cn", "zh-tw"
]

auto_lang_pairs = []
for code in AUTO_DETECT_LANGS:
    name = LANG_CODE_TO_NAME.get(code)
    if name == "Greeek": 
        name = "Greek"
    elif name is None:
        TEMP_NAME_MAP = {"zh-cn": "Chinese (Simplified)", "zh-tw": "Chinese (Traditional)"}
        name = TEMP_NAME_MAP.get(code, code)
        
    auto_lang_pairs.append(f"{name} : {code}")

CHUNK_SIZE = 3
chunks_3 = [auto_lang_pairs[i:i + CHUNK_SIZE] for i in range(0, len(auto_lang_pairs), CHUNK_SIZE)]
AUTO_LANG_DESCRIPTION_LIST = ["- " + ", ".join(chunk) for chunk in chunks_3]

ROUTER_DESCRIPTION_AUTO = (
    "**[자동 감지 모드]** 이 엔드포인트는 소스 언어(src_lang)를 자동으로 감지합니다. **src_lang 필드가 필요 없습니다.**\n\n"
    "### 🔎 자동 감지 지원 언어 (총 55개, 언어명 : 코드):\n"
    + "\n".join(AUTO_LANG_DESCRIPTION_LIST)
)

# -------------------------------------------------------------
# 🎯 기존 엔드포인트: POST /translate (src_lang 명시)
# -------------------------------------------------------------
@router.post("", description=ROUTER_DESCRIPTION)
async def translate_text(
    request: TranslationRequest,
    resources: Dict = Depends(get_translation_resources)
):
    model: M2M100ForConditionalGeneration = resources.get("model")
    tokenizer: M2M100Tokenizer = resources.get("tokenizer")
    device: torch.device = resources.get("device")
    supported_langs: Set[str] = resources.get("supported_langs")

    text = request.text
    src_lang = request.src_lang
    tgt_lang = request.tgt_lang
    
    # 소스 언어 지원 여부 확인
    if src_lang not in supported_langs:
        src_lang_name = LANG_CODE_TO_NAME.get(src_lang, "알 수 없는 언어")
        raise HTTPException(
            status_code=400, 
            detail=f"요청된 소스 언어 '{src_lang_name} ({src_lang})'은(는) M2M100에서 지원되지 않습니다."
        )
    
    # 🌟 백그라운드 작업 시작 🌟
    job_id = str(uuid.uuid4())
    translation_jobs[job_id] = {"status": "pending"}

    # 스레드 생성 및 실행 (CPU 바운드 작업 위임)
    thread = threading.Thread(
        target=perform_translation_job, 
        args=(job_id, text, src_lang, tgt_lang, model, tokenizer, device)
    )
    thread.start()
    
    # 즉시 응답 반환 (200 OK와 함께 작업 ID 반환)
    return {
        "job_id": job_id,
        "status": "pending",
        "message": "번역 작업이 백그라운드에서 시작되었습니다. /translate/result/{job_id} 엔드포인트로 결과를 확인하세요."
    }

# -------------------------------------------------------------
# 🎯 신규 엔드포인트: POST /translate/auto (src_lang 자동 감지)
# -------------------------------------------------------------
@router.post("/auto", description=ROUTER_DESCRIPTION_AUTO)
async def translate_text_auto(
    request: AutoTranslationRequest, 
    resources: Dict = Depends(get_translation_resources)
):
    model: M2M100ForConditionalGeneration = resources.get("model")
    tokenizer: M2M100Tokenizer = resources.get("tokenizer")
    device: torch.device = resources.get("device")
    supported_langs: Set[str] = resources.get("supported_langs")

    text = request.text
    tgt_lang = request.tgt_lang
    src_lang = None 

    # 1. 소스 언어 자동 감지
    try:
        detected_lang = detect(text)
        
        if detected_lang in supported_langs:
            src_lang = detected_lang
        else:
            detected_lang_name = LANG_CODE_TO_NAME.get(detected_lang, "알 수 없는 언어")
            raise HTTPException(
                status_code=400, 
                detail=f"자동 감지된 소스 언어 '{detected_lang_name} ({detected_lang})'은(는) M2M100에서 지원되지 않습니다."
            )
        
        detected_lang_name = LANG_CODE_TO_NAME.get(src_lang, src_lang)
        print(f"소스 언어 자동 감지: {detected_lang_name} ({src_lang})")
        
    except LangDetectException:
        raise HTTPException(status_code=400, detail="텍스트에서 소스 언어를 감지할 수 없습니다. 더 긴 텍스트를 제공하거나 기본 엔드포인트(/translate)를 사용해 src_lang을 지정해 주세요.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"언어 감지 중 내부 오류 발생: {e}")

    # 2. 백그라운드 작업 시작
    job_id = str(uuid.uuid4())
    translation_jobs[job_id] = {"status": "pending"}

    # 스레드 생성 및 실행 (CPU 바운드 작업 위임)
    thread = threading.Thread(
        target=perform_translation_job, 
        args=(job_id, text, src_lang, tgt_lang, model, tokenizer, device)
    )
    thread.start()
    
    # 즉시 응답 반환
    return {
        "job_id": job_id,
        "status": "pending",
        "message": "번역 작업이 백그라운드에서 시작되었습니다. /translate/result/{job_id} 엔드포인트로 결과를 확인하세요."
    }

# -------------------------------------------------------------
# 🎯 신규 엔드포인트: GET /translate/result/{job_id} (결과 확인)
# -------------------------------------------------------------
@router.get("/result/{job_id}", response_model=TranslationResponse, responses={
    200: {"model": TranslationResponse, "description": "번역 완료"},
    202: {"description": "작업 진행 중"},
    404: {"description": "작업 ID를 찾을 수 없음"},
    500: {"description": "번역 작업 실패"}
})
async def get_translation_result(job_id: str):
    job = translation_jobs.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail=f"작업 ID '{job_id}'를 찾을 수 없습니다.")

    status = job["status"]

    if status == "completed":
        # 성공 시, 저장된 결과를 반환
        return job["result"] 

    elif status == "failed":
        # 실패 시, 저장된 오류 정보와 함께 500 에러 반환
        raise HTTPException(status_code=500, detail=f"번역 작업 실패: {job['result'].get('detail', '상세 오류 확인 불가')}")
        
    else: # pending, in_progress
        # 작업이 아직 완료되지 않았음을 알리는 202 Accepted 응답 반환
        raise HTTPException(
            status_code=202,
            detail={
                "job_id": job_id,
                "status": status,
                "message": "번역 작업이 진행 중입니다. 잠시 후 다시 시도해 주세요."
            }
        )
