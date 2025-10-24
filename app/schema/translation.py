from pydantic import BaseModel, Field

LANG_CODE_TO_NAME = {
    "af": "Afrikaans", "am": "Amharic", "ar": "Arabic", "ast": "Asturian", 
    "az": "Azerbaijani", "ba": "Bashkir", "be": "Belarusian", "bg": "Bulgarian", 
    "bn": "Bengali", "br": "Breton", "bs": "Bosnian", "ca": "Catalan; Valencian", 
    "ceb": "Cebuano", "cs": "Czech", "cy": "Welsh", "da": "Danish", "de": "German", 
    "el": "Greeek", "en": "English", "es": "Spanish", "et": "Estonian", 
    "fa": "Persian", "ff": "Fulah", "fi": "Finnish", "fr": "French", 
    "fy": "Western Frisian", "ga": "Irish", "gd": "Gaelic; Scottish Gaelic", 
    "gl": "Galician", "gu": "Gujarati", "ha": "Hausa", "he": "Hebrew", 
    "hi": "Hindi", "hr": "Croatian", "ht": "Haitian; Haitian Creole", "hu": "Hungarian", 
    "hy": "Armenian", "id": "Indonesian", "ig": "Igbo", "ilo": "Iloko", 
    "is": "Icelandic", "it": "Italian", "ja": "Japanese", "jv": "Javanese", 
    "ka": "Georgian", "kk": "Kazakh", "km": "Central Khmer", "kn": "Kannada", 
    "ko": "Korean", "lb": "Luxembourgish; Letzeburgesch", "lg": "Ganda", 
    "ln": "Lingala", "lo": "Lao", "lt": "Lithuanian", "lv": "Latvian", 
    "mg": "Malagasy", "mk": "Macedonian", "ml": "Malayalam", "mn": "Mongolian", 
    "mr": "Marathi", "ms": "Malay", "my": "Burmese", "ne": "Nepali", 
    "nl": "Dutch; Flemish", "no": "Norwegian", "ns": "Northern Sotho", 
    "oc": "Occitan (post 1500)", "or": "Oriya", "pa": "Panjabi; Punjabi", 
    "pl": "Polish", "ps": "Pushto; Pashto", "pt": "Portuguese", 
    "ro": "Romanian; Moldavian; Moldovan", "ru": "Russian", "sd": "Sindhi", 
    "si": "Sinhala; Sinhalese", "sk": "Slovak", "sl": "Slovenian", "so": "Somali", 
    "sq": "Albanian", "sr": "Serbian", "ss": "Swati", "su": "Sundanese", 
    "sv": "Swedish", "sw": "Swahili", "ta": "Tamil", "th": "Thai", 
    "tl": "Tagalog", "tn": "Tswana", "tr": "Turkish", "uk": "Ukrainian", 
    "ur": "Urdu", "uz": "Uzbek", "vi": "Vietnamese", "wo": "Wolof", 
    "xh": "Xhosa", "yi": "Yiddish", "yo": "Yoruba", "zh": "Chinese", 
    "zu": "Zulu"
}

SUPPORTED_LANG_CODES = list(LANG_CODE_TO_NAME.keys())

# Swagger UI에 표시할 상세 설명 생성 (언어 이름 + 코드)
# 예: Afrikaans (af), Amharic (am), ...
LANG_PAIRS = [f"{name} ({code})" for code, name in LANG_CODE_TO_NAME.items()]
SUPPORTED_LANG_DESCRIPTION = (
    "M2M100 모델이 지원하는 언어 코드입니다. (ISO 639-1 기반):\n\n"
    + "  - " + "\n  - ".join(LANG_PAIRS)
)

class TranslationRequest(BaseModel):
    """
    번역 요청을 위한 입력 데이터 모델
    """
    text: str = Field(..., example="जीवन एक चॉकलेट बॉक्स की तरह है।", description="번역할 원문 텍스트")
    src_lang: str = Field(..., example="hi", description="원문의 소스 언어 코드 (ISO 639-1 또는 M2M100에서 지원하는 코드)")
    tgt_lang: str = Field(..., example="fr", description="번역을 원하는 타겟 언어 코드 (ISO 639-1 또는 M2M100에서 지원하는 코드)")

class TranslationResponse(BaseModel):
    """
    번역 응답 데이터 모델
    """
    original_text: str = Field(..., example="जीवन एक चॉकलेट बॉक्स की तरह है।", description="번역 요청 원문")
    translated_text: str = Field(..., example="La vie est comme une boîte de chocolats.", description="번역된 텍스트")
    src_lang: str = Field(..., example="hi", description="소스 언어 코드")
    tgt_lang: str = Field(..., example="fr", description="타겟 언어 코드")