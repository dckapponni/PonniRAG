"""
LLM layer (Gemini) for Tamil document answer generation.
Handles prompt construction, Gemini API calls (sync/async/streaming),
and answer post-processing for the Ponni RAG system.
"""
import os
import re
import logging
import time
import threading
from typing import List, Dict

from google import genai
from google.genai import types as genai_types
from guardrails import ANTI_INJECTION_PREAMBLE, sanitize_output

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
_gemini_client = None

PONNI_ABOUT_CONTEXT = """பொன்னி இதழ் பற்றிய பின்னணி தகவல்:
பொன்னி இதழ் திரு. அரு. பெரியண்ணன் மற்றும் திரு. முருகு. சுப்பிரமணியம் ஆகியோரால் 1947ஆம் ஆண்டு பிப்ரவரி மாதம் தொடங்கப்பெற்ற கலை இலக்கிய இதழ். 1947 முதல் 1955 வரை இயங்கியது. முதல் வருடத்தில் மாதம் ஓர் இதழ் என வெளிவந்த பொன்னி 1948 முதல் மாதம் ஈரிதழாக வெளிவந்தது.

காலவரிசை (Timeline): பொன்னி 1947 பிப்ரவரியில் தொடங்கியது. 1947ல் மாதம் ஒரு இதழ், 1948 முதல் மாதம் இரண்டு இதழ்கள். மொத்தம் 8 தொகுதிகள் (volumes) வெளியாகின: தொகுதி 1 (1947), தொகுதி 2 (1948), தொகுதி 3 (1949), தொகுதி 4 (1950), தொகுதி 5 (1951), தொகுதி 6 (1952), தொகுதி 7 (1953), தொகுதி 8 (1954-1955). 1955ல் இதழ் நிறுத்தப்பட்டது.

திராவிட இதழ்களின் வரிசையில் வைத்து போற்றத்தக்க இதழாகப் பொன்னி திகழ்கிறது. பகுத்தறிவு, சுயமரியாதை, சமத்துவம் ஆகியவற்றை மிகத் தீவிரமாக எடுத்துரைக்கும் இதழாக இவ்விதழ் வெளிவந்தது. வண்ண அட்டைப்படத்தில் மிக நேர்த்தியாக வடிவமைக்கப்பட்டு வெளியிடப்பெற்றது.

கவிஞர் கண்ணதாசன் தன் வனவாசம் புத்தகத்தில் 'திராவிடர் கழகத்தை ஆதரிக்கும் ஏடுகள் மிகக் குறைவாக இருந்த காலம். அந்த நேரத்தில் வண்ண முகப்பு அட்டை போட்டு அழகாக நடந்த இதழ் பொன்னி தான். பத்திரிக்கை துறையில் மற்றவர்கள் செய்துகாட்டாத புதுமை எல்லாம் அவர்கள் செய்து காட்டினார்கள்' என்று குறிப்பிட்டுள்ளார்.

தந்தை பெரியார், பேரறிஞர் அண்ணா, பாவேந்தர் பாரதிதாசன், திரு.வி.க., கலைஞர் மு. கருணாநிதி, கா. அப்பாதுரையார், கவியரசு கண்ணதாசன், டி. கே. சீனிவாசன், மு. வ., மு. அண்ணாமலை, கவிஞர் வாணிதாசன், கவிஞர் சுரதா போன்ற பல முதன்மையான இலக்கிய, அரசியல் ஆளுமைகள் பொன்னியில் எழுதியுள்ளனர்.

கவிதைகள், சிறுகதைகள், தொடர்கதைகள், நொடிக் கதைகள், நாடகங்கள், பொதுக் கட்டுரைகள், ஆய்வுக் கட்டுரைகள், ஒப்பாய்வுக் கட்டுரைகள், தொடர் கட்டுரைகள், செய்திப் பாட்டு போன்ற இலக்கிய வகைமைகளில் படைப்புகள் வெளியாகியுள்ளன.

தமிழ் இலக்கியம் மட்டுமன்றி சீனம், பாரசீகம், ரஷ்ய, கன்னடம், உருது, வடமொழி, தெலுங்கு போன்ற உலக இலக்கியங்களையும் பொன்னியில் அறிமுகம் செய்துள்ளனர்.

மாநில சுயாட்சி, இந்தித் திணிப்பு, தனித்தமிழ் பற்று, விடுதலை, அரசியல், சமூகம் சார்ந்த திராவிடச் சிந்தனைகள் கட்டுரைகளாக, கதைகளாக, கவிதைகளாக வெளிவந்தன. பாரதிதாசன் அவரின் 'குயில்' இதழ் அரசால் தடை செய்யப்பட்ட பிறகு பொன்னியில் எழுதினார். 'பாரதிதாசன் பரம்பரை கவிஞர்கள்' என்று அறிமுகப்படுத்தியது பொன்னி இதழ்.

பொன்னி இதழ் ஒரு கலை இலக்கிய இதழாக மட்டுமின்றி புரட்சி இதழாகவே இருந்திருக்கிறது. 1947 முதல் 1955 வரையிலான தமிழகத்தின் காலக் கண்ணாடியாகப் பொன்னி இதழ் விளங்குகிறது."""

TAMIL_ANSWER_SYSTEM_PROMPT = """நீங்கள் பொன்னி இதழ் தொடர்பான கேள்விகளுக்கு பதிலளிக்கும் ஒரு தமிழ் நிபுணர்.

உங்கள் பணி:
1. கொடுக்கப்பட்ட சூழல் (context) மற்றும் கேள்வியின் அடிப்படையில் விரிவான பதில் எழுதுக
2. பதில் 200 முதல் 500 சொற்கள் வரை இருக்க வேண்டும்
3. தெளிவான, எளிமையான, நடைமுறை தமிழில் எழுதுக
4. சூழலில் உள்ள தகவல்களை மட்டுமே பயன்படுத்துக – கற்பனையாக எதையும் சேர்க்காதீர்கள்
5. பதில் வாசிப்பதற்கு மிகவும் எளிதாகவும், நன்கு கட்டமைக்கப்பட்டதாகவும் இருக்க வேண்டும்
6. பதில் தொடங்கும் போதும் முடியும் போதும் எந்தச் சொலும் துண்டிக்கப்பட்டதாக இருக்கக் கூடாது
7. பதிலை எழுதி முடிக்கும் போது, கட்டாயமாக ஒரு முடிவு வாக்கியத்துடன் நிறுத்த வேண்டும். பதில் நடுவில் திடீரென நிற்கக் கூடாது

நேரடி கேள்விகளுக்கான விதி (மிக முக்கியம்):
- கேள்வி "யார்", "என்ன", "எப்போது", "எங்கே", "ஏன்", "எப்படி", "எவ்வளவு", "எத்தனை" போன்ற நேரடி தகவல் கேள்வியாக இருந்தால்:
  * முதல் வாக்கியத்திலேயே கேள்விக்கான நேரடி பதிலை தெளிவாகக் கூறுக (எ.கா. "இந்தக் கட்டுரையை எழுதியவர் ...")
  * கேள்வி கேட்பதை மறக்காமல், அதற்கான குறிப்பிட்ட தகவலை சூழலிலிருந்து கண்டுபிடித்து பதிலளிக்கவும்
  * பொதுவான சுருக்கம் எழுதாதீர்கள் — கேள்வியில் கேட்கப்பட்ட குறிப்பிட்ட தகவலை மட்டும் முன்னிலைப்படுத்துக
  * பதிலை நேரடியாக தொடங்குக — அறிமுகம் அல்லது பின்னணி விளக்கத்துடன் தொடங்க வேண்டாம்

விவரிப்பு வகை கேள்விகளுக்கான விதி:
- பயனர் விவரிப்பு வகையான கேள்வி கேட்டால் (எ.கா. "விளக்குக", "விவரி", "சுருக்கமாக கூறுக"), அனைத்து ஆவணங்களின் உள்ளடக்கத்தையும் ஒருங்கிணைத்து, சுருக்கமாக விவரித்து, ஆதாரங்களுடன் பதிலளிக்கவும்
- ஒவ்வொரு ஆவணத்தின் முக்கிய கருத்துகளையும் தெளிவாக சுருக்கி, முழுமையான பதிலை எழுதுக

சுருக்கம் / தலைப்பு சார்ந்த கேள்விகளுக்கான விதி (மிக முக்கியம்):
- கேள்வி ஒரு குறிப்பிட்ட தலைப்பு, கட்டுரை, அல்லது கருத்தை சுருக்கமாகக் கூற கேட்டால், ஆவண சூழலில் கொடுக்கப்பட்ட அனைத்து ஆவணங்களின் உள்ளடக்கத்தையும் பகுப்பாய்வு செய்து சுருக்கமாக எழுதுக
- சூழலில் உள்ள அனைத்து முக்கிய கருத்துகள், வாதங்கள், மற்றும் தகவல்களை உள்ளடக்கிய முழுமையான சுருக்கத்தை வழங்குக
- "போதுமான தகவல் இல்லை" என்று கூறாதீர்கள் — சூழலில் உள்ள தகவல்களைக் கொண்டு எவ்வளவு முடியுமோ அவ்வளவு விரிவாக பதிலளிக்கவும்
- அனைத்து ஆவணங்களிலிருந்தும் சம அளவில் தகவல்கள் கொடுக்கப்பட்டிருக்கும், ஒவ்வொரு ஆவணத்தின் முக்கிய கருத்துகளையும் பயன்படுத்துக

பொன்னி இதழ் பற்றிய கேள்விகளுக்கான விதி (மிக முக்கியம்):
- கேள்வி பொன்னி இதழைப் பற்றியதாக இருந்தால், கீழே கொடுக்கப்பட்ட "பொன்னி பின்னணி தகவல்" பகுதியை **முதன்மையாக** பயன்படுத்தி பதிலளிக்கவும்
- பின்வரும் வகையான கேள்விகள் அனைத்தும் பொன்னி இதழ் பற்றிய கேள்விகள்:
  * பொன்னி இதழ் என்ன? / பொன்னி பற்றி கூறுக / பொன்னி இதழின் வரலாறு
  * பொன்னி காலவரிசை / timeline / எப்போது தொடங்கியது / எப்போது நிறுத்தப்பட்டது
  * பொன்னி நிறுவனர் / யார் தொடங்கினர் / founder / who started ponni
  * பொன்னி எவ்வளவு காலம் இயங்கியது / how long did ponni run
  * பொன்னி தொகுதிகள் / volumes / எத்தனை இதழ்கள்
  * பொன்னி முக்கியத்துவம் / significance / importance / சிறப்பு
  * பொன்னியில் யார் எழுதினர் / who wrote in ponni / contributors / எழுத்தாளர்கள்
  * பொன்னி உள்ளடக்கம் / content types / என்ன வகையான படைப்புகள்
  * பொன்னி திராவிட இயக்கம் / Dravidian movement / பங்களிப்பு
  * what is ponni / about ponni / ponni magazine / ponni history
  * ponni timeline / ponni founding / ponni duration / ponni volumes
- இந்த வகையான கேள்விகளுக்கு, ஆவண சூழலை (document context) விட பொன்னி பின்னணி தகவலுக்கு முன்னுரிமை கொடுக்கவும்
- பொன்னி பின்னணி தகவலுடன் ஆவண சூழலையும் இணைத்து முழுமையான பதிலை எழுதுக
- கேள்வி "செய்திகள்", "பற்றி", "விவரம்" போன்றதாக இருந்தால்,மொழி பகுப்பாய்வு அல்லது இலக்கண விளக்கம் எழுதக்கூடாது.ஆவணங்களில் உள்ள தகவலை மட்டும் சுருக்கமாக விளக்க வேண்டும்.
---
பொன்னி பின்னணி தகவல்:
""" + PONNI_ABOUT_CONTEXT + """
---

மிக முக்கியமான வடிவமைப்பு விதிகள் (Formatting Rules):
- கேள்வி **புள்ளிவாரியான (points-wise)** பதிலை எதிர்பார்க்குமானால்:
  * ஒவ்வொரு புள்ளியும் தனித்தனி வரியில் எழுதப்பட வேண்டும்
  * ஒரு புள்ளி முடிந்தவுடன் அடுத்த புள்ளி புதிய வரியில் தொடங்க வேண்டும்
  * புள்ளிகளுக்கிடையே சரியான வரி இடைவெளி இருக்க வேண்டும்
  * ஒரே புள்ளியில் பல கருத்துகளை கலக்கக் கூடாது

- கேள்வி **பத்திவாரியான (paragraph-wise)** பதிலை எதிர்பார்க்குமானால்:
  * ஒவ்வொரு பத்தியும் தனித்தனி வரியில் இருக்க வேண்டும்
  * ஒவ்வொரு பத்தியின் முன்பும் பின்பும் ஒரு காலி வரி (spacing) இருக்க வேண்டும்
  * மிக நீளமான ஒரே பத்தியாக எழுதக்கூடாது
  * ஒவ்வொரு பத்தியும் ஒரு முக்கிய கருத்தை மட்டும் விளக்க வேண்டும்

எழுதும் முறை:
- முதல் வாக்கியத்தில் கேள்விக்கான நேரடியான பதிலை தெளிவாக கூறுக
- அதன் பின்னர் விவரங்கள், விளக்கங்கள், எடுத்துக்காட்டுகளை ஒழுங்காக எழுதுக
- தேவையான இடங்களில் துணைத்தலைப்புகளை பயன்படுத்தலாம்
- இறுதியில் சுருக்கமான முடிவுரை எழுதலாம்

கவனிக்க வேண்டியவை:
- CSV உள்ளடக்கத்தை பயன்படுத்தாமல் பதில் எழுதக்கூடாது.
- CSV தகவல் தொடர்பில்லையெனில் அதனை தெளிவாக குறிப்பிட வேண்டும்.
- சூழலில் இல்லாத தகவல்களை எதையும் எழுதாதீர்கள்
- "சூழலின் படி", "ஆதாரத்தின் படி" போன்ற சொற்களை பயன்படுத்த வேண்டாம்
- வாசிப்பவரின் கண்களுக்கு சோர்வு வராத வகையில் பதிலை அமைக்க வேண்டும்

இப்போது, கீழே கொடுக்கப்பட்ட கேள்வி மற்றும் சூழலின் அடிப்படையில், மேலுள்ள அனைத்து விதிகளையும் கட்டாயமாக பின்பற்றி, தெளிவாகவும் வாசிக்க எளிதாகவும் விரிவான பதிலை எழுதுக.""" + ANTI_INJECTION_PREAMBLE

_CSV_SYSTEM_PROMPT = """நீங்கள் பொன்னி இதழ் கட்டுரை தரவுத்தளத்தின் தகவல்களை வைத்து கேள்விகளுக்கு பதிலளிக்கும் தமிழ் உதவியாளர்.

முக்கிய விதி: கேள்வியை கவனமாக படித்து, கேள்வி வகைக்கு ஏற்ப பதிலளிக்கவும்.

கேள்வி வகைகள்:
1. ஆம்/இல்லை கேள்வி ("எழுதியுள்ளாரா?", "உள்ளதா?", "இருக்கிறதா?"):
   - முதல் வார்த்தையாக "ஆம்" அல்லது "இல்லை" என்று தெளிவாகக் கூறுக
   - பின்னர் ஓரிரு வாக்கியங்களில் காரணத்தை விளக்குக
   - எ.கா. "ஆம், ஆ. வ. இராமநாதன் பொன்னியில் 5 கட்டுரைகள் எழுதியுள்ளார்."

2. நேரடி கேள்வி ("யார்?", "என்ன?", "எப்போது?", "எத்தனை?"):
   - முதல் வாக்கியத்தில் நேரடியான பதிலைக் கூறுக
   - எ.கா. "மொத்தம் 12 கட்டுரைகள் கண்டறியப்பட்டுள்ளன."

3. பட்டியல் / சுருக்கக் கேள்வி:
   - மொத்த எண்ணிக்கை, முக்கிய பெயர்கள், பொதுவான போக்குகளை சுருக்கமாகக் குறிப்பிடுக
   - தரவை அப்படியே பட்டியலிடாதீர்கள்

பொது விதிகள்:
- பதிலை இயல்பான தமிழ் உரைநடையில் எழுதுக
- பதில் 30 முதல் 150 சொற்கள் வரை இருக்க வேண்டும்
- பதிலை ஒரு முடிவு வாக்கியத்துடன் நிறுத்த வேண்டும்
- தரவில் "Error" அல்லது "கண்டுபிடிக்க முடியவில்லை" இருந்தால், "இல்லை" என்று தெளிவாகக் கூறுக""" + ANTI_INJECTION_PREAMBLE


def _truncate_at_sentence_boundary(text: str) -> str:
    """Truncate text at the last complete sentence if it ends mid-sentence."""
    if not text:
        return text
    stripped = text.rstrip()
    if stripped and stripped[-1] in '.?!।':
        return stripped
    last_boundary = max(stripped.rfind('. '), stripped.rfind('.'), stripped.rfind('? '), stripped.rfind('?'),
                        stripped.rfind('! '), stripped.rfind('!'), stripped.rfind('।'))
    if last_boundary > len(stripped) * 0.5:
        return stripped[:last_boundary + 1].rstrip()
    return stripped


def _get_gemini_client():
    """Get or create the Gemini API client (singleton)."""
    global _gemini_client
    if _gemini_client is None:
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client


_gemini_health_cache = {"result": None, "timestamp": 0}
_gemini_health_lock = threading.Lock()
_GEMINI_HEALTH_TTL = 60  # seconds — recheck every 60s


def check_gemini_health(ttl: int = _GEMINI_HEALTH_TTL) -> Dict:
    """Check Gemini API health with TTL caching.

    Returns dict with keys: healthy (bool), message (str), error (str|None),
    model (str), latency_ms (float|None).
    """
    now = time.time()
    with _gemini_health_lock:
        cached = _gemini_health_cache
        if cached["result"] is not None and (now - cached["timestamp"]) < ttl:
            return cached["result"]

    # Outside lock: perform the actual check
    if not GEMINI_API_KEY:
        result = {
            "healthy": False,
            "error": "api_key_missing",
            "message": "GEMINI_API_KEY not configured",
            "model": GEMINI_MODEL,
            "latency_ms": None,
        }
    else:
        t0 = time.time()
        try:
            client = _get_gemini_client()
            client.models.generate_content(
                model=GEMINI_MODEL,
                contents="ping",
                config=genai_types.GenerateContentConfig(max_output_tokens=5),
            )
            latency = (time.time() - t0) * 1000
            result = {
                "healthy": True,
                "message": "Gemini API is responsive",
                "error": None,
                "model": GEMINI_MODEL,
                "latency_ms": round(latency, 1),
            }
        except Exception as e:
            latency = (time.time() - t0) * 1000
            result = {
                "healthy": False,
                "error": "api_call_failed",
                "message": "Gemini API is not responding",
                "model": GEMINI_MODEL,
                "latency_ms": round(latency, 1),
            }
            logger.error(f"Gemini health check failed: {e}")

    with _gemini_health_lock:
        _gemini_health_cache["result"] = result
        _gemini_health_cache["timestamp"] = time.time()

    return result


def _gemini_generation_config(
    system_instruction: str = None,
    disable_thinking: bool = False,
    max_output_tokens: int = 4096,
):
    """Return Gemini generation config with the given system instruction."""
    thinking_config = None
    if disable_thinking:
        thinking_config = genai_types.ThinkingConfig(thinking_budget=0)
    return genai_types.GenerateContentConfig(
        system_instruction=system_instruction or TAMIL_ANSWER_SYSTEM_PROMPT,
        temperature=0.0,
        max_output_tokens=max_output_tokens,
        top_p=0.9,
        thinking_config=thinking_config,
    )


_WH_PATTERNS = [
    'யார்', 'என்ன', 'எப்போது', 'எங்கே', 'எது', 'ஏன்', 'எப்படி',
    'எவ்வளவு', 'எத்தனை', 'யாவை', 'எவை',
    'who', 'what', 'when', 'where', 'which', 'why', 'how',
]


def _is_wh_question(question: str) -> bool:
    """Return True if the question is a wh/how type needing a direct answer."""
    q = question.lower()
    return any(p in q for p in _WH_PATTERNS)


def _build_user_content(
    question: str, context: str, csv_context: str, context_doc_count: int = 0,
) -> str:
    """Build the user prompt for vector-search queries. Omits empty sections.

    Repeats the question after the context block so the model attends to the
    question from both sides of the context (improves answer relevance,
    inspired by arxiv 2512.14982).
    """
    parts = [f"கேள்வி:\n{question}\n"]

    if csv_context and csv_context.strip():
        parts.append(f"========================\nCSV உள்ளடக்கம்:\n{csv_context}\n========================\n")

    if context and context.strip():
        parts.append(f"========================\nஆவண சூழல்:\n{context}\n========================\n")

    # Repeat the question after context so it's fresh in the model's attention
    parts.append(f"கேள்வி: {question}")

    if _is_wh_question(question):
        closing = (
            "மேலே உள்ள சூழலைப் பயன்படுத்தி, கேள்விக்கான நேரடியான பதிலை "
            "முதல் வாக்கியத்தில் தெளிவாகக் கூறுக. பின்னர் ஆதாரங்களுடன் "
            "விளக்கவும் (100-300 சொற்கள்):"
        )
    else:
        closing = "விரிவான பதில் (200-500 சொற்கள்):"

    if context_doc_count > 1:
        closing = (
            f"மேலே {context_doc_count} ஆவணங்கள் கொடுக்கப்பட்டுள்ளன. "
            f"அனைத்து ஆவணங்களின் தகவல்களையும் ஒருங்கிணைத்து பதிலளிக்கவும். "
            f"ஒரே ஆவணத்தை மட்டும் சுருக்காதீர்கள்.\n{closing}"
        )

    parts.append(closing)
    return "\n".join(parts)


_YES_NO_PATTERNS = [
    'உள்ளாரா', 'உள்ளதா', 'இருக்கிறதா', 'இருக்கிறாரா',
    'எழுதியுள்ளாரா', 'எழுதினாரா', 'எழுதியிருக்கிறாரா',
    'உண்டா', 'இல்லையா', 'ஆகுமா', 'முடியுமா',
    'செய்தாரா', 'செய்துள்ளாரா', 'பங்களித்தாரா',
]


def _is_yes_no_question(question: str) -> bool:
    """Return True if the question expects a yes/no answer."""
    q = question.lower()
    return any(p in q for p in _YES_NO_PATTERNS)


def _build_csv_user_content(question: str, csv_data: str) -> str:
    """Build a focused user prompt for CSV-only queries.

    Repeats the question after the data block and adds a question-type
    aware closing instruction.
    """
    # Determine closing instruction based on question type
    if _is_yes_no_question(question):
        closing = (
            "கேள்விக்கு முதலில் 'ஆம்' அல்லது 'இல்லை' என்று தெளிவாகக் கூறுக. "
            "பின்னர் ஓரிரு வாக்கியங்களில் விளக்குக (30-100 சொற்கள்)."
        )
    elif _is_wh_question(question):
        closing = (
            "கேள்விக்கான நேரடியான பதிலை முதல் வாக்கியத்தில் கூறுக. "
            "பின்னர் சுருக்கமாக விளக்குக (30-150 சொற்கள்)."
        )
    else:
        closing = (
            "தரவுத்தள தகவலை சுருக்கமாக விளக்கவும் (50-150 சொற்கள்). "
            "தரவை அப்படியே திரும்ப எழுதாதீர்கள்."
        )

    return f"""கேள்வி:
{question}

========================
கட்டுரை தரவுத்தள தகவல்:
{csv_data}
========================

கேள்வி: {question}
{closing}
"""


def _build_multi_turn_contents(history: list, current_user_content: str) -> list:
    """Build Gemini multi-turn contents list from conversation history.

    Each history entry is {"role": "user"|"assistant", "content": str}.
    Maps to Gemini format: role="user" or role="model".
    The current question (with document context) goes last.
    """
    contents = []
    for turn in history:
        role = "model" if turn["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": turn["content"]}]})
    contents.append({"role": "user", "parts": [{"text": current_user_content}]})
    return contents


def generate_llm_answer(
    question: str, context: str, csv_context: str, max_words: int = 500,
    user_content: str = None, system_prompt: str = None,
    disable_thinking: bool = False, context_doc_count: int = 0,
    history: list = None,
) -> str:
    """
    Generate LLM answer using Gemini API (synchronous).
    Pass user_content/system_prompt to override defaults (e.g. for CSV queries).
    """
    try:
        client = _get_gemini_client()
        if user_content is None:
            user_content = _build_user_content(
                question, context, csv_context,
                context_doc_count=context_doc_count,
            )

        contents = _build_multi_turn_contents(history, user_content) if history else user_content

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=_gemini_generation_config(system_prompt, disable_thinking=disable_thinking),
        )

        answer = (response.text or "").strip()

        answer = re.sub(r'[^\S\n]+', ' ', answer)
        answer = re.sub(r'\n{3,}', '\n\n', answer)

        # Check if response was truncated due to token limit
        if (response.candidates and
                response.candidates[0].finish_reason and
                str(response.candidates[0].finish_reason) == "MAX_TOKENS"):
            answer = _truncate_at_sentence_boundary(answer)
            logger.info("Response hit token limit — truncated at sentence boundary")

        word_count = len(re.findall(r'[\u0B80-\u0BFF]+|\w+', answer))
        logger.info(f"Gemini answer generated: {word_count} words")

        answer = sanitize_output(answer)
        return answer

    except Exception as e:
        logger.error(f"Gemini generation failed: {e}", exc_info=True)
        return ""


async def generate_llm_answer_async(
    question: str, context: str, csv_context: str, max_words: int = 500,
    user_content: str = None, system_prompt: str = None,
    disable_thinking: bool = False, context_doc_count: int = 0,
    history: list = None,
) -> str:
    """
    Async version of generate_llm_answer using Gemini API.
    Pass user_content/system_prompt to override defaults (e.g. for CSV queries).
    """
    try:
        client = _get_gemini_client()
        if user_content is None:
            user_content = _build_user_content(
                question, context, csv_context,
                context_doc_count=context_doc_count,
            )

        contents = _build_multi_turn_contents(history, user_content) if history else user_content

        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=_gemini_generation_config(system_prompt, disable_thinking=disable_thinking),
        )

        answer = (response.text or "").strip()

        answer = re.sub(r'[^\S\n]+', ' ', answer)
        answer = re.sub(r'\n{3,}', '\n\n', answer)

        if (response.candidates and
                response.candidates[0].finish_reason and
                str(response.candidates[0].finish_reason) == "MAX_TOKENS"):
            answer = _truncate_at_sentence_boundary(answer)
            logger.info("Async response hit token limit — truncated at sentence boundary")

        word_count = len(re.findall(r'[\u0B80-\u0BFF]+|\w+', answer))
        logger.info(f"Gemini async answer generated: {word_count} words")

        answer = sanitize_output(answer)
        return answer

    except Exception as e:
        logger.error(f"Gemini async generation failed: {e}", exc_info=True)
        return ""


def generate_llm_answer_stream(
    question: str, context: str, csv_context: str,
    user_content: str = None, system_prompt: str = None,
    disable_thinking: bool = False, context_doc_count: int = 0,
    history: list = None,
):
    """
    Generate LLM answer using Gemini API with streaming.
    Yields individual token strings as they arrive.
    Pass user_content/system_prompt to override defaults (e.g. for CSV queries).
    """
    try:
        client = _get_gemini_client()
        if user_content is None:
            user_content = _build_user_content(
                question, context, csv_context,
                context_doc_count=context_doc_count,
            )

        contents = _build_multi_turn_contents(history, user_content) if history else user_content

        for chunk in client.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=contents,
            config=_gemini_generation_config(system_prompt, disable_thinking=disable_thinking),
        ):
            token = chunk.text
            if token:
                yield token

    except Exception as e:
        logger.error(f"Gemini streaming generation failed: {e}", exc_info=True)
        return


def generate_extractive_answer(facts: List[Dict], question: str) -> str:
    """Generate fallback extractive answer if LLM fails."""
    if not facts:
        return ""

    answer_sentences = []
    for fact in facts[:5]:
        sent = fact['sentence'].strip()
        sent = re.sub(r'__.*?__|பொன்னி களஞ்சியம்', '', sent)
        sent = re.sub(r'\s+', ' ', sent).strip()
        if len(sent) > 30:
            answer_sentences.append(sent)

    if not answer_sentences:
        return ""

    answer = '. '.join(answer_sentences)
    if answer and answer[-1] not in '.!?।':
        answer += '.'

    return answer


def validate_gemini_api():
    """
    Validate that the Gemini API key is configured and working.
    Called at startup to fail fast if misconfigured.
    Uses check_gemini_health() to avoid duplicating ping logic.
    """
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not set — LLM generation will be unavailable")
        return
    logger.info(f"Validating Gemini API key (model: {GEMINI_MODEL})...")
    result = check_gemini_health()
    if result["healthy"]:
        logger.info("Gemini API validated successfully")
    else:
        logger.warning(f"Gemini API validation failed (non-fatal): {result['message']}")
