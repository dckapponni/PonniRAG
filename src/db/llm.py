"""LLM layer (Gemini) for Tamil document answer generation.

Handles prompt construction, Gemini API calls (sync/async/streaming),
and answer post-processing for the Ponni RAG system.
"""

import json
import logging
import os
import re
import threading
import time
from typing import Dict, List

import boto3
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from guardrails import ANTI_INJECTION_PREAMBLE, sanitize_output
from retry import is_retryable_gemini, with_gemini_retry, with_gemini_retry_async

from config.config import S3_BUCKET, S3_PREFIX

load_dotenv()
logger = logging.getLogger(__name__)


s3_client = boto3.client("s3")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
_gemini_client = None


def load_remaining_context_s3(doc_id: str, max_chars: int = 3000) -> str:
    """Load supplementary article content for a given document ID from S3.

    Fetches the JSON file at ``{S3_PREFIX}remaining_vol{doc_id}.json`` from
    the configured S3 bucket, iterates over every article in every issue, and
    concatenates up to 400 characters of each article's content into a single
    string capped at ``max_chars``.

    Args:
        doc_id (str): Volume/document identifier used to construct the S3 key
            (e.g. ``"3"`` → ``remaining_vol3.json``).
        max_chars (int): Maximum number of characters to return from the
            concatenated content. Defaults to ``3000``.

    Returns:
        str: Concatenated article content up to ``max_chars`` characters, with
        articles separated by double newlines. Returns an empty string if the
        S3 fetch fails or no qualifying article content is found.
    """
    try:
        key = f"{S3_PREFIX}remaining_vol{doc_id}.json"

        response = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        data = json.loads(response["Body"].read().decode("utf-8"))

        texts = []
        for issue in data.get("issues", []):
            for art in issue.get("articles", []):
                content = art.get("content", "")
                if content and len(content) > 50:
                    texts.append(content[:400])

        return "\n\n".join(texts)[:max_chars]

    except Exception as e:
        logger.warning(f"S3 remaining load failed for vol{doc_id}: {e}")
        return ""


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

ENGLISH_ANSWER_SYSTEM_PROMPT = (
    """You are an expert on Ponni magazine who answers questions in English.

Your task:
1. Write a detailed answer based on the given context and question
2. The answer should be 200 to 500 words
3. Write in clear, simple English
4. Use ONLY the information from the context — do not make up anything
5. The answer should be easy to read and well-structured
6. No words should be cut off at the beginning or end of the answer
7. The answer MUST end with a complete closing sentence — do not stop abruptly mid-sentence

Rules for direct questions (very important):
- If the question is a direct information question ("who", "what", "when", "where", "why", "how", "how many"):
  * State the direct answer clearly in the first sentence
  * Find the specific information from the context and provide it
  * Do not write a generic summary — highlight only the specific information asked
  * Start with the direct answer — do not begin with an introduction or background

Rules for descriptive questions:
- For descriptive questions ("explain", "describe", "summarize"), integrate information from all documents, summarize briefly, and provide supporting evidence

Rules for meaning/theme/summary questions (very important):
- When asked about the meaning, theme, gist, or summary of a story, poem, or article:
  * DERIVE the meaning by analyzing the actual content provided in the context
  * Do NOT look for an explicit "theme" or "meaning" statement — synthesize it from the narrative, arguments, and content
  * If the document context contains text from the work itself, that IS sufficient to answer — analyze it
- Analyze all document content and write a comprehensive summary
- Include all key points, arguments, and information from the context
- Do not say "insufficient information" — use whatever is available to provide the fullest answer possible

Rules for questions about Ponni magazine (very important):
- For questions about Ponni magazine, use the "Ponni Background Information" section below as the PRIMARY source
- Combine Ponni background information with document context for a complete answer

---
Ponni Background Information:
"""
    + PONNI_ABOUT_CONTEXT
    + """
---

Formatting rules:
- For point-wise answers: each point on its own line with proper spacing
- For paragraph answers: each paragraph separated by a blank line
- Do not write one very long paragraph

Writing style:
- State the direct answer in the first sentence
- Then provide details, explanations, and examples in an organized manner
- Use subheadings where appropriate
- End with a brief conclusion if suitable

Important:
- Do not fabricate any information not in the context
- Avoid phrases like "according to the context" or "as per the source"
- Structure the answer so it is pleasant and easy to read
- NEVER reference document numbers (e.g. "Document 1", "Document 2/3") in your answer — the user cannot see them
- NEVER comment on which documents are relevant or irrelevant, or say "the other documents discuss a different topic"
- Write your answer as a seamless, natural response — as if you already know the information
- If CSV metadata is provided alongside document content, prioritize the document content for your answer — CSV metadata (title, author, volume) is supplementary, not the main answer

Critical — Irrelevant context rule:
- This rule applies ONLY when the document context is about a completely different topic than the question
- If the context contains text from the work being asked about (even without an explicit answer), DERIVE the answer from that content — do not say "not available"
- If the context is truly about an unrelated topic, clearly state: "This information is not currently available in the database"
- Do NOT extract information from unrelated documents to fabricate an incorrect answer"""
)

TAMIL_ANSWER_SYSTEM_PROMPT = (
    """நீங்கள் பொன்னி இதழ் தொடர்பான கேள்விகளுக்கு பதிலளிக்கும் ஒரு தமிழ் நிபுணர்.

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

கருத்து / சுருக்கம் / பொருள் வகை கேள்விகளுக்கான விதி (மிக மிக முக்கியம்):
- கேள்வி ஒரு கதை, கவிதை, கட்டுரை அல்லது படைப்பின் கருத்து, சுருக்கம், பொருள், அர்த்தம் பற்றியதாக இருந்தால்:
  * ஆவண சூழலில் கொடுக்கப்பட்ட உள்ளடக்கத்தை பகுப்பாய்வு செய்து கருத்தை நீங்களே பிரித்தெடுக்க வேண்டும்
  * "கருத்து" அல்லது "சுருக்கம்" என்று வெளிப்படையாக எழுதப்பட்டிருக்கும் என்று எதிர்பார்க்க வேண்டாம் — கதை/கவிதையின் உள்ளடக்கத்திலிருந்து நீங்களே உருவாக்குக
  * ஆவண சூழலில் அந்தப் படைப்பின் உரை (text) இருந்தால், அதுவே பதிலுக்கு போதுமானது — அதை பகுப்பாய்வு செய்யுக
  * "தகவல் இல்லை" என்று கூறக் கூடாது — உள்ளடக்கம் இருக்கும்போது அதிலிருந்து பதிலை உருவாக்குக

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
"""
    + PONNI_ABOUT_CONTEXT
    + """
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
- சூழலில் இல்லாத தகவல்களை எதையும் எழுதாதீர்கள்
- "சூழலின் படி", "ஆதாரத்தின் படி" போன்ற சொற்களை பயன்படுத்த வேண்டாம்
- வாசிப்பவரின் கண்களுக்கு சோர்வு வராத வகையில் பதிலை அமைக்க வேண்டும்
- ஆவண எண்களை (எ.கா. "ஆவணம் 1", "ஆவணம் 2/3") பதிலில் ஒருபோதும் குறிப்பிடக் கூடாது — பயனருக்கு அவை தெரியாது
- எந்த ஆவணம் தொடர்புடையது, எது தொடர்பில்லாதது என்று விளக்கக் கூடாது. "மற்ற ஆவணங்கள் வேறு தலைப்பில் உள்ளன" போன்ற வாக்கியங்களை எழுதக் கூடாது
- நீங்கள் ஏற்கனவே தகவலை அறிந்தவர் போல இயல்பான பதிலை எழுதுக
- CSV தகவல் (தலைப்பு, ஆசிரியர், தொகுதி) ஆவண உள்ளடக்கத்துடன் சேர்ந்து வந்தால், ஆவண உள்ளடக்கத்திற்கு முன்னுரிமை கொடுக்கவும் — CSV தகவல் துணைத் தகவல் மட்டுமே, முக்கிய பதில் அல்ல

கூடுதல் அறிவு விதி:
- இது துணை தகவல் மட்டும்
- இதில் உள்ள தகவல்கள் முழுமையாக அமைப்புடையதாக இருக்காது
- முக்கிய பதில் மூல ஆவணத்தின் அடிப்படையில் இருக்க வேண்டும்
- தேவையானபோது மட்டும் பயன்படுத்தவும்

மிக முக்கியம் — தொடர்பில்லாத சூழல் (Irrelevant context rule):
- இந்த விதி கொடுக்கப்பட்ட ஆவண சூழல் கேள்வியின் தலைப்புக்கு முற்றிலும் வேறு தலைப்பில் இருக்கும்போது மட்டுமே பொருந்தும்
- ஆவண சூழலில் கேள்வியில் குறிப்பிடப்பட்ட படைப்பின் உரை (text) இருந்தால் (வெளிப்படையான பதில் இல்லாவிட்டாலும்), அந்த உள்ளடக்கத்திலிருந்து பதிலை உருவாக்குக — "தகவல் இல்லை" என்று கூறாதீர்கள்
- ஆவண சூழல் உண்மையிலேயே தொடர்பில்லாத தலைப்பில் இருந்தால் மட்டுமே, "இந்தத் தகவல் தற்போது தரவுத்தளத்தில் இல்லை" என்று கூறுக
- தொடர்பில்லாத ஆவணங்களிலிருந்து தகவல்களை எடுத்து தவறான பதிலை உருவாக்கக் கூடாது

இப்போது, கீழே கொடுக்கப்பட்ட கேள்வி மற்றும் சூழலின் அடிப்படையில், மேலுள்ள அனைத்து விதிகளையும் கட்டாயமாக பின்பற்றி, தெளிவாகவும் வாசிக்க எளிதாகவும் விரிவான பதிலை எழுதுக."""
)

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
- தரவில் "Error" அல்லது "கண்டுபிடிக்க முடியவில்லை" இருந்தால், "இல்லை" என்று தெளிவாகக் கூறுக"""

_CSV_SYSTEM_PROMPT_EN = """You are an English assistant that answers questions using the Ponni magazine article database.

Key rule: Read the question carefully and answer according to the question type.

Question types:
1. Yes/No question ("did they write?", "is there?", "does it exist?"):
   - Start with "Yes" or "No" clearly
   - Then explain in one or two sentences
   - E.g. "Yes, A.V. Ramanathan wrote 5 articles in Ponni."

2. Direct question ("who?", "what?", "when?", "how many?"):
   - State the direct answer in the first sentence
   - E.g. "A total of 12 articles were found."

3. List / summary question:
   - Mention total counts, key names, and general trends briefly
   - Do not simply list the raw data

Additional Knowledge Rule:
- The "Additional Knowledge" section is secondary
- It may contain noisy or unstructured data
- Use it only if relevant to the question
- Always prioritize the main document context

General rules:
- Write in natural English prose
- Answer should be 30 to 150 words
- End with a complete sentence
- If the data shows "Error" or "not found", clearly say "No" """


def _detect_garbage_tail(text: str) -> int:
    """Detect the character index where an LLM answer degrades into raw document garbage.

    Splits the text into lines and scans forward for lines that match any of
    four garbage-signal patterns:

    - Tabular data rows (e.g. ``"சொல் 0 4 0"``).
    - Raw metadata dashes (``"key — value"`` patterns without prose context).
    - Long runs of text exceeding 300 characters with no sentence-ending
      punctuation (``.``, ``?``, ``!``, ``।``).
    - Repeated short CSV-like entries (two or more consecutive
      ``"word word digit digit digit"`` groups).

    When a garbage line is found after the first 30 % of the text, the
    character offset of the last known-good line is returned so the caller
    can truncate there. Early garbage detections (within the first 30 %) are
    treated as false positives and skipped.

    Args:
        text (str): Raw LLM-generated answer text.

    Returns:
        int: Character index of the start of the first garbage line,
        or ``-1`` if no garbage is detected or the text is shorter than
        200 characters.
    """
    if not text or len(text) < 200:
        return -1

    # Split into sentences/segments by Tamil & English sentence-enders
    # Walk forward and flag when we see garbage-like segments
    lines = text.split("\n")
    good_end = 0  # char offset of last known-good position
    char_offset = 0

    # Patterns that signal raw document dump
    _garbage_patterns = [
        # Tabular data: "word 0 4 0" or "word 0 2 0"
        re.compile(r"[\u0B80-\u0BFF]+\s+\d\s+\d\s+\d"),
        # Raw metadata: consecutive "key: value" dumps without prose
        re.compile(r"(?:^|\n)[\u0B80-\u0BFF]+\s*—\s*[\u0B80-\u0BFF]"),
        # Long strings with no punctuation (>300 chars without . ? ! ।)
        re.compile(r"[^.?!।]{300,}"),
        # Repeated short entries (like raw CSV rows)
        re.compile(r"([\u0B80-\u0BFF]+\s+[\u0B80-\u0BFF]+\s+\d\s+\d\s+\d\s*){2,}"),
    ]

    for line in lines:
        line_len = len(line) + 1  # +1 for the \n
        is_garbage = False

        for pattern in _garbage_patterns:
            if pattern.search(line):
                is_garbage = True
                break

        if is_garbage:
            # Found garbage — return the position before this line
            if good_end > len(text) * 0.3:
                return good_end
            # If garbage is too early, it might be a false positive
            # — skip and continue

        if not is_garbage and line.strip():
            good_end = char_offset + line_len

        char_offset += line_len

    return -1


def _truncate_at_sentence_boundary(text: str) -> str:
    """Truncate text at the last complete sentence if it ends mid-sentence.

    If the stripped text already ends with a recognized sentence-final
    punctuation mark (``.``, ``?``, ``!``, ``।``), it is returned as-is.
    Otherwise, the text is cut at the rightmost occurrence of any of those
    markers, provided the cut point falls after the first 30 % of the string
    (to avoid over-truncation on very short texts).

    Args:
        text (str): Text that may end abruptly mid-sentence.

    Returns:
        str: Text ending at a sentence boundary, or the original stripped
        text if no suitable boundary is found in the latter 70 % of the
        string.
    """
    if not text:
        return text
    stripped = text.rstrip()
    if stripped and stripped[-1] in ".?!।":
        return stripped
    last_boundary = max(
        stripped.rfind(". "),
        stripped.rfind("."),
        stripped.rfind("? "),
        stripped.rfind("?"),
        stripped.rfind("! "),
        stripped.rfind("!"),
        stripped.rfind("।"),
    )
    if last_boundary > len(stripped) * 0.3:
        return stripped[: last_boundary + 1].rstrip()
    return stripped


def _get_gemini_client():
    """Return the Gemini API client, creating it on the first call (singleton).

    Initializes a ``genai.Client`` with ``GEMINI_API_KEY`` and caches it in
    the module-level ``_gemini_client`` variable. Subsequent calls return the
    cached instance without re-initializing.

    Returns:
        genai.Client: Authenticated Gemini API client.

    Raises:
        ValueError: If ``GEMINI_API_KEY`` is not set in the environment.
    """
    global _gemini_client
    if _gemini_client is None:
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client


_gemini_health_cache = {"result": None, "timestamp": 0}
_gemini_health_lock = threading.Lock()
_GEMINI_HEALTH_TTL = 300  # seconds — recheck every 5 min (cuts API spam)


def check_gemini_health(ttl: int = _GEMINI_HEALTH_TTL) -> Dict:
    """Check Gemini API availability with TTL-based result caching.

    Sends a minimal ``"ping"`` prompt (``max_output_tokens=5``) to the
    configured model and measures round-trip latency. Results are cached in
    ``_gemini_health_cache`` for ``ttl`` seconds so that high-frequency
    callers do not hammer the API; a cached result is returned immediately
    within the TTL window.

    Args:
        ttl (int): Cache lifetime in seconds. Defaults to
            ``_GEMINI_HEALTH_TTL`` (60 s).

    Returns:
        Dict: A dictionary with the following keys:

        - ``"healthy"`` (bool): ``True`` if the API responded successfully.
        - ``"message"`` (str): Human-readable status description.
        - ``"error"`` (str | None): Error code string, or ``None`` on success.
          Possible values: ``"api_key_missing"``, ``"api_call_failed"``.
        - ``"model"`` (str): The model name used for the ping.
        - ``"latency_ms"`` (float | None): Round-trip time in milliseconds,
          or ``None`` if the key was missing.
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


def _mark_gemini_unhealthy(error_msg: str):
    """Poison the Gemini health cache after a retryable LLM failure.

    Writes a synthetic unhealthy result into ``_gemini_health_cache`` so that
    subsequent requests within the TTL window (60 s) detect the outage and
    skip directly to the fallback path instead of wasting time retrying an
    already-failing API.  Recovery is automatic once the TTL expires and the
    next health check succeeds.

    Called by :func:`generate_llm_answer` and
    :func:`generate_llm_answer_async` when all retries on a retryable error
    (HTTP 429, 5xx, timeout) are exhausted.

    Args:
        error_msg (str): Error message string from the caught exception,
            stored in the cache entry for diagnostics.

    Returns:
        None
    """
    with _gemini_health_lock:
        _gemini_health_cache["result"] = {
            "healthy": False,
            "error": "transient_failure",
            "message": f"Gemini call failed after retries: {error_msg}",
            "model": GEMINI_MODEL,
            "latency_ms": None,
        }
        _gemini_health_cache["timestamp"] = time.time()
    logger.warning("Gemini health cache poisoned due to transient failure")


def _gemini_generation_config(
    system_instruction: str = None,
    disable_thinking: bool = False,
    max_output_tokens: int = 4096,
):
    """Build a Gemini ``GenerateContentConfig`` with security preamble prepended.

    Prepends ``ANTI_INJECTION_PREAMBLE`` before the system instruction so
    that injection-defense rules are seen first, while the response-style
    rules (at the end of ``TAMIL_ANSWER_SYSTEM_PROMPT``) remain closest to
    the model's active attention window, preserving natural Tamil prose style.

    Args:
        system_instruction (str | None): Custom system prompt. Falls back to
            ``TAMIL_ANSWER_SYSTEM_PROMPT`` when ``None``.
        disable_thinking (bool): If ``True``, sets ``thinking_budget=0`` on
            the ``ThinkingConfig`` to suppress chain-of-thought reasoning and
            reduce latency. Defaults to ``False``.
        max_output_tokens (int): Maximum number of tokens in the generated
            response. Defaults to ``4096``.

    Returns:
        genai_types.GenerateContentConfig: Fully populated generation config
        ready to pass to ``client.models.generate_content``.
    """
    prompt = system_instruction or TAMIL_ANSWER_SYSTEM_PROMPT
    full_instruction = ANTI_INJECTION_PREAMBLE + "\n\n" + prompt

    thinking_config = None
    if disable_thinking:
        thinking_config = genai_types.ThinkingConfig(thinking_budget=0)
    return genai_types.GenerateContentConfig(
        system_instruction=full_instruction,
        temperature=0.0,
        max_output_tokens=max_output_tokens,
        top_p=0.9,
        thinking_config=thinking_config,
    )


_WH_PATTERNS = [
    "யார்",
    "என்ன",
    "எப்போது",
    "எங்கே",
    "எது",
    "ஏன்",
    "எப்படி",
    "எவ்வளவு",
    "எத்தனை",
    "யாவை",
    "எவை",
    "who",
    "what",
    "when",
    "where",
    "which",
    "why",
    "how",
]


def _is_wh_question(question: str) -> bool:
    """Return True if the question is a wh- or how-type question.

    Checks the lowercased question against ``_WH_PATTERNS``, a list of Tamil
    and English question words (``யார்``, ``என்ன``, ``who``, ``what``,
    ``when``, etc.). A match indicates the question expects a direct,
    fact-stating answer rather than a descriptive passage.

    Args:
        question (str): User question string.

    Returns:
        bool: ``True`` if any wh-pattern is found in the lowercased question;
        ``False`` otherwise.
    """
    q = question.lower()
    return any(p in q for p in _WH_PATTERNS)


_CONTENT_DISPLAY_PATTERNS = [
    "காட்டு",
    "காட்டுக",
    "படிக்க",
    "படி",
    "வாசிக்க",
    "முழு உள்ளடக்கம்",
    "முழு கதை",
    "முழு கட்டுரை",
    "முழு கவிதை",
    "கதையை காட்டு",
    "கட்டுரையை காட்டு",
    "கவிதையை காட்டு",
    "உள்ளடக்கத்தை காட்டு",
    "என்ன எழுதியுள்ளார்",
    "show content",
    "show the article",
    "show the story",
    "show the poem",
    "display",
    "read the",
    "full text",
    "full content",
]


def is_content_display_query(question: str) -> bool:
    """Return True if the question requests display of raw article content.

    Checks the lowercased question against ``_CONTENT_DISPLAY_PATTERNS``,
    which covers Tamil phrases (``முழு கட்டுரை``, ``காட்டு``) and English
    phrases (``full text``, ``show the article``, ``read the``). A match
    signals that the user wants the original text reproduced rather than a
    summary, which triggers an expanded context window in the orchestrator.

    Args:
        question (str): User question string.

    Returns:
        bool: ``True`` if any content-display pattern is found; ``False``
        otherwise.
    """
    q = question.lower()
    return any(p in q for p in _CONTENT_DISPLAY_PATTERNS)


def _get_system_prompt(language: str = "ta") -> str:
    """Return the appropriate answer system prompt for the given language.

    Args:
        language (str): BCP-47 language code. Pass ``"en"`` to receive
            ``ENGLISH_ANSWER_SYSTEM_PROMPT``; any other value returns
            ``TAMIL_ANSWER_SYSTEM_PROMPT``. Defaults to ``"ta"``.

    Returns:
        str: The system prompt string for the requested language.
    """
    if language == "en":
        return ENGLISH_ANSWER_SYSTEM_PROMPT
    return TAMIL_ANSWER_SYSTEM_PROMPT


def _get_csv_system_prompt(language: str = "ta") -> str:
    """Return the appropriate CSV query system prompt for the given language.

    Args:
        language (str): BCP-47 language code. Pass ``"en"`` to receive
            ``_CSV_SYSTEM_PROMPT_EN``; any other value returns
            ``_CSV_SYSTEM_PROMPT``. Defaults to ``"ta"``.

    Returns:
        str: The CSV-focused system prompt string for the requested language.
    """
    if language == "en":
        return _CSV_SYSTEM_PROMPT_EN
    return _CSV_SYSTEM_PROMPT


def _build_user_content(
    question: str,
    context: str,
    csv_context: str,
    context_doc_count: int = 0,
    language: str = "ta",
) -> str:
    """Build the user-turn prompt for vector-search-backed queries.

    Assembles a structured prompt from the question, optional CSV metadata,
    document context, and optional S3 supplementary content. Non-empty
    sections are wrapped in ``========================`` delimiters with
    descriptive labels. The question is repeated after the context block so
    the model attends to it from both sides of the context window (improves
    answer relevance).

    A closing instruction is appended based on the question type:

    - **Content display queries** (detected via :func:`is_content_display_query`):
      instructs the model to reproduce the original text as-is.
    - **Wh-questions** (detected via :func:`_is_wh_question`): instructs the
      model to state the direct answer in the first sentence.
    - **All other queries**: requests a 200–500 word descriptive answer.

    When ``context_doc_count > 1``, a multi-document integration reminder is
    prepended to the closing instruction.

    Args:
        question (str): User question in Tamil or English.
        context (str): Concatenated document text from vector search results.
        csv_context (str): Formatted CSV metadata rows from semantic search.
        context_doc_count (int): Number of distinct documents included in
            ``context``; used to add a multi-document integration reminder.
            Defaults to ``0``.
        language (str): BCP-47 language code controlling all label strings
            and closing instructions. Defaults to ``"ta"``.

    Returns:
        str: Fully assembled user-turn prompt string joined by newlines.
    """
    en = language == "en"
    q_label = "Question" if en else "கேள்வி"
    # Use neutral labels — the LLM sometimes repeats these verbatim
    csv_label = "Article Database" if en else "கட்டுரை தகவல்"
    doc_label = "Reference Material" if en else "மூல ஆவணம்"

    parts = [f"{q_label}:\n{question}\n"]

    if csv_context and csv_context.strip():
        parts.append(
            f"========================\n"
            f"{csv_label}:\n{csv_context}\n"
            f"========================\n"
        )

    if context and context.strip():
        parts.append(
            f"========================\n"
            f"{doc_label}:\n{context}\n"
            f"========================\n"
        )
    # 🔥 ADD THIS BLOCK (LLM-only remaining.json)
    remaining_context = ""

    try:
        doc_id_match = re.search(
            r"(?:மலர்|Volume|doc_id)\s*[:\-]?\s*(\d+)", csv_context or ""
        )
        if doc_id_match:
            doc_id = doc_id_match.group(1)
            remaining_context = load_remaining_context_s3(doc_id)
    except Exception as e:
        logger.warning(f"Doc_id extraction failed: {e}")

    if remaining_context:
        parts.append(
            f"========================\n"
            f"{'Additional Knowledge' if en else 'கூடுதல் அறிவு'}:\n"
            f"{remaining_context}\n"
            f"========================\n"
        )
    # Repeat the question after context so it's fresh in the model's attention
    parts.append(f"{q_label}: {question}")

    if is_content_display_query(question):
        if en:
            closing = (
                "The user wants to READ the actual content. "
                "Display the content from the context as completely as possible. "
                "Preserve the original text — do not summarize or paraphrase. "
                "Add a brief title/author header if available. "
                "Show as much of the original content as provided:"
            )
        else:
            closing = (
                "பயனர் உண்மையான உள்ளடக்கத்தை படிக்க விரும்புகிறார். "
                "சூழலில் உள்ள உள்ளடக்கத்தை முடிந்தவரை முழுமையாகக் காட்டுக. "
                "அசல் உரையை அப்படியே காட்டுக — சுருக்கமாகவோ "
                "மாற்றியோ எழுத வேண்டாம். "
                "தலைப்பு/ஆசிரியர் தகவல் இருந்தால் மேலே சுருக்கமாக குறிப்பிடுக. "
                "சூழலில் உள்ள அசல் உள்ளடக்கத்தை அதிகமாகக் காட்டுக:"
            )
    elif _is_wh_question(question):
        if en:
            closing = (
                "Using the context above, state the direct answer clearly in the "
                "first sentence. Then explain with evidence (100-300 words):"
            )
        else:
            closing = (
                "மேலே உள்ள சூழலைப் பயன்படுத்தி, கேள்விக்கான நேரடியான பதிலை "
                "முதல் வாக்கியத்தில் தெளிவாகக் கூறுக. பின்னர் ஆதாரங்களுடன் "
                "விளக்கவும் (100-300 சொற்கள்):"
            )
    else:
        closing = (
            "Detailed answer (200-500 words):"
            if en
            else "விரிவான பதில் (200-500 சொற்கள்):"
        )

    if context_doc_count > 1:
        if en:
            closing = (
                f"{context_doc_count} documents are provided above. "
                f"Integrate information from all documents in your answer. "
                f"Do not summarize only one document.\n{closing}"
            )
        else:
            closing = (
                f"மேலே {context_doc_count} ஆவணங்கள் கொடுக்கப்பட்டுள்ளன. "
                f"அனைத்து ஆவணங்களின் தகவல்களையும் ஒருங்கிணைத்து பதிலளிக்கவும். "
                f"ஒரே ஆவணத்தை மட்டும் சுருக்காதீர்கள்.\n{closing}"
            )

    parts.append(closing)
    return "\n".join(parts)


_YES_NO_PATTERNS = [
    "உள்ளாரா",
    "உள்ளதா",
    "இருக்கிறதா",
    "இருக்கிறாரா",
    "எழுதியுள்ளாரா",
    "எழுதினாரா",
    "எழுதியிருக்கிறாரா",
    "உண்டா",
    "இல்லையா",
    "ஆகுமா",
    "முடியுமா",
    "செய்தாரா",
    "செய்துள்ளாரா",
    "பங்களித்தாரா",
]


def _is_yes_no_question(question: str) -> bool:
    """Return True if the question expects a yes/no answer.

    Checks the lowercased question against ``_YES_NO_PATTERNS``, a list of
    Tamil verb suffixes and question particles that indicate a binary
    yes/no expectation (e.g. ``எழுதியுள்ளாரா``, ``உள்ளதா``, ``இருக்கிறாரா``).

    Args:
        question (str): User question string.

    Returns:
        bool: ``True`` if any yes/no pattern is found in the lowercased
        question; ``False`` otherwise.
    """
    q = question.lower()
    return any(p in q for p in _YES_NO_PATTERNS)


def _build_csv_user_content(question: str, csv_data: str, language: str = "ta") -> str:
    """Build a focused user-turn prompt for CSV-only (author/metadata) queries.

    Wraps the raw CSV data in a labeled delimiter block and appends a
    question-type-aware closing instruction. The question is repeated
    before the closing instruction so the model attends to it after
    processing the data.

    Closing instruction selection:

    - **Yes/no questions** (via :func:`_is_yes_no_question`): instructs the
      model to start with ``"ஆம்"`` / ``"இல்லை"`` (or ``"Yes"`` / ``"No"``).
    - **Wh-questions** (via :func:`_is_wh_question`): instructs the model to
      state the direct answer in the first sentence.
    - **All other questions**: instructs the model to summarize the data in
      50–150 words without repeating the raw rows.

    Args:
        question (str): User question in Tamil or English.
        csv_data (str): Pre-formatted CSV query result string from
            :func:`handle_author_query`.
        language (str): BCP-47 language code controlling all label strings
            and closing instructions. Defaults to ``"ta"``.

    Returns:
        str: Fully assembled user-turn prompt string.
    """
    en = language == "en"
    q_label = "Question" if en else "கேள்வி"
    db_label = "Article Database Information" if en else "கட்டுரை தரவுத்தள தகவல்"

    # Determine closing instruction based on question type
    if _is_yes_no_question(question):
        if en:
            closing = (
                "Start with 'Yes' or 'No' clearly. "
                "Then explain in one or two sentences (30-100 words)."
            )
        else:
            closing = (
                "கேள்விக்கு முதலில் 'ஆம்' அல்லது 'இல்லை' என்று தெளிவாகக் கூறுக. "
                "பின்னர் ஓரிரு வாக்கியங்களில் விளக்குக (30-100 சொற்கள்)."
            )
    elif _is_wh_question(question):
        if en:
            closing = (
                "State the direct answer in the first sentence. "
                "Then explain briefly (30-150 words)."
            )
        else:
            closing = (
                "கேள்விக்கான நேரடியான பதிலை முதல் வாக்கியத்தில் கூறுக. "
                "பின்னர் சுருக்கமாக விளக்குக (30-150 சொற்கள்)."
            )
    else:
        if en:
            closing = (
                "Summarize the database information briefly (50-150 words). "
                "Do not simply repeat the raw data."
            )
        else:
            closing = (
                "தரவுத்தள தகவலை சுருக்கமாக விளக்கவும் (50-150 சொற்கள்). "
                "தரவை அப்படியே திரும்ப எழுதாதீர்கள்."
            )

    return f"""{q_label}:
{question}

========================
{db_label}:
{csv_data}
========================

{q_label}: {question}
{closing}
"""


def _build_multi_turn_contents(history: list, current_user_content: str) -> list:
    """Build a Gemini-format multi-turn contents list from conversation history.

    Converts each turn in ``history`` from the internal
    ``{"role": "user"|"assistant", "content": str}`` format to the Gemini
    ``{"role": "user"|"model", "parts": [{"text": str}]}`` format, then
    appends the current user turn (which already includes document context)
    as the final entry.

    Args:
        history (list): Validated conversation history as a list of
            ``{"role": str, "content": str}`` dicts, ordered oldest-first.
        current_user_content (str): The fully assembled user-turn prompt for
            the current question, produced by :func:`_build_user_content` or
            :func:`_build_csv_user_content`.

    Returns:
        list: A list of Gemini content dicts ready to pass as the
        ``contents`` argument to ``client.models.generate_content``.
    """
    contents = []
    for turn in history:
        role = "model" if turn["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": turn["content"]}]})
    contents.append({"role": "user", "parts": [{"text": current_user_content}]})
    return contents


def generate_llm_answer(
    question: str,
    context: str,
    csv_context: str,
    max_words: int = 500,
    user_content: str = None,
    system_prompt: str = None,
    disable_thinking: bool = False,
    context_doc_count: int = 0,
    history: list = None,
    language: str = "ta",
    max_output_tokens: int = 4096,
) -> str:
    """Generate a complete LLM answer using the Gemini API (synchronous).

    Builds the generation config and user content (unless overrides are
    provided), constructs either a single-turn or multi-turn contents list,
    and calls the Gemini API via the retry wrapper. Post-processes the
    response by collapsing excess whitespace, truncating at sentence
    boundaries on ``MAX_TOKENS`` finish reason, stripping raw document
    garbage tails, and running output sanitization.

    Args:
        question (str): User question in Tamil or English.
        context (str): Concatenated document text from vector search results.
        csv_context (str): Formatted CSV metadata rows from semantic search.
        max_words (int): Target maximum word count hint (informational;
            the hard cap is ``max_output_tokens``). Defaults to ``500``.
        user_content (str | None): Pre-built user prompt. When provided,
            ``context``, ``csv_context``, and ``context_doc_count`` are
            ignored. Defaults to ``None``.
        system_prompt (str | None): Custom system instruction. Falls back to
            the language-appropriate default when ``None``. Defaults to
            ``None``.
        disable_thinking (bool): If ``True``, suppresses chain-of-thought
            reasoning (``thinking_budget=0``) to reduce latency. Defaults to
            ``False``.
        context_doc_count (int): Number of distinct documents in ``context``,
            used to add a multi-document integration reminder. Defaults to
            ``0``.
        history (list | None): Validated conversation history for multi-turn
            mode, or ``None`` for single-turn. Defaults to ``None``.
        language (str): BCP-47 language code for prompt and message
            selection. Defaults to ``"ta"``.
        max_output_tokens (int): Hard token cap on the generated response.
            Defaults to ``4096``.

    Returns:
        str: Cleaned, sanitized answer string. Returns an empty string if
        the API call fails or raises an exception; marks Gemini unhealthy
        in the health cache when the error is retryable.
    """
    try:
        client = _get_gemini_client()
        if system_prompt is None:
            system_prompt = _get_system_prompt(language)
        if user_content is None:
            user_content = _build_user_content(
                question,
                context,
                csv_context,
                context_doc_count=context_doc_count,
                language=language,
            )

        contents = (
            _build_multi_turn_contents(history, user_content)
            if history
            else user_content
        )

        response = with_gemini_retry(
            client.models.generate_content,
            model=GEMINI_MODEL,
            contents=contents,
            config=_gemini_generation_config(
                system_prompt,
                disable_thinking=disable_thinking,
                max_output_tokens=max_output_tokens,
            ),
        )

        answer = (response.text or "").strip()

        answer = re.sub(r"[^\S\n]+", " ", answer)
        answer = re.sub(r"\n{3,}", "\n\n", answer)

        # Check if response was truncated due to token limit
        if (
            response.candidates
            and response.candidates[0].finish_reason
            and str(response.candidates[0].finish_reason) == "MAX_TOKENS"
        ):
            answer = _truncate_at_sentence_boundary(answer)
            logger.info("Response hit token limit — truncated at sentence boundary")

        # Detect and remove raw document garbage at the end
        garbage_pos = _detect_garbage_tail(answer)
        if garbage_pos > 0:
            answer = _truncate_at_sentence_boundary(answer[:garbage_pos])
            logger.info(f"Truncated garbage tail at position {garbage_pos}")

        word_count = len(re.findall(r"[\u0B80-\u0BFF]+|\w+", answer))
        logger.info(f"Gemini answer generated: {word_count} words")

        answer = sanitize_output(answer)
        return answer

    except Exception as e:
        logger.error(f"Gemini generation failed: {e}", exc_info=True)
        if is_retryable_gemini(e):
            _mark_gemini_unhealthy(str(e))
        return ""


async def generate_llm_answer_async(
    question: str,
    context: str,
    csv_context: str,
    max_words: int = 500,
    user_content: str = None,
    system_prompt: str = None,
    disable_thinking: bool = False,
    context_doc_count: int = 0,
    history: list = None,
    language: str = "ta",
    max_output_tokens: int = 4096,
) -> str:
    """Generate a complete LLM answer using the Gemini API (asynchronous).

    Async counterpart of :func:`generate_llm_answer`. Calls
    ``client.aio.models.generate_content`` via the async retry wrapper so
    the event loop is not blocked during the Gemini round-trip. All
    post-processing steps (whitespace normalization, sentence-boundary
    truncation, garbage-tail detection, output sanitization) are identical
    to the synchronous version.

    Args:
        question (str): User question in Tamil or English.
        context (str): Concatenated document text from vector search results.
        csv_context (str): Formatted CSV metadata rows from semantic search.
        max_words (int): Target maximum word count hint (informational).
            Defaults to ``500``.
        user_content (str | None): Pre-built user prompt override. Defaults
            to ``None``.
        system_prompt (str | None): Custom system instruction override.
            Defaults to ``None``.
        disable_thinking (bool): If ``True``, suppresses chain-of-thought
            reasoning. Defaults to ``False``.
        context_doc_count (int): Number of distinct documents in ``context``.
            Defaults to ``0``.
        history (list | None): Validated conversation history for multi-turn
            mode, or ``None`` for single-turn. Defaults to ``None``.
        language (str): BCP-47 language code. Defaults to ``"ta"``.
        max_output_tokens (int): Hard token cap on the generated response.
            Defaults to ``4096``.

    Returns:
        str: Cleaned, sanitized answer string. Returns an empty string on
        failure; marks Gemini unhealthy in the health cache for retryable
        errors.
    """
    try:
        client = _get_gemini_client()
        if system_prompt is None:
            system_prompt = _get_system_prompt(language)
        if user_content is None:
            user_content = _build_user_content(
                question,
                context,
                csv_context,
                context_doc_count=context_doc_count,
                language=language,
            )

        contents = (
            _build_multi_turn_contents(history, user_content)
            if history
            else user_content
        )

        response = await with_gemini_retry_async(
            client.aio.models.generate_content,
            model=GEMINI_MODEL,
            contents=contents,
            config=_gemini_generation_config(
                system_prompt,
                disable_thinking=disable_thinking,
                max_output_tokens=max_output_tokens,
            ),
        )

        answer = (response.text or "").strip()

        answer = re.sub(r"[^\S\n]+", " ", answer)
        answer = re.sub(r"\n{3,}", "\n\n", answer)

        if (
            response.candidates
            and response.candidates[0].finish_reason
            and str(response.candidates[0].finish_reason) == "MAX_TOKENS"
        ):
            answer = _truncate_at_sentence_boundary(answer)
            logger.info(
                "Async response hit token limit — truncated at sentence boundary"
            )

        # Detect and remove raw document garbage at the end
        garbage_pos = _detect_garbage_tail(answer)
        if garbage_pos > 0:
            answer = _truncate_at_sentence_boundary(answer[:garbage_pos])
            logger.info(f"Async: truncated garbage tail at position {garbage_pos}")

        word_count = len(re.findall(r"[\u0B80-\u0BFF]+|\w+", answer))
        logger.info(f"Gemini async answer generated: {word_count} words")

        answer = sanitize_output(answer)
        return answer

    except Exception as e:
        logger.error(f"Gemini async generation failed: {e}", exc_info=True)
        if is_retryable_gemini(e):
            _mark_gemini_unhealthy(str(e))
        return ""


def generate_llm_answer_stream(
    question: str,
    context: str,
    csv_context: str,
    user_content: str = None,
    system_prompt: str = None,
    disable_thinking: bool = False,
    context_doc_count: int = 0,
    history: list = None,
    language: str = "ta",
    max_output_tokens: int = 4096,
):
    """Generate an LLM answer using the Gemini API with token streaming.

    Calls ``client.models.generate_content_stream`` via the retry wrapper
    and yields each non-empty ``chunk.text`` string as it arrives from the
    API. The caller is responsible for accumulating tokens, post-processing,
    and caching; this generator yields raw token strings with no additional
    transformation.

    On exception, logs the error, marks Gemini unhealthy if the error is
    retryable, and returns without yielding further tokens (the caller
    detects the short token count and triggers the fallback path).

    Args:
        question (str): User question in Tamil or English.
        context (str): Concatenated document text from vector search results.
        csv_context (str): Formatted CSV metadata rows from semantic search.
        user_content (str | None): Pre-built user prompt override. Defaults
            to ``None``.
        system_prompt (str | None): Custom system instruction override.
            Defaults to ``None``.
        disable_thinking (bool): If ``True``, suppresses chain-of-thought
            reasoning. Defaults to ``False``.
        context_doc_count (int): Number of distinct documents in ``context``.
            Defaults to ``0``.
        history (list | None): Validated conversation history for multi-turn
            mode, or ``None`` for single-turn. Defaults to ``None``.
        language (str): BCP-47 language code. Defaults to ``"ta"``.
        max_output_tokens (int): Hard token cap on the generated response.
            Defaults to ``4096``.

    Yields:
        str: Individual token strings from the Gemini streaming response.
    """
    try:
        client = _get_gemini_client()
        if system_prompt is None:
            system_prompt = _get_system_prompt(language)
        if user_content is None:
            user_content = _build_user_content(
                question,
                context,
                csv_context,
                context_doc_count=context_doc_count,
                language=language,
            )

        contents = (
            _build_multi_turn_contents(history, user_content)
            if history
            else user_content
        )

        stream = with_gemini_retry(
            client.models.generate_content_stream,
            model=GEMINI_MODEL,
            contents=contents,
            config=_gemini_generation_config(
                system_prompt,
                disable_thinking=disable_thinking,
                max_output_tokens=max_output_tokens,
            ),
        )
        for chunk in stream:
            token = chunk.text
            if token:
                yield token

    except Exception as e:
        logger.error(f"Gemini streaming generation failed: {e}", exc_info=True)
        if is_retryable_gemini(e):
            _mark_gemini_unhealthy(str(e))
        return


def generate_extractive_answer(facts: List[Dict], question: str) -> str:
    """Generate a keyword-extractive fallback answer from pre-selected fact sentences.

    Concatenates up to five fact sentences (each longer than 30 characters)
    after stripping internal markup tokens (``__…__``, ``பொன்னி களஞ்சியம்``)
    and collapsing whitespace. Used when the LLM is unavailable or returns an
    insufficient response.

    Args:
        facts (List[Dict]): List of fact dicts produced by
            :func:`extract_key_facts`, each containing at minimum a
            ``"sentence"`` key.
        question (str): User question (currently unused; reserved for
            future relevance scoring).

    Returns:
        str: A single string of joined sentences ending with a period, or an
        empty string if no qualifying sentences are found in ``facts``.
    """
    if not facts:
        return ""

    answer_sentences = []
    for fact in facts[:5]:
        sent = fact["sentence"].strip()
        sent = re.sub(r"__.*?__|பொன்னி களஞ்சியம்", "", sent)
        sent = re.sub(r"\s+", " ", sent).strip()
        if len(sent) > 30:
            answer_sentences.append(sent)

    if not answer_sentences:
        return ""

    answer = ". ".join(answer_sentences)
    if answer and answer[-1] not in ".!?।":
        answer += "."

    return answer


def validate_gemini_api():
    """Validate that the Gemini API key is configured and reachable at startup.

    Provides a fast-fail check on application startup. Delegates to
    :func:`check_gemini_health` to avoid duplicating the ping logic. Logs
    a warning (non-fatal) if the API key is absent or the health check fails,
    allowing the service to start in a degraded state with extractive
    fallback active.

    Returns:
        None
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
