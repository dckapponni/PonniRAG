"""
Mock Search Module for Ponni RAG System.
Provides sample Tamil literary data for testing without requiring Qdrant or AWS.
Enable by setting environment variable: USE_MOCK_DATA=true
"""

import os
import random
from typing import Dict, List, Any

# Check if mock mode is enabled
USE_MOCK_DATA = os.getenv("USE_MOCK_DATA", "false").lower() in ("true", "1", "yes")

# Sample Tamil literary mock data
MOCK_SOURCES = [
    {
        "doc_issue": "இதழ் 1",
        "volume": "தொகுதி 1 (1947)",
        "heading": "திராவிட இயக்கத்தின் தோற்றம்",
        "author": "அரு. பெரியண்ணன்",
        "content": "திராவிட இயக்கம் தமிழ்நாட்டின் சமூக, அரசியல் வரலாற்றில் மிக முக்கியமான இடத்தைப் பிடித்துள்ளது. சுயமரியாதை இயக்கத்தின் கொள்கைகளை அடிப்படையாகக் கொண்டு உருவான இவ்வியக்கம், சாதி ஒழிப்பு, பெண்கள் உரிமை, பகுத்தறிவு ஆகியவற்றை முன்னிறுத்தியது.",
    },
    {
        "doc_issue": "இதழ் 3",
        "volume": "தொகுதி 1 (1947)",
        "heading": "பாரதிதாசன் கவிதைகள்",
        "author": "பாரதிதாசன்",
        "content": "தமிழுக்கு முதல் சிறப்பு தமிழுக்கே தான் முதல் சிறப்பு என்று பாடிய பாவேந்தர் பாரதிதாசன், தமிழ் இலக்கிய உலகில் புரட்சிக் கவிஞராக விளங்கினார். அவரது கவிதைகள் சமூக மாற்றத்திற்கான கருவிகளாக அமைந்தன.",
    },
    {
        "doc_issue": "இதழ் 5",
        "volume": "தொகுதி 2 (1948)",
        "heading": "சுயமரியாதை இயக்கம்",
        "author": "ஈ.வெ.ரா. பெரியார்",
        "content": "சுயமரியாதை என்பது ஒவ்வொரு மனிதனுக்கும் உள்ள அடிப்படை உரிமை. சாதி, மத பேதங்களை ஒழித்து, அனைவரும் சமம் என்ற கொள்கையை நிலைநாட்ட வேண்டும். பகுத்தறிவே நமது வழிகாட்டி.",
    },
    {
        "doc_issue": "இதழ் 7",
        "volume": "தொகுதி 2 (1948)",
        "heading": "தமிழ் மொழியின் பெருமை",
        "author": "முருகு. சுப்பிரமணியம்",
        "content": "தமிழ் மொழி உலகின் மிகப் பழமையான மொழிகளில் ஒன்று. சங்க இலக்கியங்கள் முதல் இன்றைய நவீன இலக்கியம் வரை தமிழ் தொடர்ந்து வளர்ந்து வருகிறது. தமிழின் சிறப்பை உலகறியச் செய்வது நமது கடமை.",
    },
    {
        "doc_issue": "இதழ் 12",
        "volume": "தொகுதி 3 (1949)",
        "heading": "பெண்கள் விடுதலை",
        "author": "அண்ணா துரை",
        "content": "பெண்கள் சமூகத்தின் அரை பாகம். அவர்களுக்கு கல்வி, வேலைவாய்ப்பு, சமூக உரிமைகள் அனைத்தும் சமமாக வழங்கப்பட வேண்டும். பெண் விடுதலை இல்லாமல் சமூக முன்னேற்றம் சாத்தியமில்லை.",
    },
    {
        "doc_issue": "இதழ் 15",
        "volume": "தொகுதி 4 (1950)",
        "heading": "இந்தி எதிர்ப்பு போராட்டம்",
        "author": "கா. அப்பாதுரையார்",
        "content": "இந்தி திணிப்புக்கு எதிராக தமிழ்நாடு போராடியது. தமிழ் மொழியின் அடையாளத்தைக் காப்பாற்ற வேண்டும் என்ற உணர்வு மக்களிடையே பரவியது. மொழி உரிமை என்பது அடிப்படை உரிமை.",
    },
    {
        "doc_issue": "இதழ் 18",
        "volume": "தொகுதி 5 (1951)",
        "heading": "கலை இலக்கியம்",
        "author": "கண்ணதாசன்",
        "content": "கலையும் இலக்கியமும் சமூகத்தின் கண்ணாடி. படைப்பாளிகள் தங்கள் எழுத்துக்கள் மூலம் சமூக மாற்றத்தை ஏற்படுத்த முடியும். பொன்னி இதழ் இதற்கு சிறந்த எடுத்துக்காட்டு.",
    },
    {
        "doc_issue": "இதழ் 21",
        "volume": "தொகுதி 6 (1952)",
        "heading": "விடுதலை தினம்",
        "author": "மு. கருணாநிதி",
        "content": "இந்திய விடுதலை நமக்கு சுதந்திரத்தை தந்தது. ஆனால் உண்மையான விடுதலை சமூக சமத்துவம் நிலைநாட்டப்படும் போதே கிடைக்கும். ஒடுக்கப்பட்ட மக்களின் உரிமைகளுக்காக போராட வேண்டும்.",
    },
]

MOCK_ANSWERS = {
    "default": "பொன்னி இதழ் 1947 முதல் 1955 வரை வெளியான ஒரு முக்கியமான திராவிட கலை இலக்கிய இதழ் ஆகும். இது திரு. அரு. பெரியண்ணன் மற்றும் திரு. முருகு. சுப்பிரமணியம் ஆகியோரால் தொடங்கப்பட்டது.",
    "authors": "பொன்னி இதழில் பல முக்கிய எழுத்தாளர்கள் எழுதியுள்ளனர்: தந்தை பெரியார், பேரறிஞர் அண்ணா, பாவேந்தர் பாரதிதாசன், கவியரசு கண்ணதாசன், கலைஞர் மு. கருணாநிதி, திரு.வி.க., கா. அப்பாதுரையார், மு. வ., கவிஞர் வாணிதாசன், கவிஞர் சுரதா போன்றோர் குறிப்பிடத்தக்கவர்கள்.",
    "bharathidasan": "பாரதிதாசன் பொன்னி இதழில் பல கவிதைகள் மற்றும் கட்டுரைகள் எழுதியுள்ளார். அவரது 'குயில்' இதழ் தடை செய்யப்பட்ட பிறகு பொன்னியில் எழுதத் தொடங்கினார். பாரதிதாசன் பரம்பரை கவிஞர்கள் என்ற பெயரில் அவரைப் பின்பற்றி எழுதியவர்களை பொன்னி அறிமுகப்படுத்தியது.",
    "dravidian": "பொன்னி இதழ் திராவிட இயக்கத்திற்கு முக்கியமான பங்களிப்பை செய்தது. பகுத்தறிவு, சுயமரியாதை, சமத்துவம் ஆகிய கொள்கைகளை தீவிரமாக எடுத்துரைத்தது. சாதி மற்றும் மத மூடநம்பிக்கைகளுக்கு எதிரான கருத்துகளை வலுவாக முன்வைத்தது.",
    "founder": "பொன்னி இதழ் திரு. அரு. பெரியண்ணன் மற்றும் திரு. முருகு. சுப்பிரமணியம் ஆகியோரால் 1947ஆம் ஆண்டு பிப்ரவரி மாதம் தொடங்கப்பட்டது. இவர்கள் இருவரும் திராவிட கருத்தியலை பரப்புவதில் முக்கிய பங்காற்றினர்.",
}


def get_mock_answer(question: str) -> str:
    """Generate appropriate mock answer based on question keywords."""
    question_lower = question.lower()

    # Tamil keywords
    if any(word in question for word in ["ஆசிரியர்", "எழுத்தாளர்", "யார்"]):
        return MOCK_ANSWERS["authors"]
    if any(word in question for word in ["பாரதிதாசன்", "பாவேந்தர்"]):
        return MOCK_ANSWERS["bharathidasan"]
    if any(word in question for word in ["திராவிட", "இயக்கம்"]):
        return MOCK_ANSWERS["dravidian"]
    if any(word in question for word in ["நிறுவனர்", "தொடங்கிய", "founder"]):
        return MOCK_ANSWERS["founder"]

    # English keywords
    if any(word in question_lower for word in ["author", "writer", "who"]):
        return MOCK_ANSWERS["authors"]
    if "bharathidasan" in question_lower:
        return MOCK_ANSWERS["bharathidasan"]
    if any(word in question_lower for word in ["dravidian", "movement"]):
        return MOCK_ANSWERS["dravidian"]
    if any(word in question_lower for word in ["founder", "started", "began"]):
        return MOCK_ANSWERS["founder"]

    return MOCK_ANSWERS["default"]


def get_mock_sources(num_sources: int = 3) -> List[Dict[str, Any]]:
    """Return random mock sources."""
    return random.sample(MOCK_SOURCES, min(num_sources, len(MOCK_SOURCES)))


def mock_ask_question(
    question: str,
    top_k: int = 10,
    return_formatted: bool = True,
    use_llm: bool = True
) -> Dict[str, Any]:
    """
    Mock implementation of ask_question for testing.

    Args:
        question: The user's question
        top_k: Number of sources to return
        return_formatted: Whether to return formatted output
        use_llm: Whether to simulate LLM usage

    Returns:
        Dictionary with 'answer' and 'sources' keys
    """
    answer = get_mock_answer(question)
    sources = get_mock_sources(min(top_k, 5))

    return {
        "answer": answer,
        "sources": sources,
        "query_type": "mock",
    }


def check_mock_health() -> Dict[str, Any]:
    """Return mock health status."""
    return {
        "healthy": True,
        "mock_mode": True,
        "message": "Running in mock mode - no database connection required",
        "points_count": len(MOCK_SOURCES),
    }
