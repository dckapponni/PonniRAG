import time
import streamlit as st
import sys
import io
import base64
from pathlib import Path
from PIL import Image

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

BASE_DIR = Path(__file__).resolve().parent
QDRANT_PATH = str(BASE_DIR / "qdrant_data")
IMG_DIR = BASE_DIR.parent / "img"  # Path to img folder

# Import the hybrid search function
from hybrid_search import ask_question

# PDF mapping - converts Google Drive sharing links to direct download links
PDF_LINKS = {
    # Volume 1 - 7 links
    "vol_1_issue_1": "https://drive.google.com/file/d/15_pcwcwltj8OCPavNp4LNCp1MN1jFLGq/view?usp=drive_link",
    "vol_1_issue_2": "https://drive.google.com/file/d/1QmJ6KV_oI8Z2gcpXhv_T4_q8kR67VYIU/view?usp=drive_link",
    "vol_1_issue_3": "https://drive.google.com/file/d/1CusAeyXs31YQkRVhVsIb6_4CNcrIYaoG/view?usp=drive_link",
    "vol_1_issue_4": "https://drive.google.com/file/d/1_j2uleRMjxvWGWzeDcbLzevPswT_HcEi/view?usp=drive_link",
    "vol_1_issue_5": "https://drive.google.com/file/d/1SigjEvoy12AyRpgkzgJ6sXbPHlfiw5w5/view?usp=drive_link",
    "vol_1_issue_6": "https://drive.google.com/file/d/1QZEcO900blzW1dGEOMA-I17iw46J5m0B/view?usp=drive_link",
    "vol_1_issue_7": "https://drive.google.com/file/d/1JPxexx7LdAOQSwbXfYOlUJ8W1uM5nw45/view?usp=drive_link",
    # Volume 2 - 19 links
    "vol_2_issue_1": "https://drive.google.com/file/d/1z38az6xTDWe95FSRyEO63-qKXACoQp1A/view?usp=drive_link",
    "vol_2_issue_2": "https://drive.google.com/file/d/18MyHva-XJ1mBm_oeEv8Ra4Xw_FwxfOTq/view?usp=drive_link",
    "vol_2_issue_3": "https://drive.google.com/file/d/1hI2u0AtI_J-yHu6HyobANSeb_q-GWnRU/view?usp=drive_link",
    "vol_2_issue_4": "https://drive.google.com/file/d/12BaNT0KhkNv4BIWx_PufXy3bKglucTYn/view?usp=drive_link",
    "vol_2_issue_5": "https://drive.google.com/file/d/138jLRANTCcaUvYo2lURk9527u-wDumQl/view?usp=drive_link",
    "vol_2_issue_6": "https://drive.google.com/file/d/1fakJXwpPheRV-x4DOlKEYhnz2eZ8FYML/view?usp=drive_link",
    "vol_2_issue_7": "https://drive.google.com/file/d/14WpSLrcqWqRXB9sQIvmRlWS2ffPKyerC/view?usp=drive_link",
    "vol_2_issue_8": "https://drive.google.com/file/d/1iS-DxhUWzhwFdQ2N8uXbQNSe4g6ERw9f/view?usp=drive_link",
    "vol_2_issue_9": "https://drive.google.com/file/d/19oYzhpJ3UStw0tuXNZGTe0jz15CR91g7/view?usp=drive_link",
    "vol_2_issue_10": "https://drive.google.com/file/d/1npT20F1SFtSohbADJ5KjMftobmk_8IE1/view?usp=drive_link",
    "vol_2_issue_11": "https://drive.google.com/file/d/1pyYijl87y4l2psVSDObNhRtldygVZTmj/view?usp=drive_link",
    "vol_2_issue_12": "https://drive.google.com/file/d/11bp1LhKuIo2bZ0jRA9tCuC-0MxJSBkpl/view?usp=drive_link",
    "vol_2_issue_13": "https://drive.google.com/file/d/1JIQ5hAfIwUZC4gtJ7pQwhGqbc22oB6Wd/view?usp=drive_link",
    "vol_2_issue_14": "https://drive.google.com/file/d/1fxE00dRfSpCatbfCTKYhayXGy0wrJPxA/view?usp=drive_link",
    "vol_2_issue_15": "https://drive.google.com/file/d/1Qff94UM9e5hf1BuOiKysss8mlEdh3b5R/view?usp=drive_link",
    "vol_2_issue_16": "https://drive.google.com/file/d/1dS2yKwsBnGks3T1Th6I-bWeDfMCCl9QY/view?usp=drive_link",
    "vol_2_issue_17": "https://drive.google.com/file/d/1TaZiHaUIIipsMhoD8r0feYlrWVX-yX2I/view?usp=drive_link",
    "vol_2_issue_18": "https://drive.google.com/file/d/1gqVxJq6y6quCoF7bmCz1dosdr413CU_6/view?usp=drive_link",
    "vol_2_issue_19": "https://drive.google.com/file/d/1cGxDRnviqN-Mnjh3oFiHtHU1Mzs3YMPa/view?usp=drive_link",
    
    # Volume 3 - 23 links
    "vol_3_issue_1": "https://drive.google.com/file/d/1bw0RfNl2wdNBxwO7R7kRdqr6PPqDqBcD/view?usp=drive_link",
    "vol_3_issue_2": "https://drive.google.com/file/d/1C52AW_nIYTpKClUekTDdoTZ4-12y5ekR/view?usp=drive_link",
    "vol_3_issue_3": "https://drive.google.com/file/d/1cfedNXaNjHJp5BRL4Xsh-2g1XlwtWSbX/view?usp=drive_link",
    "vol_3_issue_4": "https://drive.google.com/file/d/1DthckWeiaw4mQRiek64xce_OZS-NI1XQ/view?usp=drive_link",
    "vol_3_issue_5": "https://drive.google.com/file/d/1DthckWeiaw4mQRiek64xce_OZS-NI1XQ/view?usp=drive_link",
    "vol_3_issue_6": "https://drive.google.com/file/d/1IHdvdQYxSvjdf8X3iQEm_2f0zVQNiIbd/view?usp=drive_link",
    "vol_3_issue_7": "https://drive.google.com/file/d/1igYV0_jt42tl8-Fz_KFMtDGEtd-tRmQF/view?usp=drive_link",
    "vol_3_issue_8": "https://drive.google.com/file/d/1Huk1ju6L6phhEWyAVMSk_VDM9RXO98eB/view?usp=drive_link",
    "vol_3_issue_9": "https://drive.google.com/file/d/1zHTGqBVPx67YFayq9yZTHUdzSoA0ckya/view?usp=drive_link",
    "vol_3_issue_10": "https://drive.google.com/file/d/1q18M2VHpWVFtZ2cTOkS5PiqiwtEatnjD/view?usp=drive_link",
    "vol_3_issue_11": "https://drive.google.com/file/d/1X_F-B058RfduC0OHtmekinaMx-rWrMLH/view?usp=drive_link",
    "vol_3_issue_12": "https://drive.google.com/file/d/1HlJ9Teqhsq7b8JYoxVHrld4kV_oeERY4/view?usp=drive_link",
    "vol_3_issue_13": "https://drive.google.com/file/d/1cX8Kn_T5bHH4zsVZNB9azRZ2WPScv8mP/view?usp=drive_link",
    "vol_3_issue_14": "https://drive.google.com/file/d/1TjNf5OZ92TrCYb8u_GxU2VN4XF7_L4I1/view?usp=drive_link",
    "vol_3_issue_15": "https://drive.google.com/file/d/1wXHq8A_DdFqSFoNMBKp1G_jWcNJwvuH5/view?usp=drive_link",
    "vol_3_issue_16": "https://drive.google.com/file/d/1FE0mSIeG6jkuiVbBzeL7Od133PKH0v_E/view?usp=drive_link",
    "vol_3_issue_17": "https://drive.google.com/file/d/1dUurYWv-L1CpoYDGOoyvtwhCPiyEi3Xx/view?usp=drive_link",
    "vol_3_issue_18": "https://drive.google.com/file/d/1bD9SKZ4oKeAXIh4q2cdHvC4yCqPgkoFz/view?usp=drive_link",
    "vol_3_issue_19": "https://drive.google.com/file/d/1NWBUsxs8-9X8VcKazuXNz8OPAq7ficxK/view?usp=drive_link",
    "vol_3_issue_20": "https://drive.google.com/file/d/1snIy-Rl3U6SMdFVn9nFdIWzHJWxHI1zf/view?usp=drive_link",
    "vol_3_issue_21": "https://drive.google.com/file/d/1ycey9n4-ymwM4q1SY3nKssDLQxsQNQAq/view?usp=drive_link",

    # Volume 4 - 9 links
    "vol_4_issue_1": "https://drive.google.com/file/d/1dy9bemThpceYmuk0ou9FnNcefG-yZuzT/view?usp=drive_link",
    "vol_4_issue_2": "https://drive.google.com/file/d/1-Df9c2zy7ExmWNeMKS2cfw8NXvA37lvm/view?usp=drive_link",
    "vol_4_issue_3": "https://drive.google.com/file/d/171EKzUoUZVq6Kmwzr2AG4MWRH57m1Kpo/view?usp=drive_link",
    "vol_4_issue_4": "https://drive.google.com/file/d/1TLFQ6sCDEkqpVYKYOeDq283JmialFhuM/view?usp=drive_link",
    "vol_4_issue_5": "https://drive.google.com/file/d/1oinlAzKQBdnzaN7VJsUdiX7eBkiBcU_q/view?usp=drive_link",
    "vol_4_issue_6": "https://drive.google.com/file/d/1Kyv0RWQQuNqkUnu7fvTEz9VNAPhojhW8/view?usp=drive_link",
    "vol_4_issue_7": "https://drive.google.com/file/d/1yIFgx7gRkY_7hXxxvQ4CQoE9l0p2oKG1/view?usp=drive_link",
    "vol_4_issue_8": "https://drive.google.com/file/d/1dtMfMpiKY6aK69j0FeiaTAFcHEkqDazd/view?usp=drive_link",
    "vol_4_issue_9": "https://drive.google.com/file/d/1KWVF1hV9FXlWj_7uFXCvIVtmdXIf8IhD/view?usp=drive_link",
    
    # Volume 5 - 21 links
    "vol_5_issue_1": "https://drive.google.com/file/d/1ypRlyNFhr45pDQ7lARbaxWM3A4jJKiGb/view?usp=drive_link",
    "vol_5_issue_2": "https://drive.google.com/file/d/12xH7Z1wyaETQfg6ufdxP0b8Uc863tzSN/view?usp=drive_link",
    "vol_5_issue_3": "https://drive.google.com/file/d/1EkpyI-XnKGkWN6n0S-2yp-ELafigTkRv/view?usp=drive_link",
    "vol_5_issue_4": "https://drive.google.com/file/d/1cYjXfqJUjcpa2fQeedk2Wxd_X2elrMrr/view?usp=drive_link",
    "vol_5_issue_5": "https://drive.google.com/file/d/14pFIpJbIGEAb_suVBikOAKR-KmzSOwl1/view?usp=drive_link",
    "vol_5_issue_6": "https://drive.google.com/file/d/1ExQB12KbNF2KWUoVd-GuM7Jg7BwQcpQC/view?usp=drive_link",
    "vol_5_issue_7": "https://drive.google.com/file/d/1NEyXZTIasu4ZEnK5z35mkfd2dkH1sWwO/view?usp=drive_link",
    "vol_5_issue_8": "https://drive.google.com/file/d/1QVEKnwOV3ccXdbTHeKdpmutnSNUDwScg/view?usp=drive_link",
    "vol_5_issue_9": "https://drive.google.com/file/d/1aaJTys-tpKk1y23-9Xy5kVHqKJVx2lOU/view?usp=drive_link",
    "vol_5_issue_10": "https://drive.google.com/file/d/13wOEO_sHXeg71rX74p69w-_q3lP73ZbM/view?usp=drive_link",
    "vol_5_issue_11": "https://drive.google.com/file/d/1E55IVZSQl5DbEUEaKoU8bfwdvIAxMJ4W/view?usp=drive_link",
    "vol_5_issue_12": "https://drive.google.com/file/d/1ZqYx6ZtoqP_PJgX5MwI8fpxishGHV0LS/view?usp=drive_link",
    "vol_5_issue_13": "https://drive.google.com/file/d/1gIJcEqzUzyhc68VZpM2IdfH-m21q1odB/view?usp=drive_link",
    "vol_5_issue_14": "https://drive.google.com/file/d/1cJsVE6LTMSYrWgCikkjvkiLGENonF1S5/view?usp=drive_link",
    "vol_5_issue_15": "https://drive.google.com/file/d/1x9-IPCV3VCmmHEO3vOtExnVDLidOC5-Y/view?usp=drive_link",
    "vol_5_issue_16": "https://drive.google.com/file/d/1jmmm_x8ZR64y4nuVaLPiCPcyVqsK1BL0/view?usp=drive_link",
    "vol_5_issue_17": "https://drive.google.com/file/d/11p_wqB9iEbajDqUzd1sX-8HP4xWbHjwR/view?usp=drive_link",
    "vol_5_issue_18": "https://drive.google.com/file/d/1ZgITDHduzn_gnARE66H7Eb0Md4AN4D3x/view?usp=drive_link",
    "vol_5_issue_19": "https://drive.google.com/file/d/1lSn7IX0LothH7hfurFrOnbtC_FZgaOfd/view?usp=drive_link",
    "vol_5_issue_20": "https://drive.google.com/file/d/19vW2NFu9rTDBaiHLy_84aq7O9ns-DIJ5/view?usp=drive_link",
    "vol_5_issue_21": "https://drive.google.com/file/d/1RbrDRGgUe4FvPY11Vu8GDHILrmPq_pvQ/view?usp=drive_link",
    
    # Volume 6 - 22 links
    "vol_6_issue_1": "https://drive.google.com/file/d/1feI81Ml_IkbSUalZM2e_emuZI1WsZPJD/view?usp=drive_link",
    "vol_6_issue_2": "https://drive.google.com/file/d/1OLPI3LZXkihQXv4UIR1MGLqxf53fIZG7/view?usp=drive_link",
    "vol_6_issue_3": "https://drive.google.com/file/d/1jGpYWaoQiqkUBZAExAWfpuBk6Yy_bQ8b/view?usp=drive_link",
    "vol_6_issue_4": "https://drive.google.com/file/d/1fnOAxDssKmZBqj4zjnsmGHK8SGRM8zlq/view?usp=drive_link",
    "vol_6_issue_5": "https://drive.google.com/file/d/1JAFIF9iWVL8dJ-mQ4PlEmqLeVDrdh0xK/view?usp=drive_link",
    "vol_6_issue_6": "https://drive.google.com/file/d/1JvBVjz4CqDjMo3TDIlOcQnfLdJodkaBf/view?usp=drive_link",
    "vol_6_issue_7": "https://drive.google.com/file/d/1Rx3XdvmT58DJdNK-7cdOC0e6YmfAQ0Yj/view?usp=drive_link",
    "vol_6_issue_8": "https://drive.google.com/file/d/1CRFu92MF9dnFd2ZkPhhj0vMv11WTdFhc/view?usp=drive_link",
    "vol_6_issue_9": "https://drive.google.com/file/d/1yRyvm363G_Md1Dx0MndLOcmeGWRrymFe/view?usp=drive_link",
    "vol_6_issue_10": "https://drive.google.com/file/d/13RiJ1BSO1qeFQJmUdtviPK0cNh1u4TQ9/view?usp=drive_link",
    "vol_6_issue_11": "https://drive.google.com/file/d/1Vu2CV4fFBg3o3Mcj6Ikx95qiUdJF-FR_/view?usp=drive_link",
    "vol_6_issue_12": "https://drive.google.com/file/d/1R5ISLfscyVWPdHSNN2-7MdvZK1XenLuH/view?usp=drive_link",
    "vol_6_issue_13": "https://drive.google.com/file/d/12NuAqn6EZ_8eB-5B2cLI73K0Lg2KIjSh/view?usp=drive_link",
    "vol_6_issue_14": "https://drive.google.com/file/d/16Fu7Qm_IsDzXgMikYxvRGcuTfos6Qdf5/view?usp=drive_link",
    "vol_6_issue_15": "https://drive.google.com/file/d/19utkz8Ea1W3Jzzw2KOxZf_8zs0uklyrP/view?usp=drive_link",
    "vol_6_issue_16": "https://drive.google.com/file/d/1NADYyCCPBPO1tZTIchpIwUVizUTdD1JB/view?usp=drive_link",
    "vol_6_issue_17": "https://drive.google.com/file/d/1Ayh7Fm0ZxJKMhxH126IP9bx-E5k4aOJh/view?usp=drive_link",
    "vol_6_issue_18": "https://drive.google.com/file/d/1a96PJJyH4aTEYdRDbKSTfZZf1W1Za53F/view?usp=drive_link",
    
    # Volume 7 - 1 link
    "vol_7_issue_1": "https://drive.google.com/file/d/1jcOAE3bj_oUrFydLZM-wsDJxg9rqmn3H/view?usp=drive_link",
    
    # Volume 8 - 12 links
    "vol_8_issue_1": "https://drive.google.com/file/d/18TndZkat0MBilBdnp1GR_3z6zwFjBbnn/view?usp=drive_link",
    "vol_8_issue_2": "https://drive.google.com/file/d/118ZRbQ2w5cR0v5nTfmOx9AeSJTrc7my_/view?usp=drive_link",
    "vol_8_issue_3": "https://drive.google.com/file/d/10_0kXiZXXkiFCM-l6Ztu6A04GgJEv33k/view?usp=drive_link",
    "vol_8_issue_4": "https://drive.google.com/file/d/1ympAbQ3t2uUTtipD0pc-q3B_2iLv5kWd/view?usp=drive_link",
    "vol_8_issue_5": "https://drive.google.com/file/d/1GWx0-Hqjo41UYSYCugNnKB0TdmR4_Kqr/view?usp=drive_link",
    "vol_8_issue_6": "https://drive.google.com/file/d/1zpfBU-HYvo-hTEdji0jfvlydVO6agHA6/view?usp=drive_link",
    "vol_8_issue_7": "https://drive.google.com/file/d/16SwATZmphv3FcSx68eg_jD7MUblcJgFU/view?usp=drive_link",
    "vol_8_issue_8": "https://drive.google.com/file/d/1FIN4zCB0bXd_Ad5QK1SaTlEKKRaTejPD/view?usp=drive_link",
    "vol_8_issue_9": "https://drive.google.com/file/d/1F4ew3_uM1Ent6DJwJk4kn4aIfCy_kmRt/view?usp=drive_link",
    "vol_8_issue_10": "https://drive.google.com/file/d/1S-PkHiWzabHkySHf7Cw-vNVVUIfUbofa/view?usp=drive_link",

}

# -----------------------------------------------------------------------------
# Configuration & CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Ponni Archive",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# Localization & State Management
# -----------------------------------------------------------------------------
# Initialize Language
if "language" not in st.session_state:
    st.session_state.language = "ta"  # Default to Tamil

# Handle Query Parameters
query_params = st.query_params
current_page = query_params.get("page", "home")
selected_volume = query_params.get("volume", None)
selected_issue = query_params.get("issue", None)

# Check if language toggle was requested
if "lang" in query_params:
    st.session_state.language = query_params["lang"]
    # clear param to avoid sticking
    current_params = st.query_params.to_dict()
    current_params.pop("lang", None)
    st.query_params.clear()
    st.query_params.update(current_params)
    st.rerun()

lang = st.session_state.language


TRANSLATIONS = {
    "ta": {
        "app_title": "பொன்னி களஞ்சியம்",
        "nav_ask_ai": "AI-யிடம் கேளுங்கள்",
        "nav_library": "நூலகம்",
        "nav_about": "பற்றி",
        "nav_login": "உள்நுழை",
        "nav_toggle": "English",
        "hero_input_placeholder": "பொன்னி வரலாறு பற்றி கேளுங்கள்...",
        "sugg_founder": "பொன்னி இதழ் ஆசிரியர்கள்",
        "sugg_poets": "பாரதிதாசன் எழுதிய கட்டுரைகள்",
        "sugg_dravidian": "திராவிட இயக்கம்",
        "sugg_archive": "வேண்டாத ஆசை ஆசிரியர்",
        "lib_title": "பொன்னி மின் நூலகம்",
        "lib_desc": "1947–1955 வரையிலான அரிய தொகுப்புகளை ஆராயுங்கள்.",
        "lib_vol": "தொகுதி",
        "lib_back": "← நூலகத்திற்கு திரும்பு",
        "lib_back_issues": "← இதழ்களுக்கு திரும்பு",
        "sources_title": "ஆதாரங்கள்",
        "searching": "தேடுகிறது...",
        "document_type": "ஆவண வகை",
        "article": "கட்டுரை",
        "intro": "அறிமுகம்",
        "filter_author": "எழுத்தாளர்",
        "volume": "தொகுதி",
        "heading": "தலைப்பு",
        "read": "படிக்க",
        "issue": "இதழ்",
        "issue_label": "இதழ்",
        "malar_label": "மலர்",
        "title_label": "தலைப்பு",
        "author_label": "எழுத்தாளர்",
        "words": "வார்த்தைகள்",
        "read_more": "மேலும் படிக்க",
        "show_less": "குறைவாக காட்டு",
        "about_mission_text": '''திராவிட கருத்தியலைப் பட்டித்தொட்டி எங்கும் பரப்பும் முயற்சிக்குத் திராவிட கருத்தியலாளர்கள் பல்வேறு ஊடகங்களைப் கைக்கொண்டனர். அவற்றுள் இதழ்கள் குறிப்பிடத்தக்கன. குடியரசு, விடுதலை, திராவிடநாடு, திராவிடன், போர்வாள், தனியரசு, கிளர்ச்சி, குயில் போன்ற இதழ்கள் மிகப்பெரிய அளவில் அறிவு அரசியல் தளத்தில் தமிழ் மக்களிடையே பெரும் தாக்கத்தை ஏற்படுத்தின. இவ்விதழ்கள் பகுத்தறிவு, சுயமரியாதை, சமத்துவம் போன்ற கொள்கைகளை மக்களிடையே பரப்பியதுடன், சாதி, மத மூடநம்பிக்கைகளுக்கு எதிரான கருத்துகளை மிகக் காத்திரமாக முன்வைத்தன.

1900களில் வெளிவந்த இதழ்கள் சமூக மாற்றத்திற்கும் முன்னேற்றத்திற்கும் பெருந்துணையாக அமைந்துள்ளன என்பது வரலாற்று ரீதியான உண்மை. 1947 முதல் 1955 வரை இயங்கிய கலை இலக்கிய இதழ் 'பொன்னி'. பொன்னி இதழ் திரு. அரு. பெரியண்ணன் மற்றும் திரு. முருகு. சுப்பிரமணியம் ஆகியோரால் 1947ஆம் ஆண்டு பிப்ரவரி மாதம் தொடங்கப்பெற்றது. தொடங்கப்பட்ட முதல் வருடத்தில் மாதம் ஓர் இதழ் என வெளிவந்த பொன்னி 1948 முதல் மாதம் ஈரிதழாக வெளிவந்தது.

திராவிட இதழ்களின் வரிசையில் வைத்து போற்றத்தக்க பெரிதும் அறியப்படாத இதழாகப் பொன்னி இதழ் திகழ்கிறது. பகுத்தறிவு, சுயமரியாதை, சமத்துவம் ஆகியவற்றை மிகத் தீவிரமாக எடுத்துரைக்கும் இதழாக இவ்விதழ் வெளிவந்தது. தமிழகத்தின் தலைசிறந்த எழுத்தாளர்களும் படைப்பாளர்களும் தம் சீரிய கருத்துகளை இவ்விதழின்வழி எடுத்துரைத்தனர். தமிழ்ச் சமூகத்தை அறிவுச் சமூகமாக்கும் முன்னெடுப்பில் பொன்னி இதழின் பணி தலையாயதாகும்.

திராவிடக் கருத்தியலை துப்பாக்கியாகச் செயல்பட்ட திரு. அரு. பெரியண்ணன் அவர்களும், உள்வாங்கி இரட்டைக்குழல் திரு. முருகு. சுப்பிரமணியம் அவர்களும் இணைந்து 1947ஆம் ஆண்டு பிப்ரவரி மாதம் பொன்னி இதழைத் தொடங்கினர். பொன்னி இதழ் வண்ண அட்டைப்படத்தில் மிக நேர்த்தியாக வடிவமைக்கப்பட்டு வெளியிடப்பெற்றது. கவிஞர் கண்ணதாசன் தன் வனவாசம் புத்தகத்தில் 'திராவிடர் கழகத்தை ஆதரிக்கும் ஏடுகள் மிகக் குறைவாக இருந்த காலம். அவையும் அழகில்லாமல், அச்சுப்பிழை மிகுந்து வெளிவந்தன. அந்த நேரத்தில் வண்ண முகப்பு அட்டை போட்டு அழகாக நடந்த இதழ் 'பொன்னி' தான். பத்திரிக்கை துறையில் மற்றவர்கள் செய்துகாட்டாத புதுமை எல்லாம் அவர்கள் செய்து காட்டினார்கள். இன்றும் தமிழகத்தில் சிலரை அச்சுக்கலை நிபுணர்கள் என்று தேர்ந்தெடுத்தால், அவர்களில் பெரியண்ணன் மிக முக்கியமானவராக இருப்பார்' என்று குறிப்பிட்டுள்ளார். 

தமிழ் இலக்கிய உலகில் முக்கியமான கவிஞர் பாரதிதாசன் அவரின் 'குயில்' இதழ் அரசால் தடை செய்யப்பட்ட பிறகு பொன்னியில் எழுதினார். அவரின் கொள்கைகளையும் நடையையும் பின்பற்றி எழுதியவர்களை 'பாரதிதாசன் பரம்பரை கவிஞர்கள்' என்று அறிமுகப்படுத்தியது பொன்னி இதழ். 

இவ்விதழில் மாநில சுயாட்சி, இந்தித் திணிப்பு, தனித்தமிழ் பற்று, விடுதலை, அரசியல், சமூகம் சார்ந்த திராவிடச் சிந்தனைகள் கட்டுரைகளாக, கதைகளாக, கவிதைகளாக. கிறனாய்வுகளாக, துணுக்குகளாக வெளிவந்தன.

தந்தை பெரியார், பேரறிஞர் அண்ணா, பாவேந்தர், திரு.வி.க., கலைஞர் மு. கருணாநிதி, கா. அப்பாதுரையார், கவியரசு கண்ணதாசன், டி. கே. சீனிவாசன், மு. வ., மு. அண்ணாமலை, கவிஞர் வாணிதாசன், கவிஞர் சுரதா போன்ற பல முதன்மையான இலக்கிய, அரசியல் ஆளுமைகள் பொன்னியில் எழுதியுள்ளனர்.

கவிதைகள், சிறுகதைகள், தொடர்கதைகள், நொடிக் கதைகள், நாடகங்கள், பொதுக் கட்டுரைகள், ஆய்வுக் கட்டுரைகள், ஒப்பாய்வுக் கட்டுரைகள், தொடர் கட்டுரைகள், செய்திப் பாட்டு போன்ற இலக்கிய வகைமைகளில் பொன்னியில் படைப்புகள் வெளியாகியுள்ளன. இது மட்டுமன்றி அட்டைப்படக் குறிப்பு, மகளிர் அழகுக் குறிப்புகள், குழந்தை வளர்ப்புமுறை, பொன்னி வாழ்த்துகள், விகடங்கள், சிறுவர் அரங்கம் (சிறுவர் இலக்கியம்) போன்ற படைப்புகளும் இடம்பெற்றுள்ளன.

1948ல் போராட்டச் செய்தி நாட்குறிப்பு என்னும் தலைப்பில் கா. அப்பாதுரையார் அவர்கள், அக்கால விடுதலைப் போராட்ட நிலவரங்களை பதிவு செய்துள்ளார். எங்கே, யார் எதற்காக கைது செய்யப் படுகிறார்கள்? அவர்களுக்கு என்ன தண்டனை? போன்றவற்றை இப்பகுதியில் காணமுடிகிறது. தமிழ் இலக்கியம் மட்டுமன்றி சீனம், யார் எதற்காகக் கைது செய்யப் உலக இலக்கியங்களையும் பொன்னியில் அறிமுகம் செய்துள்ளனர். பாரசீகம், ரஷ்ய, கன்னடம், உருது, வடமொழி, தெலுங்கு போன்ற உலக இலக்கியங்களையும் பொன்னியில் அறிமுகம் செய்துள்ளனர்.

நாடக விளம்பரங்கள், புத்தக விளம்பரங்கள், திரைப்பட விளம்பரங்கள், வணிக விளம்பரங்கள் போன்றவை பொன்னி இதழில் இடம் பெற்றுள்ளன. கலையுலகம் என்ற பகுதியின் கீழ் திரைப்படங்கள், நாடகங்களின் விமர்சனங்களை எழுதியுள்ளனர். பொன்னியில் மேலும் ஒரு சிறப்பிற்குரிய விஷயம் அதில் இடம்பெற்றுள்ள படங்கள் மற்றும் ஓவியங்கள். படைப்பின் தலைப்புகளை வரைந்து இதழில் சேர்த்துள்ளனர். புதுமைப்பித்தன் நினைவுகளைப் பற்றி அவரது மனைவி கமலா அவர்கள் பொன்னி இதழில் எழுதியுள்ளார். பொன்னி இதழ் தொடங்கப்பெற்ற காலத்திலிருந்து இந்தி எதிர்ப்பு குறித்தான எழுத்துகள் தொடர்ந்து காத்திரமாக இடம்பெற்றுள்ளது. அறிஞர்களும் மக்களும் இதில் எழுதியுள்ளனர். பொன்னி இதழ் விடுதலை போராட்ட காலகட்டத்தில் வெளியான இதழ் என்பதால், அக்கால அரசியல் சூழ்நிலைகள் மற்றும் சமூக நிலைகள் படைப்புகளில் பிரதிபலிக்கின்றன. 

மார்க்சியம், பெண்ணியம் போன்ற இசங்களும் பொன்னி இதழில் இடம்பெற்றுள்ளன. இன்றைய தமிழ்நாடு 1947இல் மதராஸ் மாகாணமாக இருந்தது. 1950ல் அது மெட்ராஸ் மாநிலமாக மாறியது. இது தொடர்பான கட்டுரைகள் பொன்னியில் இடம்பெற்றுள்ளன. பொன்னியில் பார்ப்பனியத்திற்கு எதிரான கருத்துகளும் திராவிடத்தை ஆதரிக்கும் கருத்துகளும் வலுவாகத் தொடர்ந்து இடம்பெற்று வந்திருக்கின்றன. மாநில சுயாட்சி, தனித்தமிழ் போன்றவற்றைக் குறித்தும் பொன்னியில் எழுதப்பட்டுள்ளன. ஓர் இலக்கியம் கருத்துடன் சேர்ந்து காலத்திற்கு ஏற்ப அமைந்தால் மட்டுமே அது நிலைத்து நிற்கும். பொன்னியில் இடம் பெற்றுள்ள படைப்புகளும் அக்காலகட்ட சூழலுக்கு ஏற்ப அமைந்திருக்கின்றன. பொன்னி இதழ் ஒரு கலை இலக்கிய இதழாக மட்டுமின்றி புரட்சி இதழாகவே இருந்திருக்கிறது.

1947 முதல் 1955 வரையிலான தமிழகத்தின் காலக் கண்ணாடியாகப் பொன்னி இதழ் விளங்குகிறது. தொடக்க காலத் திராவிடக் கருத்தியல்களையும், அவை பரப்பப்பெற்ற வடிவங்களையும் முறைகளையும் ஆயும் ஆய்வாளர்களுக்கு மிகச் சிறந்த களமாகப் பொன்னி இதழ்கள் அமையும். 1947க்கு பிறகான எழுத்துருக்கள், சிந்தனைகள், உரிமை முழக்கங்கள், கேலிச் சித்திரங்கள், நூலறிமுகங்கள் போன்றவற்றை அறியவும், அவற்றை ஆய்வுக்குட்படுத்தவும் பெரும் வாய்ப்பை பொன்னி இதழ்கள் ஏற்படுத்திக் கொடுக்கும்.



''',
        "footer_founded": "நிறுவப்பட்டது: 1947",
        "footer_founder": "A.R. Periyannan & Murugu Subramanium"
    },
    "en": {
        "app_title": "Ponni Archive",
        "nav_ask_ai": "Ask AI",
        "nav_library": "Library",
        "nav_about": "About",
        "nav_login": "Log In",
        "nav_toggle": "தமிழ்",
        "hero_input_placeholder": "Ask about Ponni history...",
        "sugg_founder": "Writers of Ponni Magazine",
        "sugg_poets": "Articles by Bharathidasan",
        "sugg_dravidian": "Dravidian Movement",
        "sugg_archive": "The Unwanted Desire Teacher",
        "lib_title": "Ponni Digital Library",
        "lib_desc": "Browse restored volumes from 1947–1955.",
        "lib_vol": "Volume",
        "lib_back": "← Back to Library",
        "lib_back_issues": "← Back to Issues",
        "sources_title": "Sources",
        "searching": "Searching...",
        "document_type": "Document Type",
        "article": "Article",
        "intro": "Introduction",
        "filter_author": "Author",
        "volume": "Volume",
        "heading": "Heading",
        "read": "Read",
        "issue": "Issue",
        "issue_label": "Issue",
        "malar_label": "Malar",
        "title_label": "Title",
        "author_label": "Author",
        "words": "words",
        "read_more": "Read More",
        "show_less": "Show Less",
        "about_mission_text": """The Dravidian ideologies were propagated through various media by Dravidian ideologists. Among them, magazines were significant. Magazines like Kudiyarasu, Viduthalai, Dravidan, Porval, Thaniyarasu, Klerchi, and Kuyil had a profound impact on the Tamil populace in the realm of knowledge politics. These magazines disseminated ideas of rationalism, self-respect, and equality while boldly opposing caste and religious superstitions.""",
        "footer_founded": "Founded: 1947",
        "footer_founder": "A.R. Periyannan & Murugu Subramanium"
    }
}

def t(key):
    """Helper function to get translation"""
    return TRANSLATIONS[lang].get(key, key)

st.markdown(
    """
<style>
    /* 1. Global Reset & Fonts */
        
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Force white background on EVERYTHING */
    * {
        scrollbar-color: #cbd5e1 #f1f5f9;
    }

    html, body {
        background: #ffffff !important;
    }

    .stApp {
        background: #ffffff !important;
        font-family: 'Inter', sans-serif;
        color: #1e3a8a;
        min-height: 100vh;
    }

    /* Ensure all main containers have white background */
    section.main,
    section.main > div,
    section[data-testid="stMain"],
    div[data-testid="stAppViewContainer"],
    div[data-testid="stApp"],
    .main,
    .block-container {
        background: #ffffff !important;
    }

    /* Fix grey container backgrounds */
    div[data-testid="stHorizontalBlock"],
    div[data-testid="column"],
    div[data-testid="stVerticalBlock"] {
        background: #ffffff !important;
        background-color: #ffffff !important;
    }

    /* 2. Navigation Bar */
    .nav-container {
        display: flex;
        justify-content: flex-start;
        align-items: center;
        padding: 1rem 2rem;
        background: rgba(255,255,255,0.9);
        backdrop-filter: blur(10px);
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        z-index: 1000;
        border-bottom: 1px solid rgba(0,0,0,0.05);
    }

    .logo {
        font-weight: 800;
        font-size: 1.25rem;
        letter-spacing: -0.5px;
        color: #1e3a8a;
        text-decoration: none !important;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-right: 3rem;
    }

    .nav-links {
        display: flex;
        gap: 2rem;
        font-size: 0.95rem;
        font-weight: 500;
        align-items: center;
        flex-grow: 1;
    }

    .nav-link {
        color: #64748b;
        text-decoration: none !important;
        transition: color 0.2s;
    }
    .nav-link:hover {
        color: #1e3a8a;
        text-decoration: none !important;
    }

    .lang-toggle {
        font-size: 0.85rem;
        color: #64748b;
        text-decoration: none !important;
        border: 1px solid #e2e8f0;
        padding: 0.25rem 0.75rem;
        border-radius: 99px;
        margin-left: 1rem;
    }
    .lang-toggle:hover {
        background: #f1f5f9;
        color: #1e3a8a;
        text-decoration: none !important;
    }

    /* 3. Hero Section */
    .hero-container {
        text-align: center;
        padding-top: 8rem;
        padding-bottom: 3rem;
        max-width: 800px;
        margin: 0 auto;
    }

    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        background: -webkit-linear-gradient(45deg, #1e3a8a, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }

    .hero-subtitle {
        color: #64748b;
        font-size: 1.25rem;
        margin-bottom: 2rem;
    }

    /* 4. Chat Input Styling */
    [data-testid="stBottomBlockContainer"],
    [data-testid="stBottom"],
    section[data-testid="stBottom"],
    .stChatFloatingInputContainer,
    div[data-baseweb="base-input"] {
        background: #ffffff !important;
        background-color: #ffffff !important;
    }

    div[data-testid="stChatInput"],
    div[data-testid="stChatInput"] > div,
    div[data-testid="stChatInput"] > div > div {
        background: white !important;
        background-color: white !important;
    }

    div[data-testid="stChatInput"] {
        position: fixed;
        bottom: 2rem;
        left: 50%;
        transform: translateX(-50%);
        width: 100%;
        max-width: 800px !important;
        border-radius: 2rem !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1) !important;
        background: white !important;
        border: 1px solid #e2e8f0 !important;
        padding: 0.6rem 0.75rem !important;
        z-index: 900;
    }
    div[data-testid="stSpinner"],
    div[data-testid="stSpinner"] > div {
        background: white !important;
        background-color: white !important;
    }

    /* Fix any remaining gray backgrounds */
    .stChatFloatingInputContainer,
    section[data-testid="stChatFloatingInputContainer"] {
        background: white !important;
        background-color: white !important;
    }
    div[data-testid="stChatInput"] input,
    div[data-testid="stChatInput"] textarea {
        background: white !important;
        color: #1e293b !important;
    }

    div[data-testid="stChatInput"] input::placeholder,
    div[data-testid="stChatInput"] textarea::placeholder {
        color: #94a3b8 !important;
    }

    section[data-testid="stBottom"],
    section[data-testid="stBottom"] *,
    div[class*="bottom"],
    div[class*="Bottom"],
    div[class*="floating"],
    div[class*="Floating"] {
        background: white !important;
        background-color: white !important;
    }

    footer,
    .main footer,
    [data-testid="stStatusWidget"],
    [data-testid="stDecoration"] {
        background: white !important;
        background-color: white !important;
    }

    div[data-testid="stChatMessage"] p,
    div[data-testid="stChatMessage"] span,
    div[data-testid="stChatMessage"] div,
    div[data-testid="stChatMessage"] li {
        color: #1e293b !important;
    }

    div[data-testid="stMarkdownContainer"] {
        color: #1e293b !important;
    }

    /* 5. Message Styling */
    div[data-testid="stChatMessageContent"] {
        background-color: #ffffff;
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        color: #1e293b !important;
    }

    div[data-testid="stChatMessage"] div[data-testid="stChatMessageContent"] {
        background-color: #f8fafc;
        color: #1e293b !important;
    }

    div[data-testid="stChatMessage"].stChatMessage-user div[data-testid="stChatMessageContent"] {
        background-color: #ffffff;
        color: #1e293b !important;
    }

    /* 6. Source Cards */
    .source-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 0.75rem;
    }

    .source-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }

    .source-title {
        font-weight: 600;
        color: #1e3a8a;
        font-size: 0.95rem;
    }

    .source-meta {
        color: #64748b;
        font-size: 0.85rem;
        margin-bottom: 0.5rem;
    }

    .source-preview {
        color: #334155;
        font-size: 0.9rem;
        line-height: 1.5;
        font-style: italic;
        border-left: 3px solid #3b82f6;
        padding-left: 0.75rem;
        margin-top: 0.5rem;
    }

    /* 7. Hide elements */
    header[data-testid="stHeader"] { 
        display: none; 
    }

    div[data-testid="stSidebar"] { 
        display: none; 
    }

    /* 8. Button Styling */
    .stButton > button {
        background-color: white !important;
        color: #1e3a8a !important;
        border: none !important;
        border-radius: 0.75rem !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: all 0.2s !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
    }

    .stButton > button:hover {
        background-color: #f8fafc !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    }

    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* 9. PDF Viewer Styling */
    .pdf-container {
        width: 100%;
        height: 800px;
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-top: 1rem;
    }

    .pdf-container iframe {
        width: 100%;
        height: 100%;
        border: none;
    }

    /* 10. Issues Grid Styling - 4 per row */
    div[data-testid="column"] {
        background: #ffffff !important;
        padding: 1.5rem;
        border-radius: 0.75rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
    }

    div[data-testid="column"]:hover {
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
        transform: translateY(-4px);
    }

    div[data-testid="column"] h3 {
        color: #1e3a8a;
        font-size: 1.1rem;
        margin-bottom: 1rem;
        text-align: center;
        font-weight: 600;
    }

    /* Issue card styling */
    div[data-testid="column"] .stButton {
        width: 100%;
    }

    div[data-testid="column"] .stButton > button {
        width: 100%;
        background-color: #1e3a8a !important;
        color: white !important;
        border-radius: 0.5rem !important;
        padding: 0.75rem !important;
        font-weight: 600 !important;
        transition: all 0.2s !important;
        box-shadow: 0 2px 6px rgba(30, 58, 138, 0.2) !important;
    }

    div[data-testid="column"] .stButton > button:hover {
        background-color: #3b82f6 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(30, 58, 138, 0.3) !important;
    }
        
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Navbar
# -----------------------------------------------------------------------------
target_lang = "en" if lang == "ta" else "ta"
toggle_page_param = f"&page={current_page}" if current_page else ""

st.markdown(
    f"""
    <div class="nav-container">
        <a href="/?page=home" target="_self" class="logo">
            {t('app_title')}
        </a>
        <div class="nav-links">
            <a href="/?page=home" target="_self" class="nav-link">{t('nav_ask_ai')}</a>
            <a href="/?page=library" target="_self" class="nav-link">{t('nav_library')}</a>
            <a href="/?page=about" target="_self" class="nav-link">{t('nav_about')}</a>
            <a href="/?lang={target_lang}{toggle_page_param}" target="_self" class="lang-toggle">{t('nav_toggle')}</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)



def render_home():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    def handle_suggestion(prompt_text):
        st.session_state.temp_submit = prompt_text

    # ------------------ HERO SECTION ------------------
    if not st.session_state.messages:
        st.markdown(
            f"""
            <div class="hero-container">
                <h1 class="hero-title">{t('app_title')}</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_SpacerL, col_Main, col_SpacerR = st.columns([1, 2, 1])
        with col_Main:
            c1, c2 = st.columns(2)

            with c1:
                if st.button(f"📓 {t('sugg_founder')}", use_container_width=True):
                    # Fixed: Send Tamil question instead of English
                    if lang == "ta":
                        handle_suggestion("பொன்னி இதழில் எழுதிய ஆசிரியர்கள் யார்?")
                    else:
                        handle_suggestion("Who are the authors in ponni magazine?")
                    st.rerun()

                if st.button(f"✍️ {t('sugg_poets')}", use_container_width=True):
                    # Fixed: Send Tamil question instead of English
                    if lang == "ta":
                        handle_suggestion("பாரதிதாசன் எழுதிய கட்டுரைகள் பட்டியலிடுக")
                    else:
                        handle_suggestion("List the articles written by Bharathidasan")
                    st.rerun()

            with c2:
                if st.button(f"⚖️ {t('sugg_dravidian')}", use_container_width=True):
                    # Fixed: Send Tamil question instead of English
                    if lang == "ta":
                        handle_suggestion("பொன்னி திராவிட இயக்கத்திற்கு எவ்வாறு பங்களித்தது?")
                    else:
                        handle_suggestion("How did Ponni contribute to the Dravidian movement?")
                    st.rerun()

                if st.button(f"🏛️ {t('sugg_archive')}", use_container_width=True):
                    # Fixed: Send Tamil question instead of English
                    if lang == "ta":
                        handle_suggestion("வேண்டாத ஆசை ஆசிரியர் யார்?")
                    else:
                        handle_suggestion("Who is the founder of Ponni magazine?")
                    st.rerun()

    # ------------------ CHAT HISTORY ------------------
    else:
        st.markdown("<div style='height: 6rem;'></div>", unsafe_allow_html=True)

        for msg_idx, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

                if msg.get("sources"):
                    with st.expander(
                        f"{t('sources_title')} ({len(msg['sources'])} {t('document_type').lower()})"
                    ):
                        for idx, src in enumerate(msg["sources"], 1):
                            # ✅ FIX: Properly handle both ScoredPoint objects and dicts
                            if hasattr(src, 'payload') and src.payload:
                                # Qdrant ScoredPoint object
                                payload = src.payload
                                metadata = payload.get("metadata", {})
                                full_content = payload.get("content", "").strip()
                                
                                # Extract metadata fields safely
                                doc_issue = metadata.get("doc_issue")
                                volume = metadata.get("volume")
                                heading = metadata.get("heading")
                                author = metadata.get("author")  # Only for author documents
                            else:
                                # Dict format (from updated hybrid_search.py)
                                doc_issue = src.get("doc_issue")
                                volume = src.get("volume")
                                heading = src.get("heading")
                                author = src.get("author")
                                full_content = src.get("content", "").strip()
                            
                            # Build metadata parts with Tamil labels
                            meta_parts = []
                            
                            # Issue (இதழ்)
                            if doc_issue:
                                meta_parts.append(f"{t('issue_label')}: {doc_issue}")
                            
                            # Malar (மலர்)
                            if volume:
                                meta_parts.append(f"{t('malar_label')}: {volume}")
                            
                            # Title (தலைப்பு)
                            if heading:
                                meta_parts.append(f"{t('title_label')}: {heading}")
                            
                            # Author (எழுத்தாளர்) - Only if exists
                            if author:
                                meta_parts.append(f"{t('author_label')}: {author}")

                            meta_str = " • ".join(meta_parts)

                            read_more_key = f"read_more_{msg_idx}_{idx}"
                            if read_more_key not in st.session_state:
                                st.session_state[read_more_key] = False

                            if len(full_content) > 300:
                                preview_content = full_content[:300] + "..."

                                st.markdown(
                                    f"""
                                    <div class="source-card">
                                        <div class="source-header">
                                            <span class="source-title">📄 {t('sources_title')} {idx}</span>
                                        </div>
                                        <div class="source-meta">{meta_str}</div>
                                        <div class="source-preview">
                                            {full_content if st.session_state[read_more_key] else preview_content}
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                                button_label = t('show_less') if st.session_state[read_more_key] else t('read_more')

                                if st.button(
                                    button_label,
                                    key=f"btn_{read_more_key}",
                                ):
                                    st.session_state[read_more_key] = (
                                        not st.session_state[read_more_key]
                                    )
                                    st.rerun()
                            else:
                                st.markdown(
                                    f"""
                                    <div class="source-card">
                                        <div class="source-header">
                                            <span class="source-title">📄 {t('sources_title')} {idx}</span>
                                        </div>
                                        <div class="source-meta">{meta_str}</div>
                                        <div class="source-preview">{full_content}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

    # ------------------ INPUT HANDLING ------------------
    if "temp_submit" in st.session_state:
        user_input = st.session_state.temp_submit
        del st.session_state.temp_submit
    else:
        user_input = st.chat_input(t("hero_input_placeholder"))

    # ------------------ QUERY PROCESSING ------------------
    if user_input:
        # Add user message first
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )
        
        # Force rerun to show user message before spinner
        st.rerun()
        
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        with st.spinner(t("searching")):
            try:
                # Get the last user message
                last_user_msg = st.session_state.messages[-1]["content"]
                
                result = ask_question(last_user_msg) or {}

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result.get("answer", ""),
                        "sources": result.get("sources", []),
                    }
                )
            except Exception as e:
                error_msg = (
                    "மன்னிக்கவும், பிழை ஏற்பட்டது"
                    if lang == "ta"
                    else "Sorry, an error occurred"
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": f"{error_msg}: {str(e)}",
                        "sources": [],
                    }
                )

        st.rerun()

def render_library():
    st.markdown("<div style='height: 6rem;'></div>", unsafe_allow_html=True)
    st.markdown(f"## 📚 {t('lib_title')}")
    st.markdown(t('lib_desc'))
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Volume data with images
    volumes = [
        {"id": 1, "title": f"பொன்னி\n\n {t('lib_vol')} 1\n\n", "desc": "1947", "image": "Volume1.jpg"},
        {"id": 2, "title": f"பொன்னி\n\n {t('lib_vol')} 2\n\n", "desc": "1948", "image": "Volume2.jpg"},
        {"id": 3, "title": f"பொன்னி\n\n {t('lib_vol')} 3\n\n", "desc": "1949", "image": "Volume3.jpg"},
        {"id": 4, "title": f"பொன்னி\n\n {t('lib_vol')} 4\n\n", "desc": "1950", "image": "Volume4.jpg"},
        {"id": 5, "title": f"பொன்னி\n\n {t('lib_vol')} 5\n\n", "desc": "1951", "image": "Volume5.jpg"},
        {"id": 6, "title": f"பொன்னி\n\n {t('lib_vol')} 6\n\n", "desc": "1952", "image": "Volume6.jpg"},
        {"id": 7, "title": f"பொன்னி\n\n {t('lib_vol')} 7\n\n", "desc": "1953", "image": "Volume7.jpg"},
        {"id": 8, "title": f"பொன்னி\n\n {t('lib_vol')} 8\n\n", "desc": "1954", "image": "Volume8.jpg"},
    ]
    
    # Add custom CSS to make buttons look like text
    st.markdown("""
    <style>
    div[data-testid="stButton"] > button {
        background: transparent !important;
        border: none !important;
        padding: 0.5rem 0 !important;
        color: #1e3a8a !important;
        cursor: pointer !important;
        box-shadow: none !important;
        text-align: center !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }
    div[data-testid="stButton"] > button:hover {
        background: transparent !important;
        transform: none !important;
        box-shadow: none !important;
        opacity: 0.7 !important;
        border: none !important;
    }
    div[data-testid="stButton"] > button:active {
        transform: none !important;
        background: transparent !important;
        box-shadow: none !important;
    }
    div[data-testid="stButton"] > button:focus {
        background: transparent !important;
        box-shadow: none !important;
        border: none !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create 2 columns per row
    for i in range(0, len(volumes), 3):
        cols = st.columns(3, gap="medium")
        
        for j in range(3):
            if i + j < len(volumes):
                vol = volumes[i + j]
                col = cols[j]
                
                with col:
                    # Load and display image
                    img_path = IMG_DIR / vol["image"]
                    
                    if img_path.exists():
                        try:
                            img = Image.open(img_path)
                            if img.mode != "RGB":
                                img = img.convert("RGB")
                            
                            # Resize to smaller thumbnail
                            img.thumbnail((250, 375), Image.Resampling.LANCZOS)
                            
                            # Convert image to base64
                            buffered = io.BytesIO()
                            img.save(buffered, format="JPEG")
                            img_str = base64.b64encode(buffered.getvalue()).decode()
                            card_html = f"""
                            <a href="?page=issues&volume={vol['id']}" 
                            target="_self" 
                            style="text-decoration:none; display:block; width:100%;">

                            <div style="
                                background:#ffffff;
                                border:1px solid #e2e8f0;
                                border-radius:0.75rem;
                                padding:1rem;
                                text-align:center;
                                box-shadow:0 2px 8px rgba(0,0,0,0.08);
                                transition:all 0.3s ease;
                                cursor:pointer;
                                width:350px;             
                                margin:0 auto; 
                            ">

                            <img src="data:image/jpeg;base64,{img_str}"
                                style="max-width:100%; height:auto; border-radius:0.5rem; margin-bottom:0.8rem; pointer-events:auto;">

                            <div style="font-weight:600; font-size:1.1rem; color:#1e3a8a; line-height:1.4;">
                                பொன்னி<br>
                                {t('lib_vol')} {vol['id']}
                                <div style="margin-top:0.4rem;"></div>
                                <span style="font-weight:500; color:#64748b;">{vol['desc']}</span>
                            </div>

                            </div>
                            </a>
                            """
                            st.markdown(card_html, unsafe_allow_html=True)
                                
                        except Exception as e:
                            st.error(f"Error loading {vol['image']}: {e}")
                    else:
                        st.warning(f"Not found: {vol['image']}")
        
        # Add spacing between rows
        st.markdown("<br><br>", unsafe_allow_html=True)

def set_page(**params):
    """Safely update query params without losing language"""
    qp = dict(st.query_params)
    qp.update({k: v for k, v in params.items() if v is not None})
    st.query_params.clear()
    st.query_params.update(qp)

def render_issues(volume_id):
    st.markdown("<div style='height: 6rem;'></div>", unsafe_allow_html=True)

    # Back button (safe routing, no lang loss)
    if st.button(t("lib_back")):
        set_page(page="library")
        st.rerun()

    st.markdown(f"## 📑 {t('lib_vol')} {volume_id}")
    st.markdown("<br>", unsafe_allow_html=True)

    from pathlib import Path
    
    # Ensure IMG_DIR is a Path object
    if isinstance(IMG_DIR, str):
        base_dir = Path(IMG_DIR)
    else:
        base_dir = IMG_DIR
    
    # Volume folder name: "volume {volume_id} cover images"
    volume_folder = base_dir / f"volume {volume_id} cover images"
    
    # Get all images from the folder and sort them
    if volume_folder.exists():
        # Get all image files (jpg, png, jpeg)
        image_files = []
        for ext in ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG', '*.JPEG']:
            image_files.extend(volume_folder.glob(ext))
        
        # Sort by filename
        image_files.sort()
        
        # Create issues_data from available images
        issues_data = []
        issue_counter = 1
        
        for img_path in image_files:
            # Check if this issue has a PDF link
            key = f"vol_{volume_id}_issue_{issue_counter}"
            if key in PDF_LINKS:
                issues_data.append({
                    "issue_num": issue_counter,
                    "has_pdf": True,
                    "image_path": img_path
                })
            issue_counter += 1
    else:
        issues_data = []
        st.warning(f"Image folder not found: {volume_folder}")

    # Custom CSS for issue cards
    st.markdown("""
    <style>
    .issue-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 0.75rem;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
        cursor: pointer;
        height: 100%;
        text-decoration: none;
        display: block;
    }
    .issue-card:hover {
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
        transform: translateY(-4px);
        text-decoration: none;
    }
    .issue-card:active {
        transform: translateY(-2px);
    }
    .issue-card img {
        width: 100%;
        height: 280px;
        object-fit: cover;
    }
    .issue-card-title {
        padding: 1rem;
        text-align: center;
        color: #1e3a8a;
        font-weight: 600;
        font-size: 1.1rem;
        background: #f8fafc;
    }
    /* Hide the read button */
    div[data-testid="column"] .stButton {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Render 4 issues per row
    for i in range(0, len(issues_data), 4):
        cols = st.columns(4, gap="medium")
        
        for j in range(4):
            if i + j < len(issues_data):
                issue = issues_data[i + j]
                col = cols[j]
                
                with col:
                    # Load and display issue image
                    try:
                        img = Image.open(issue["image_path"])
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        
                        # Resize to consistent size
                        img.thumbnail((300, 400), Image.Resampling.LANCZOS)
                        
                        # Convert to base64
                        buffered = io.BytesIO()
                        img.save(buffered, format="JPEG", quality=85)
                        img_str = base64.b64encode(buffered.getvalue()).decode()
                        
                        # Create clickable card with proper link
                        card_html = f"""
                        <a href="?page=pdf_viewer&volume={volume_id}&issue={issue['issue_num']}" target="_self" class="issue-card" style="text-decoration: none;">
                            <img src="data:image/jpeg;base64,{img_str}" alt="{t('issue')} {issue['issue_num']}">
                            <div class="issue-card-title">{t('issue')} {issue['issue_num']}</div>
                        </a>
                        """
                        st.markdown(card_html, unsafe_allow_html=True)
                    except Exception as e:
                        # Skip this issue if image can't be loaded
                        st.error(f"Failed to load {issue['image_path']}: {e}")
        
        # Add spacing between rows
        st.markdown("<br>", unsafe_allow_html=True)

      
def render_pdf_viewer(volume_id, issue_num):
    
    st.markdown("<div style='height: 6rem;'></div>", unsafe_allow_html=True)

    # Back button (safe routing, no lang loss)
    if st.button(t("lib_back_issues")):
        set_page(page="issues", volume=volume_id)
        st.rerun()

    st.markdown(
        f"## 📖 {t('lib_vol')} {volume_id} - {t('issue')} {issue_num}"
    )

    pdf_key = f"vol_{volume_id}_issue_{issue_num}"
    pdf_url = PDF_LINKS.get(pdf_key)

    if not pdf_url:
        st.warning(
            f"PDF not available for {t('lib_vol')} {volume_id}, {t('issue')} {issue_num}"
        )
        return

    # ✅ Support BOTH Google Drive URL formats (NO CSS CHANGE)
    try:
        if "id=" in pdf_url:
            file_id = pdf_url.split("id=")[1].split("&")[0]
        elif "/d/" in pdf_url:
            file_id = pdf_url.split("/d/")[1].split("/")[0]
        else:
            st.error("Invalid PDF URL format")
            return
    except Exception:
        st.error("Invalid PDF URL format")
        return

    embed_url = f"https://drive.google.com/file/d/{file_id}/preview"

    # ✅ EXACT SAME HTML + CSS AS ORIGINAL
    st.markdown(
        f"""
        <div class="pdf-container">
            <iframe
                src="{embed_url}"
                width="100%"
                height="800px"
                allow="autoplay">
            </iframe>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"[Open PDF in new tab]({pdf_url})",
        unsafe_allow_html=True,
    )


def render_about():
    # Reduced top padding to 1rem
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    
    # Add custom CSS for about page
    st.markdown("""
    <style>
    .about-main-container {
        padding: 1.5rem;
        max-width: 100%;
    }
    .about-content-wrapper {
        max-width: 900px;
        margin: 0 auto;
        padding: 0 15%;
    }
    .about-heading {
        color: #1e3a8a;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 2rem;
        text-align: center;
        padding-bottom: 1rem;
        font-family: 'Inter', sans-serif;
    }
    .about-text {
        line-height: 1.8;
        text-align: justify;
        color: #1e293b;
        font-size: 0.95rem;
        margin-bottom: 2rem;
        font-family: 'Inter', sans-serif;
    }
    .about-single-image {
        display: flex;
        justify-content: center;
        margin: 0 0 2rem 0; 
    }
    .about-single-image img {
        max-width: 50% !important;  /* Increased from 20% to 30% */
        width: auto !important;
        height: auto !important;
        border-radius: 1rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
        display: block;
        margin: 0 auto;
    }
    .about-double-image {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin: 2rem 0;
    }
    .about-double-image > div {
        flex: 0 0 10% !important;
        max-width: 10% !important;
    }
    .about-double-image img {
        width: 100% !important;
        max-width: 100% !important;
        height: auto !important;
        border-radius: 1rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
    }
    /* Override Streamlit's default image styling */
    .stImage {
        max-width: 100% !important;
    }
    .stImage > img {
        max-width: 100% !important;
        width: auto !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main container
    st.markdown('<div class="about-main-container">', unsafe_allow_html=True)
    st.markdown('<div class="about-content-wrapper">', unsafe_allow_html=True)
    
    # Heading
    st.markdown('<h2 class="about-heading">பொன்னி களஞ்சியம்\n\n\n</h2>', unsafe_allow_html=True)
    
    # First text block
    st.markdown('''<div class="about-text">
    திராவிட கருத்தியலைப் பட்டித்தொட்டி எங்கும் பரப்பும் முயற்சிக்குத் திராவிட கருத்தியலாளர்கள் பல்வேறு ஊடகங்களைப் கைக்கொண்டனர். அவற்றுள் இதழ்கள் குறிப்பிடத்தக்கன. குடியரசு, விடுதலை, திராவிடநாடு, திராவிடன், போர்வாள், தனியரசு, கிளர்ச்சி, குயில் போன்ற இதழ்கள் மிகப்பெரிய அளவில் அறிவு அரசியல் தளத்தில் தமிழ் மக்களிடையே பெரும் தாக்கத்தை ஏற்படுத்தின. இவ்விதழ்கள் பகுத்தறிவு, சுயமரியாதை, சமத்துவம் போன்ற கொள்கைகளை மக்களிடையே பரப்பியதுடன், சாதி, மத மூடநம்பிக்கைகளுக்கு எதிரான கருத்துகளை மிகக் காத்திரமாக முன்வைத்தன.
    <br><br>
    1900களில் வெளிவந்த இதழ்கள் சமூக மாற்றத்திற்கும் முன்னேற்றத்திற்கும் பெருந்துணையாக அமைந்துள்ளன என்பது வரலாற்று ரீதியான உண்மை. 1947 முதல் 1955 வரை இயங்கிய கலை இலக்கிய இதழ் 'பொன்னி'. பொன்னி இதழ் திரு. அரு. பெரியண்ணன் மற்றும் திரு. முருகு. சுப்பிரமணியம் ஆகியோரால் 1947ஆம் ஆண்டு பிப்ரவரி மாதம் தொடங்கப்பெற்றது. தொடங்கப்பட்ட முதல் வருடத்தில் மாதம் ஓர் இதழ் என வெளிவந்த பொன்னி 1948 முதல் மாதம் ஈரிதழாக வெளிவந்தது.
    </div>''', unsafe_allow_html=True)
    
    st.markdown('<div class="about-single-image">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])  # Adjusted ratio for 30% width
    with col2:
        load_image("about1")
    st.markdown('</div>', unsafe_allow_html=True)
    # Second text block
    st.markdown('''<div class="about-text">
    திராவிட இதழ்களின் வரிசையில் வைத்து போற்றத்தக்க பெரிதும் அறியப்படாத இதழாகப் பொன்னி இதழ் திகழ்கிறது. பகுத்தறிவு, சுயமரியாதை, சமத்துவம் ஆகியவற்றை மிகத் தீவிரமாக எடுத்துரைக்கும் இதழாக இவ்விதழ் வெளிவந்தது. தமிழகத்தின் தலைசிறந்த எழுத்தாளர்களும் படைப்பாளர்களும் தம் சீரிய கருத்துகளை இவ்விதழின்வழி எடுத்துரைத்தனர். தமிழ்ச் சமூகத்தை அறிவுச் சமூகமாக்கும் முன்னெடுப்பில் பொன்னி இதழின் பணி தலையாயதாகும்.
    <br><br>
    திராவிடக் கருத்தியலை துப்பாக்கியாகச் செயல்பட்ட திரு. அரு. பெரியண்ணன் அவர்களும், உள்வாங்கி இரட்டைக்குழல் திரு. முருகு. சுப்பிரமணியம் அவர்களும் இணைந்து 1947ஆம் ஆண்டு பிப்ரவரி மாதம் பொன்னி இதழைத் தொடங்கினர். பொன்னி இதழ் வண்ண அட்டைப்படத்தில் மிக நேர்த்தியாக வடிவமைக்கப்பட்டு வெளியிடப்பெற்றது. கவிஞர் கண்ணதாசன் தன் வனவாசம் புத்தகத்தில் 'திராவிடர் கழகத்தை ஆதரிக்கும் ஏடுகள் மிகக் குறைவாக இருந்த காலம். அவையும் அழகில்லாமல், அச்சுப்பிழை மிகுந்து வெளிவந்தன. அந்த நேரத்தில் வண்ண முகப்பு அட்டை போட்டு அழகாக நடந்த இதழ் 'பொன்னி' தான். பத்திரிக்கை துறையில் மற்றவர்கள் செய்துகாட்டாத புதுமை எல்லாம் அவர்கள் செய்து காட்டினார்கள். இன்றும் தமிழகத்தில் சிலரை அச்சுக்கலை நிபுணர்கள் என்று தேர்ந்தெடுத்தால், அவர்களில் பெரியண்ணன் மிக முக்கியமானவராக இருப்பார்' என்று குறிப்பிட்டுள்ளார்.
    <br><br>
    தமிழ் இலக்கிய உலகில் முக்கியமான கவிஞர் பாரதிதாசன் அவரின் 'குயில்' இதழ் அரசால் தடை செய்யப்பட்ட பிறகு பொன்னியில் எழுதினார். அவரின் கொள்கைகளையும் நடையையும் பின்பற்றி எழுதியவர்களை 'பாரதிதாசன் பரம்பரை கவிஞர்கள்' என்று அறிமுகப்படுத்தியது பொன்னி இதழ்.
    </div>''', unsafe_allow_html=True)
    
    # Two images side by side (about2 and about3)
    st.markdown('<div class="about-double-image">', unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")
    with col1:
        load_image("about2")
    with col2:
        load_image("about3")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Third text block
    st.markdown('''<div class="about-text">
    இவ்விதழில் மாநில சுயாட்சி, இந்தித் திணிப்பு, தனித்தமிழ் பற்று, விடுதலை, அரசியல், சமூகம் சார்ந்த திராவிடச் சிந்தனைகள் கட்டுரைகளாக, கதைகளாக, கவிதைகளாக. கிறனாய்வுகளாக, துணுக்குகளாக வெளிவந்தன.
    <br><br>
    தந்தை பெரியார், பேரறிஞர் அண்ணா, பாவேந்தர், திரு.வி.க., கலைஞர் மு. கருணாநிதி, கா. அப்பாதுரையார், கவியரசு கண்ணதாசன், டி. கே. சீனிவாசன், மு. வ., மு. அண்ணாமலை, கவிஞர் வாணிதாசன், கவிஞர் சுரதா போன்ற பல முதன்மையான இலக்கிய, அரசியல் ஆளுமைகள் பொன்னியில் எழுதியுள்ளனர்.
    <br><br>
    கவிதைகள், சிறுகதைகள், தொடர்கதைகள், நொடிக் கதைகள், நாடகங்கள், பொதுக் கட்டுரைகள், ஆய்வுக் கட்டுரைகள், ஒப்பாய்வுக் கட்டுரைகள், தொடர் கட்டுரைகள், செய்திப் பாட்டு போன்ற இலக்கிய வகைமைகளில் பொன்னியில் படைப்புகள் வெளியாகியுள்ளன. இது மட்டுமன்றி அட்டைப்படக் குறிப்பு, மகளிர் அழகுக் குறிப்புகள், குழந்தை வளர்ப்புமுறை, பொன்னி வாழ்த்துகள், விகடங்கள், சிறுவர் அரங்கம் (சிறுவர் இலக்கியம்) போன்ற படைப்புகளும் இடம்பெற்றுள்ளன.
    </div>''', unsafe_allow_html=True)
    
    # Two more images side by side (about4 and about5)
    st.markdown('<div class="about-double-image">', unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")
    with col1:
        load_image("about4")
    with col2:
        load_image("about5")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Fourth text block
    st.markdown('''<div class="about-text">
    1948ல் போராட்டச் செய்தி நாட்குறிப்பு என்னும் தலைப்பில் கா. அப்பாதுரையார் அவர்கள், அக்கால விடுதலைப் போராட்ட நிலவரங்களை பதிவு செய்துள்ளார். எங்கே, யார் எதற்காக கைது செய்யப் படுகிறார்கள்? அவர்களுக்கு என்ன தண்டனை? போன்றவற்றை இப்பகுதியில் காணமுடிகிறது. தமிழ் இலக்கியம் மட்டுமன்றி சீனம், யார் எதற்காகக் கைது செய்யப் உலக இலக்கியங்களையும் பொன்னியில் அறிமுகம் செய்துள்ளனர். பாரசீகம், ரஷ்ய, கன்னடம், உருது, வடமொழி, தெலுங்கு போன்ற உலக இலக்கியங்களையும் பொன்னியில் அறிமுகம் செய்துள்ளனர்.
    <br><br>
    நாடக விளம்பரங்கள், புத்தக விளம்பரங்கள், திரைப்பட விளம்பரங்கள், வணிக விளம்பரங்கள் போன்றவை பொன்னி இதழில் இடம் பெற்றுள்ளன. கலையுலகம் என்ற பகுதியின் கீழ் திரைப்படங்கள், நாடகங்களின் விமர்சனங்களை எழுதியுள்ளனர். பொன்னியில் மேலும் ஒரு சிறப்பிற்குரிய விஷயம் அதில் இடம்பெற்றுள்ள படங்கள் மற்றும் ஓவியங்கள். படைப்பின் தலைப்புகளை வரைந்து இதழில் சேர்த்துள்ளனர். புதுமைப்பித்தன் நினைவுகளைப் பற்றி அவரது மனைவி கமலா அவர்கள் பொன்னி இதழில் எழுதியுள்ளார். பொன்னி இதழ் தொடங்கப்பெற்ற காலத்திலிருந்து இந்தி எதிர்ப்பு குறித்தான எழுத்துகள் தொடர்ந்து காத்திரமாக இடம்பெற்றுள்ளது. அறிஞர்களும் மக்களும் இதில் எழுதியுள்ளனர். பொன்னி இதழ் விடுதலை போராட்ட காலகட்டத்தில் வெளியான இதழ் என்பதால், அக்கால அரசியல் சூழ்நிலைகள் மற்றும் சமூக நிலைகள் படைப்புகளில் பிரதிபலிக்கின்றன.
    <br><br>
    மார்க்சியம், பெண்ணியம் போன்ற இசங்களும் பொன்னி இதழில் இடம்பெற்றுள்ளன. இன்றைய தமிழ்நாடு 1947இல் மதராஸ் மாகாணமாக இருந்தது. 1950ல் அது மெட்ராஸ் மாநிலமாக மாறியது. இது தொடர்பான கட்டுரைகள் பொன்னியில் இடம்பெற்றுள்ளன. பொன்னியில் பார்ப்பனியத்திற்கு எதிரான கருத்துகளும் திராவிடத்தை ஆதரிக்கும் கருத்துகளும் வலுவாகத் தொடர்ந்து இடம்பெற்று வந்திருக்கின்றன. மாநில சுயாட்சி, தனித்தமிழ் போன்றவற்றைக் குறித்தும் பொன்னியில் எழுதப்பட்டுள்ளன. ஓர் இலக்கியம் கருத்துடன் சேர்ந்து காலத்திற்கு ஏற்ப அமைந்தால் மட்டுமே அது நிலைத்து நிற்கும். பொன்னியில் இடம் பெற்றுள்ள படைப்புகளும் அக்காலகட்ட சூழலுக்கு ஏற்ப அமைந்திருக்கின்றன. பொன்னி இதழ் ஒரு கலை இலக்கிய இதழாக மட்டுமின்றி புரட்சி இதழாகவே இருந்திருக்கிறது.
    <br><br>
    1947 முதல் 1955 வரையிலான தமிழகத்தின் காலக் கண்ணாடியாகப் பொன்னி இதழ் விளங்குகிறது. தொடக்க காலத் திராவிடக் கருத்தியல்களையும், அவை பரப்பப்பெற்ற வடிவங்களையும் முறைகளையும் ஆயும் ஆய்வாளர்களுக்கு மிகச் சிறந்த களமாகப் பொன்னி இதழ்கள் அமையும். 1947க்கு பிறகான எழுத்துருக்கள், சிந்தனைகள், உரிமை முழக்கங்கள், கேலிச் சித்திரங்கள், நூலறிமுகங்கள் போன்றவற்றை அறியவும், அவற்றை ஆய்வுக்குட்படுத்தவும் பெரும் வாய்ப்பை பொன்னி இதழ்கள் ஏற்படுத்திக் கொடுக்கும்.
    </div>''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def load_image(image_name):
    """Helper function to load images with multiple extension attempts"""
    for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
        img_path = IMG_DIR / f"{image_name}{ext}"
        if img_path.exists():
            try:
                img = Image.open(img_path)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                st.image(img, use_container_width=False)
                return
            except:
                continue
# -------------------------------------------------------------------------
# Main Routing
# -------------------------------------------------------------------------
if current_page == "library":
    render_library()
elif current_page == "issues":
    render_issues(selected_volume)
elif current_page == "pdf_viewer":
    render_pdf_viewer(selected_volume, selected_issue)
elif current_page == "about":
    render_about()
else:
    render_home()