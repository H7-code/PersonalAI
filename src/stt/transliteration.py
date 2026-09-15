"""
ARIA Deterministic Perso-Arabic to Roman Urdu Phonetic Transliterator
Provides deterministic normalization of Nastaliq/Arabic script Urdu to Roman Urdu (Latin script).
"""

import re
from typing import Dict

# Common Urdu words mapping for high-frequency vocabulary
URDU_WORD_MAP: Dict[str, str] = {
    "آپ": "aap",
    "کا": "ka",
    "کی": "ki",
    "کے": "ke",
    "کو": "ko",
    "سے": "se",
    "میں": "main",
    "پر": "par",
    "ہے": "hai",
    "ہیں": "hain",
    "ہو": "ho",
    "ہوں": "hoon",
    "تھا": "tha",
    "تھی": "thi",
    "تھے": "the",
    "دن": "din",
    "کیسا": "kaisa",
    "کیسی": "kaisi",
    "کیسے": "kaisay",
    "رہا": "raha",
    "رہی": "rahi",
    "رہے": "rahey",
    "مدد": "madad",
    "کرنا": "karna",
    "کر": "kar",
    "کرو": "karo",
    "چاہتا": "chahta",
    "چاہتی": "chahti",
    "بہت": "bahut",
    "اچھا": "acha",
    "اچھی": "achi",
    "سوال": "sawaal",
    "سوچنا": "sochna",
    "سوچ": "soch",
    "دینا": "dena",
    "ہوگا": "hoga",
    "ہوگی": "hogi",
    "ٹھیک": "theek",
    "اب": "ab",
    "بات": "baat",
    "کرتے": "karte",
    "میرا": "mera",
    "میری": "meri",
    "نام": "naam",
    "اور": "aur",
    "اسسٹنٹ": "assistant",
    "شکریہ": "shukriya",
    "اہم": "important",
    "سسٹم": "system",
    "سیٹپ": "setup",
    "اپڈیٹ": "update",
    "گیا": "gaya",
    "گئی": "gayi",
    "تھوڑی": "thori",
    "تھوڑا": "thora",
    "زیادہ": "zyada",
    "چیک": "check",
    "فائل": "file",
    "ڈاؤنلوڈ": "download",
    "مکمل": "complete",
    "سکتے": "sakte",
    "نیٹ ورک": "network",
    "کنکشن": "connection",
    "سپیڈ": "speed",
    "ماڈل": "model",
    "لوڈ": "load",
    "انتظار": "wait",
    "پہلے": "pehle",
    "بیک اپ": "backup",
    "پھر": "phir",
    "ڈیلیٹ": "delete",
    "میسج": "message",
    "ایرر": "error",
    "لگ": "lag",
    "ایپ": "app",
    "کریش": "crash",
    "ریسٹارٹ": "restart",
    "دیا": "diya",
    "بالکل": "bilkul",
    "یقین": "yaqeen",
    "ایک": "ek",
    "سیکنڈ": "second",
}

# Character-level phonetic mappings
URDU_CHAR_MAP: Dict[str, str] = {
    "ا": "a", "آ": "aa", "ب": "b", "پ": "p", "ت": "t", "ٹ": "t", "ث": "s",
    "ج": "j", "چ": "ch", "ح": "h", "خ": "kh", "د": "d", "ڈ": "d", "ذ": "z",
    "ر": "r", "ڑ": "r", "ز": "z", "ژ": "zh", "س": "s", "ش": "sh", "ص": "s",
    "ض": "z", "ط": "t", "ظ": "z", "ع": "a", "غ": "gh", "ف": "f", "ق": "q",
    "ک": "k", "گ": "g", "ل": "l", "م": "m", "ن": "n", "ں": "n", "و": "o",
    "ہ": "h", "ۂ": "h", "ۃ": "t", "ھ": "h", "ء": "", "ی": "i", "ے": "e",
    "ئ": "y", "ۓ": "e", "۔": ".", "،": ",", "؟": "?",
}

def is_perso_arabic(text: str) -> bool:
    """Detects if text contains Arabic/Persian/Urdu Unicode script."""
    return bool(re.search(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]", text))

def remove_diacritics(text: str) -> str:
    """Removes Arabic/Urdu diacritics (Zabar, Zer, Pesh, Tashdeed, Sukun, Tanween)."""
    return re.sub(r"[\u064B-\u065F\u0670]", "", text)

def transliterate_to_roman_urdu(text: str) -> str:
    """
    Converts Perso-Arabic Urdu text into Roman Urdu (Latin script).
    Retains existing Latin words (e.g. English technical terms).
    """
    if not is_perso_arabic(text):
        return text

    clean_text = remove_diacritics(text)
    words = clean_text.split()
    translated_words = []

    for w in words:
        clean_w = re.sub(r"[^\u0600-\u06FF]", "", w)
        punct = re.sub(r"[\u0600-\u06FF]", "", w)
        
        # Check dictionary
        if clean_w in URDU_WORD_MAP:
            translated_words.append(URDU_WORD_MAP[clean_w] + punct)
        elif is_perso_arabic(clean_w):
            # Phonetic character mapping
            phonetic = "".join(URDU_CHAR_MAP.get(c, "") for c in clean_w)
            translated_words.append(phonetic + punct)
        else:
            translated_words.append(w)

    result = " ".join(translated_words)
    # Clean up double spaces or dangling punctuation
    result = re.sub(r"\s+", " ", result).strip()
    return result
