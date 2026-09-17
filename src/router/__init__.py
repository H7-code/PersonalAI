"""
ARIA Language Router Package
Classifies inputs into ENGLISH or URDU. Mixed English/Urdu uses URDU mode.
"""
from src.router.language_router import LanguageRouter, LanguageMode, RoutingResult

__all__ = ["LanguageRouter", "LanguageMode", "RoutingResult"]
