"""Tests for clarification bypass heuristics.

Default: NO split unless we detect:
1. Multi-area pattern with compound separator connecting areas
2. Sentence >20 words  
3. Implicit phrases that need transformation
"""

import pytest
import sys
import os

sys.path.insert(0, os.getcwd())

from multistage_assist.capabilities.clarification import ClarificationCapability
from multistage_assist.utils.german_utils import IMPLICIT_PHRASES


class TestMultiAreaDetection:
    """Test _has_multi_area_pattern() method."""
    
    @pytest.fixture
    def capability(self):
        """Create ClarificationCapability instance."""
        cap = ClarificationCapability.__new__(ClarificationCapability)
        return cap
    
    def test_no_und_no_multi_area(self, capability):
        """Text without compound separator should never be multi-area."""
        assert capability._has_multi_area_pattern("Schalte Licht im Büro an") is False
        assert capability._has_multi_area_pattern("Alle Lichter aus") is False
    
    def test_explicit_multi_area_pattern(self, capability):
        """'im X und im Y' pattern should be detected."""
        assert capability._has_multi_area_pattern(
            "Schalte das Licht im Wohnzimmer und im Schlafzimmer aus"
        ) is True
    
    def test_in_der_pattern(self, capability):
        """'in der X und in der Y' pattern should be detected."""
        assert capability._has_multi_area_pattern(
            "Licht in der Küche an und in der Garage aus"
        ) is True
    
    def test_floor_pattern(self, capability):
        """Floor patterns like 'Erdgeschoss und Obergeschoss' should be detected."""
        assert capability._has_multi_area_pattern(
            "Rolläden im Erdgeschoss und Obergeschoss herunter"
        ) is True
        assert capability._has_multi_area_pattern(
            "Alle Lichter im ersten Stock und zweiten Stock aus"
        ) is True
    
    def test_device_and_device_no_split(self, capability):
        """'Licht und Rollladen' in same area should NOT trigger split."""
        # No area indicator on both sides
        assert capability._has_multi_area_pattern(
            "Licht und Rollladen im Büro"
        ) is False
    
    def test_action_and_action_no_split(self, capability):
        """'an und aus' or 'hoch und runter' should NOT trigger split."""
        assert capability._has_multi_area_pattern(
            "Rollladen hoch und Licht an"
        ) is False
    
    def test_same_area_different_actions_detected(self, capability):
        """'Licht in Küche an und im Flur aus' should be detected (different areas)."""
        assert capability._has_multi_area_pattern(
            "Licht in der Küche an und im Flur aus"
        ) is True


class TestClarificationBypassDecisions:
    """Test when clarification should bypass vs call LLM."""
    
    def test_simple_command_bypasses(self):
        """Simple command without triggers should bypass LLM."""
        cap = ClarificationCapability.__new__(ClarificationCapability)
        
        # These should all return False (no split needed)
        simple_commands = [
            "Schalte Licht an",
            "Rollladen im Büro hoch",
            "Stell die Heizung auf 22 Grad",
            "Licht in der Küche aus",
            "Timer auf 10 Minuten",
        ]
        
        for cmd in simple_commands:
            # Check none of the LLM triggers fire
            text_lower = cmd.lower()
            word_count = len(cmd.split())
            
            needs_rephrasing = any(p in text_lower for p in IMPLICIT_PHRASES)
            is_long = word_count > 20
            is_multi_area = cap._has_multi_area_pattern(cmd)
            
            assert not needs_rephrasing, f"'{cmd}' incorrectly detected implicit phrase"
            assert not is_long, f"'{cmd}' incorrectly detected as long"
            assert not is_multi_area, f"'{cmd}' incorrectly detected as multi-area"
    
    def test_implicit_phrase_triggers_llm(self):
        """Implicit phrases should trigger LLM call."""
        phrases_that_need_llm = [
            "Es ist zu dunkel",
            "Im Büro ist es zu kalt",
            "Zu hell hier",
            "Das Wohnzimmer ist zu warm",
        ]
        
        for phrase in phrases_that_need_llm:
            text_lower = phrase.lower()
            needs_rephrasing = any(p in text_lower for p in IMPLICIT_PHRASES)
            assert needs_rephrasing, f"'{phrase}' should trigger LLM for rephrasing"
    
    def test_long_sentence_triggers_llm(self):
        """Sentence >20 words should trigger LLM."""
        long_sentence = (
            "Schalte bitte das Licht im Wohnzimmer an und dann auch noch "
            "die Heizung auf 22 Grad einstellen und vielleicht auch noch "
            "den Rollladen ein bisschen herunterfahren"
        )
        
        word_count = len(long_sentence.split())
        assert word_count > 20, f"Test sentence only has {word_count} words"
    
    def test_multi_area_triggers_llm(self):
        """Multi-area patterns should trigger LLM."""
        cap = ClarificationCapability.__new__(ClarificationCapability)
        
        multi_area_commands = [
            "Licht im Wohnzimmer und im Schlafzimmer aus",
            "Rolläden im Erdgeschoss und Obergeschoss hoch",
            "Alle Lichter in der Küche an und im Flur aus",
        ]
        
        for cmd in multi_area_commands:
            assert cap._has_multi_area_pattern(cmd), f"'{cmd}' should be detected as multi-area"


class TestEdgeCases:
    """Test edge cases for clarification bypass."""
    
    def test_timer_with_und_not_multi_area(self):
        """Timer commands with 'und' in description should not be multi-area."""
        cap = ClarificationCapability.__new__(ClarificationCapability)
        
        # "und" here is in the timer label, not connecting areas
        assert cap._has_multi_area_pattern(
            "Timer für 10 Minuten für Nudeln und Soße"
        ) is False
    
    def test_calendar_event_not_multi_area(self):
        """Calendar events should not trigger multi-area split."""
        cap = ClarificationCapability.__new__(ClarificationCapability)
        
        # "und" connects a time range or event parts
        assert cap._has_multi_area_pattern(
            "Termin um 15 Uhr Meeting und Präsentation"
        ) is False
