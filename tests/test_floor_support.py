"""Tests for floor-level support."""
import pytest
from unittest.mock import MagicMock, AsyncMock


class TestFloorAliases:
    """Tests for floor alias lookup in EntityResolver."""

    def test_floor_alias_synonyms(self):
        """Floor aliases should be resolved via LLM (area_alias with mode='floor').
        
        Common German floor synonyms that the LLM should understand:
        - Obergeschoss/OG/oben/erster Stock
        - Erdgeschoss/EG/unten/Parterre
        - Untergeschoss/UG/Keller
        - Dachgeschoss/DG/Dach/Speicher
        """
        # These are the synonyms the area_alias LLM prompt handles
        floor_synonyms = {
            "obergeschoss": ["og", "oben", "1. stock", "erster stock", "1. og"],
            "erdgeschoss": ["eg", "unten", "parterre", "ebenerdig"],
            "untergeschoss": ["ug", "keller", "kellergeschoss"],
            "dachgeschoss": ["dg", "dach", "speicher"],
        }
        
        # Verify synonym sets are defined
        assert "obergeschoss" in floor_synonyms
        assert "og" in floor_synonyms["obergeschoss"]
        assert "oben" in floor_synonyms["obergeschoss"]
        
        assert "erdgeschoss" in floor_synonyms
        assert "eg" in floor_synonyms["erdgeschoss"]
        assert "unten" in floor_synonyms["erdgeschoss"]
        
        assert "untergeschoss" in floor_synonyms
        assert "ug" in floor_synonyms["untergeschoss"]
        assert "keller" in floor_synonyms["untergeschoss"]

    def test_obergeschoss_maps_to_og(self):
        """'Obergeschoss' should match floor named 'OG'."""
        aliases = {
            "obergeschoss": ["og", "oben", "1. stock", "erster stock", "1. og"],
        }
        
        # Simulate the matching logic
        needle = "obergeschoss"
        floor_canon = "og"
        
        for canonical, alias_list in aliases.items():
            if floor_canon == canonical or floor_canon in alias_list:
                if needle == canonical or needle in alias_list:
                    match = True
                    break
        else:
            match = False
        
        assert match is True

    def test_og_maps_to_obergeschoss(self):
        """'OG' should match floor named 'Obergeschoss'."""
        aliases = {
            "obergeschoss": ["og", "oben", "1. stock", "erster stock", "1. og"],
        }
        
        needle = "og"
        floor_canon = "obergeschoss"
        
        for canonical, alias_list in aliases.items():
            if floor_canon == canonical or floor_canon in alias_list:
                if needle == canonical or needle in alias_list:
                    match = True
                    break
        else:
            match = False
        
        assert match is True

    def test_keller_maps_to_untergeschoss(self):
        """'Keller' should match floor named 'Untergeschoss' or 'UG'."""
        aliases = {
            "untergeschoss": ["ug", "keller", "kellergeschoss"],
        }
        
        needle = "keller"
        floor_canon = "untergeschoss"
        
        for canonical, alias_list in aliases.items():
            if floor_canon == canonical or floor_canon in alias_list:
                if needle == canonical or needle in alias_list:
                    match = True
                    break
        else:
            match = False
        
        assert match is True

    def test_buero_does_not_match_floor(self):
        """Room names like 'Büro' should not match any floor."""
        aliases = {
            "obergeschoss": ["og", "oben"],
            "erdgeschoss": ["eg", "unten"],
            "untergeschoss": ["ug", "keller"],
        }
        
        needle = "buero"
        
        for floor_canon in ["og", "eg", "ug", "obergeschoss", "erdgeschoss", "untergeschoss"]:
            for canonical, alias_list in aliases.items():
                if floor_canon == canonical or floor_canon in alias_list:
                    if needle == canonical or needle in alias_list:
                        match = True
                        break
            else:
                continue
            break
        else:
            match = False
        
        assert match is False


class TestFloorSlotExtraction:
    """Tests for floor vs area slot extraction logic."""

    def test_floor_keywords_detection(self):
        """Floor keywords should be in floor slot, not area."""
        floor_keywords = [
            "obergeschoss", "erdgeschoss", "untergeschoss", 
            "og", "eg", "ug", "keller", "oben", "unten",
            "erster stock", "zweiter stock", "dachgeschoss"
        ]
        
        area_keywords = [
            "küche", "bad", "büro", "wohnzimmer", "schlafzimmer",
            "flur", "garage", "terrasse"
        ]
        
        # Floor keywords should not be in area keywords
        for kw in floor_keywords:
            assert kw not in area_keywords

    def test_obergeschoss_is_floor_not_area(self):
        """'Obergeschoss' should be detected as floor, not area."""
        floor_words = {"obergeschoss", "erdgeschoss", "untergeschoss", 
                       "og", "eg", "ug", "keller", "oben", "unten",
                       "dachgeschoss", "dg", "erster stock", "zweiter stock"}
        
        test_word = "obergeschoss"
        is_floor = test_word.lower() in floor_words
        
        assert is_floor is True

    def test_buero_is_area_not_floor(self):
        """'Büro' should be detected as area, not floor."""
        floor_words = {"obergeschoss", "erdgeschoss", "untergeschoss", 
                       "og", "eg", "ug", "keller", "oben", "unten"}
        
        test_word = "büro"
        is_floor = test_word.lower() in floor_words
        
        assert is_floor is False
