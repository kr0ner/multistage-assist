"""Tests for fuzzy matching utilities (REQ-QUAL-001, REQ-TEST-002).

Covers: fuzzy_match, fuzzy_match_best, fuzzy_match_all, levenshtein_distance,
normalize_for_fuzzy, fuzzy_match_candidates.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from multistage_assist.utils.fuzzy_utils import (
    fuzzy_match,
    levenshtein_distance,
    normalize_for_fuzzy,
)


class TestFuzzyMatchSync:
    """Tests for synchronous fuzzy matching (difflib fallback)."""

    def test_identical_strings_score_100(self):
        assert fuzzy_match("Wohnzimmer", "Wohnzimmer") == 100

    def test_empty_query_returns_0(self):
        assert fuzzy_match("", "something") == 0

    def test_empty_candidate_returns_0(self):
        assert fuzzy_match("something", "") == 0

    def test_both_empty_returns_0(self):
        assert fuzzy_match("", "") == 0

    def test_none_query_returns_0(self):
        assert fuzzy_match(None, "something") == 0

    def test_similar_strings_high_score(self):
        score = fuzzy_match("wohnzimmer", "Wohnzimmer")
        assert score == 100  # Case insensitive

    def test_different_strings_low_score(self):
        score = fuzzy_match("garage", "schlafzimmer")
        assert score < 50


class TestLevenshteinDistance:
    """Tests for Levenshtein edit distance."""

    def test_identical_strings(self):
        assert levenshtein_distance("abc", "abc") == 0

    def test_single_insertion(self):
        assert levenshtein_distance("abc", "abcd") == 1

    def test_single_deletion(self):
        assert levenshtein_distance("abcd", "abc") == 1

    def test_single_substitution(self):
        assert levenshtein_distance("abc", "adc") == 1

    def test_empty_strings(self):
        assert levenshtein_distance("", "") == 0

    def test_one_empty(self):
        assert levenshtein_distance("abc", "") == 3

    def test_german_umlauts(self):
        assert levenshtein_distance("küche", "kuche") == 1


class TestNormalizeForFuzzy:
    """Tests for text normalization."""

    def test_removes_articles(self):
        result = normalize_for_fuzzy("den Keller")
        assert "den" not in result
        assert "keller" in result

    def test_removes_prepositions(self):
        result = normalize_for_fuzzy("im Wohnzimmer")
        # "im" as a standalone word should be removed
        assert result.strip() == "wohnzimmer"

    def test_lowercases(self):
        assert normalize_for_fuzzy("KÜCHE") == normalize_for_fuzzy("küche")

    def test_empty_string(self):
        assert normalize_for_fuzzy("") == ""

    def test_none_returns_empty(self):
        assert normalize_for_fuzzy(None) == ""


class TestFuzzyMatchBestAsync:
    """Tests for async fuzzy_match_best."""

    @pytest.mark.asyncio
    async def test_exact_match_returns_high_score(self):
        from multistage_assist.utils.fuzzy_utils import fuzzy_match_best
        result = await fuzzy_match_best("küche", ["küche", "bad", "garage"])
        assert result is not None
        assert result[0] == "küche"
        assert result[1] == 100

    @pytest.mark.asyncio
    async def test_no_match_below_threshold(self):
        from multistage_assist.utils.fuzzy_utils import fuzzy_match_best
        result = await fuzzy_match_best("xyz", ["küche", "bad"], threshold=90)
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_query_returns_none(self):
        from multistage_assist.utils.fuzzy_utils import fuzzy_match_best
        assert await fuzzy_match_best("", ["a", "b"]) is None

    @pytest.mark.asyncio
    async def test_empty_candidates_returns_none(self):
        from multistage_assist.utils.fuzzy_utils import fuzzy_match_best
        assert await fuzzy_match_best("test", []) is None


class TestFuzzyMatchAllAsync:
    """Tests for async fuzzy_match_all."""

    @pytest.mark.asyncio
    async def test_returns_sorted_matches(self):
        from multistage_assist.utils.fuzzy_utils import fuzzy_match_all
        results = await fuzzy_match_all("bad", ["bad", "bade", "garage"], threshold=50)
        assert len(results) >= 1
        assert results[0][0] == "bad"
        # Verify descending score order
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_empty_returns_empty(self):
        from multistage_assist.utils.fuzzy_utils import fuzzy_match_all
        assert await fuzzy_match_all("", ["a"]) == []


class TestFuzzyMatchCandidates:
    """Tests for fuzzy_match_candidates (dict-based matching)."""

    @pytest.mark.asyncio
    async def test_matches_by_name(self):
        from multistage_assist.utils.fuzzy_utils import fuzzy_match_candidates
        candidates = [
            {"name": "Familienkalender", "entity_id": "calendar.family"},
            {"name": "Arbeitskalender", "entity_id": "calendar.work"},
        ]
        result = await fuzzy_match_candidates("Familie", candidates, threshold=60)
        assert result == "calendar.family"

    @pytest.mark.asyncio
    async def test_no_match_returns_none(self):
        from multistage_assist.utils.fuzzy_utils import fuzzy_match_candidates
        candidates = [
            {"name": "Familienkalender", "entity_id": "calendar.family"},
        ]
        result = await fuzzy_match_candidates("xyz", candidates, threshold=90)
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_candidates_returns_none(self):
        from multistage_assist.utils.fuzzy_utils import fuzzy_match_candidates
        result = await fuzzy_match_candidates("test", [])
        assert result is None
