import pytest
from unittest.mock import AsyncMock, MagicMock
import asyncio


async def test_async_mock_return_value():
    m = AsyncMock(return_value={"a": 1})
    res = await m()
    print(f"DEBUG: res type: {type(res)}")
    print(f"DEBUG: res: {res}")
    assert isinstance(res, dict)
    assert res["a"] == 1


async def test_async_mock_side_effect_async_func():
    async def side_effect():
        return {"a": 1}

    m = AsyncMock(side_effect=side_effect)
    res = await m()
    print(f"DEBUG: res type: {type(res)}")
    print(f"DEBUG: res: {res}")
    assert isinstance(res, dict)
    assert res["a"] == 1


async def test_magic_mock_speech():
    m = MagicMock()
    m.speech = {"plain": {"speech": "Done"}}
    print(f"DEBUG: m.speech type: {type(m.speech)}")
    print(f"DEBUG: m.speech: {m.speech}")
    assert isinstance(m.speech, dict)
    assert m.speech["plain"]["speech"] == "Done"
