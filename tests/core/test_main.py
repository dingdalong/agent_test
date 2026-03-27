# tests/core/test_main.py
"""Tests for main.py entry point."""
import main


class TestMainModule:

    def test_main_function_exists(self):
        assert hasattr(main, "main")
        assert callable(main.main)

    def test_main_is_coroutine_function(self):
        import asyncio
        assert asyncio.iscoroutinefunction(main.main)
