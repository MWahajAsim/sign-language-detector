"""Unit tests for text accumulator and sentence building."""

from pathlib import Path
import sys
import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.text_builder import TextAccumulator


def test_text_accumulator_append_and_spacing():
    """Verify characters append cleanly and whitespace is properly managed."""
    builder = TextAccumulator()

    builder.append("H")
    builder.append("E")
    builder.append("L")
    builder.append("L")
    builder.append("O")
    assert builder.get_text() == "HELLO"

    # Add space
    builder.add_space()
    assert builder.get_text() == "HELLO "

    # Consecutive spaces should not duplicate
    builder.add_space()
    assert builder.get_text() == "HELLO "

    builder.append("WORLD")
    assert builder.get_text() == "HELLO WORLD"


def test_text_accumulator_backspace_and_clear():
    """Verify backspace removes characters and clear empties the string."""
    builder = TextAccumulator("TEST")

    builder.backspace()
    assert builder.get_text() == "TES"

    builder.clear()
    assert builder.get_text() == ""

    # Backspace on empty string does not crash
    builder.backspace()
    assert builder.get_text() == ""
