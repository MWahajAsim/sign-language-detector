"""Text accumulator and sentence builder module.

Handles appending recognized fingerspelled letters into coherent words and sentences,
managing whitespace, backspace deletion, and session resets.
"""

from typing import List


class TextAccumulator:
    """Maintains text state and sentence building operations."""

    def __init__(self, initial_text: str = ""):
        self.text: str = initial_text
        self.history: List[str] = [initial_text]

    def append(self, character: str) -> str:
        """Append a recognized character to the text buffer.

        Args:
            character: String of length 1 or special token ('SPACE', 'BACKSPACE').

        Returns:
            Updated text string.
        """
        if not character:
            return self.text

        if character == "SPACE":
            return self.add_space()
        elif character == "BACKSPACE":
            return self.backspace()
        elif character == "CLEAR":
            return self.clear()

        self.text += character
        self.history.append(self.text)
        return self.text

    def add_space(self) -> str:
        """Append a single whitespace character if not already trailing."""
        if self.text and not self.text.endswith(" "):
            self.text += " "
            self.history.append(self.text)
        return self.text

    def backspace(self) -> str:
        """Remove the last character from the accumulated text."""
        if len(self.text) > 0:
            self.text = self.text[:-1]
            self.history.append(self.text)
        return self.text

    def clear(self) -> str:
        """Clear all accumulated text."""
        self.text = ""
        self.history = [""]
        return self.text

    def undo(self) -> str:
        """Revert to the previous text state."""
        if len(self.history) > 1:
            self.history.pop()
            self.text = self.history[-1]
        return self.text

    def get_text(self) -> str:
        """Return the current accumulated text."""
        return self.text
