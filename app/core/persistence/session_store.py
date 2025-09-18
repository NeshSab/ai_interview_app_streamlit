"""
Purpose: Session transcript & metadata storage (now in-memory; later SQLite/file/Cloud).
Why: Reopen sessions, analytics, export.

What is inside (now):
InMemorySessionStore with get/set/reset.
Later:
SQLiteSessionStore with simple schema: sessions, messages, metrics.
Methods: append_message, list_sessions, load(session_id), save(session_id).

Testing:
In-memory: simple state tests.
SQLite: tmp DB fixture; migration tests.
"""

from core.models import Message


class InMemorySessionStore:
    def __init__(self) -> None:
        self._messages: list[Message] = []

    def get(self) -> list[Message]:
        return self._messages

    def set(self, msgs: list[Message]) -> None:
        self._messages = msgs[:]

    def reset(self) -> None:
        self._messages = []
