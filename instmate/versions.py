"""Adopted arrangement versions per history (in memory).

A history starts at generate. Only results that passed their checks are adopted; undo moves the head back to
the parent version and returns its stored result unchanged (no regeneration), so every bar's chord and every
arrangement event is identical to what was adopted.

A history also holds the clash proposals a natural-language request ended on (`set_pending`): failed checks the user may
force with an explicit confirmation. They are bound to the version they were made for and disappear as soon as the head
moves (adopt or undo), so a confirmation can never be applied to a different version.
"""
from __future__ import annotations

import copy
import threading
import uuid


class UnknownHistory(KeyError):
    pass


class NothingToUndo(ValueError):
    pass


class StaleBase(ValueError):
    pass


class VersionStore:
    def __init__(self) -> None:
        self._histories: dict[str, dict] = {}
        self._lock = threading.Lock()

    def _get(self, history: str) -> dict:
        try:
            return self._histories[history]
        except KeyError:
            raise UnknownHistory(history) from None

    @staticmethod
    def _info(history: str, h: dict) -> dict:
        v = h["versions"][h["head"]]
        return {"history": history, "index": v["index"], "parent": v["parent"], "can_undo": v["parent"] is not None}

    def _add(self, history: str, h: dict, parent: int | None, state: dict, result: dict) -> dict:
        index = len(h["versions"])
        h["versions"].append({"index": index, "parent": parent,
                              "state": copy.deepcopy(state), "result": copy.deepcopy(result)})
        h["head"] = index
        h["pending"] = None                  # proposals belong to the version they were made for
        return self._info(history, h)

    def start(self, state: dict, result: dict) -> dict:
        history = uuid.uuid4().hex
        with self._lock:
            h = self._histories[history] = {"versions": [], "head": None, "pending": None}
            return self._add(history, h, None, state, result)

    def adopt(self, history: str, parent: int, state: dict, result: dict) -> dict:
        """Append a checked result as a child of `parent`, which must still be the head."""
        with self._lock:
            h = self._get(history)
            if h["head"] != parent:
                raise StaleBase(f"version {parent} is no longer current (current is {h['head']})")
            return self._add(history, h, parent, state, result)

    def set_pending(self, history: str, base_index: int, proposals: list[dict]) -> None:
        """Replace the forceable proposals of this history (empty list clears them). Ignored if `base_index` is not the head."""
        with self._lock:
            h = self._get(history)
            h["pending"] = {"base_index": base_index, "proposals": copy.deepcopy(proposals)} \
                if proposals and h["head"] == base_index else None

    def pending(self, history: str, base_index: int) -> list[dict]:
        """The proposals made for `base_index`, or [] when there are none or the head has moved on."""
        with self._lock:
            h = self._get(history)
            p = h["pending"]
            return copy.deepcopy(p["proposals"]) if p and p["base_index"] == base_index == h["head"] else []

    def head(self, history: str) -> dict:
        with self._lock:
            h = self._get(history)
            return {**copy.deepcopy(h["versions"][h["head"]]), "info": self._info(history, h)}

    def undo(self, history: str) -> dict:
        with self._lock:
            h = self._get(history)
            parent = h["versions"][h["head"]]["parent"]
            if parent is None:
                raise NothingToUndo("already at the first version")
            h["head"] = parent
            h["pending"] = None
            return {**copy.deepcopy(h["versions"][parent]), "info": self._info(history, h)}

    def listing(self, history: str) -> dict:
        with self._lock:
            h = self._get(history)
            return {"history": history, "head": h["head"],
                    "versions": [{"index": v["index"], "parent": v["parent"], "chords": v["state"]["chords"]}
                                 for v in h["versions"]]}
