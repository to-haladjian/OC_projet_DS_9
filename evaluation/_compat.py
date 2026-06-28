"""Compatibility shim so Ragas 0.4.x imports on this project's langchain 1.x stack.

``ragas.llms.base`` hard-imports ``ChatVertexAI`` / ``VertexAI`` from
``langchain_community``, but those were dropped from ``langchain-community`` 0.4.x. We
never use Vertex AI, so we register a stub module before Ragas is imported. Import this
module (and call :func:`apply`) *before* importing anything from ``ragas``.
"""

from __future__ import annotations

import sys
import types

import langchain_community.llms as community_llms


def apply() -> None:
    """Register stub ``langchain_community`` Vertex AI symbols (idempotent)."""
    module_name = "langchain_community.chat_models.vertexai"
    if module_name not in sys.modules:
        stub = types.ModuleType(module_name)

        class ChatVertexAI:  # noqa: D401 - unused stub, only needed for the import
            """Stub: Vertex AI is not used in this project."""

        stub.ChatVertexAI = ChatVertexAI
        sys.modules[module_name] = stub

    # ``from langchain_community.llms import VertexAI`` also needs to resolve.
    if not hasattr(community_llms, "VertexAI"):
        community_llms.VertexAI = sys.modules[module_name].ChatVertexAI


apply()
