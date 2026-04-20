"""Wrapper deprecated — usage transitoire Phase 1.9 (§18).

@spec specs/llm-factory.spec.md v1.2
"""
from __future__ import annotations

import warnings
from typing import Any


def deprecated_direct_chat_anthropic(**kwargs: Any) -> Any:
    """Wrapper transitoire vers ChatAnthropic directement.

    Emet un DeprecationWarning FR. A supprimer en Phase 2.0.
    """
    warnings.warn(
        "Usage direct de ChatAnthropic detecte. "
        "Migrer vers wincorp_odin.llm.create_model('<nom>') "
        "en definissant le modele dans wincorp-urd/referentiels/models.yaml. "
        "Ce wrapper sera retire en Phase 2.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Import tardif pour eviter de charger langchain_anthropic si jamais appele
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(**kwargs)
