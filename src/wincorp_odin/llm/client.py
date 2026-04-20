"""Factory pour client SDK Anthropic brut (pas LangChain).

@spec specs/llm-factory.spec.md v1.3.2 §27

Consommateurs cibles : `wincorp-heimdall` (services extraction.py,
categorization.py, chat_agent.py, ocr.py, pipeline_ocr.py) qui utilisent
l'API brute `client.messages.create(...)` plutot que `model.invoke(...)`
(LangChain).

Phase 1.9a de la migration DeerFlow : `create_client()` est le second entry
point de la factory — jumeau de `create_model()` mais retourne
`anthropic.Anthropic` au lieu de `ChatAnthropic`. Partage la meme config
YAML (`wincorp-urd/referentiels/models.yaml`) et le meme mecanisme de
resolution de secrets via `${VAR}` env.

Pas de cache : le SDK Anthropic gere en interne un pool HTTP (httpx) qui
rend l'instanciation quasi-gratuite ; pas de wrapping middlewares
(breaker/retry/tokens) cote SDK brut — ces fonctionnalites vivent dans
l'adapter LangChain. Les consommateurs qui veulent les middlewares migrent
vers `create_model()` en Phase 1.9b (non decidee).
"""
from __future__ import annotations

import anthropic

from wincorp_odin.llm.config import load_models_config
from wincorp_odin.llm.exceptions import (
    CapabilityMismatchError,
    ModelNotFoundError,
)

# Prefixes des `use:` compatibles SDK Anthropic brut. La cle API partagee
# (`ANTHROPIC_API_KEY`) permet de reutiliser la meme entree models.yaml
# pour l'adapter LangChain (`langchain_anthropic:ChatAnthropic`) et pour le
# SDK raw (`anthropic:Anthropic`). Tout autre provider (`langchain_openai:*`,
# `langchain_deepseek:*`, etc.) leve `CapabilityMismatchError`.
_ANTHROPIC_COMPATIBLE_PREFIXES = ("anthropic:", "langchain_anthropic:")


def create_client(name: str) -> anthropic.Anthropic:
    """Retourne un client `anthropic.Anthropic` configure depuis models.yaml.

    Args:
        name: Cle logique du modele declaree dans `models.yaml` (ex.
            `"claude-sonnet"`). Le champ `use:` du modele doit commencer
            par `anthropic:` ou `langchain_anthropic:` — les deux partagent
            la meme cle API.

    Returns:
        Instance `anthropic.Anthropic` prete pour `client.messages.create(...)`.
        Chaque appel retourne une nouvelle instance (pas de cache — voir
        note module).

    Raises:
        ModelNotFoundError: `name` absent de `models.yaml` ou modele disabled.
            Le message FR liste les modeles disponibles (tri alpha).
        CapabilityMismatchError: le champ `use:` pointe vers un provider
            non Anthropic-compatible (ex. `langchain_openai:ChatOpenAI`).
            Le message FR redirige vers `create_model()`.
        ModelConfigError (et sous-classes) : voir `load_models_config`
            (YAML invalide, `${VAR}` non resolue, conflit OneDrive, etc.).
    """
    configs = load_models_config()

    cfg = configs.get(name)
    if cfg is None or cfg.disabled:
        available = sorted(n for n, c in configs.items() if not c.disabled)
        raise ModelNotFoundError(
            f"[ERREUR] Modele '{name}' introuvable dans models.yaml. "
            f"Modeles disponibles : {available}. "
            f"Verifier wincorp-urd/referentiels/models.yaml."
        )

    if not cfg.use.startswith(_ANTHROPIC_COMPATIBLE_PREFIXES):
        raise CapabilityMismatchError(
            f"[ERREUR] create_client() ne supporte que les providers "
            f"Anthropic-compatibles (use: commencant par "
            f"'anthropic:' ou 'langchain_anthropic:'). "
            f"Modele '{name}' utilise '{cfg.use}'. "
            f"Pour les autres providers, utiliser create_model() (LangChain)."
        )

    return anthropic.Anthropic(api_key=cfg.api_key_resolved)
