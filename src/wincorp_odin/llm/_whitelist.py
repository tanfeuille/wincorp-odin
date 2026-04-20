"""Whitelist stricte des extra_kwargs acceptes par provider (R13).

@spec specs/llm-factory.spec.md v1.2

Interdit explicitement : base_url, default_headers, http_client, api_key,
anthropic_api_url (echappatoires reseau/secrets).
"""
from __future__ import annotations

from wincorp_odin.llm.exceptions import ExtraKwargsForbiddenError

# Table de whitelist par identifiant provider (format `pkg.module:Classe`).
PROVIDER_EXTRA_KWARGS_WHITELIST: dict[str, frozenset[str]] = {
    "langchain_anthropic:ChatAnthropic": frozenset(
        {"temperature", "top_p", "top_k", "stop_sequences", "streaming"}
    ),
}


def validate_extra_kwargs(
    model_name: str, use: str, extra_kwargs: dict[str, object] | None
) -> None:
    """Verifie que les cles d'extra_kwargs sont toutes dans la whitelist provider.

    Args:
        model_name: Nom logique (pour message d'erreur).
        use: Identifiant provider `pkg.module:Classe`.
        extra_kwargs: Dict a verifier (peut etre None/vide).

    Raises:
        ExtraKwargsForbiddenError: si au moins une cle est hors whitelist.
    """
    if not extra_kwargs:
        return
    allowed = PROVIDER_EXTRA_KWARGS_WHITELIST.get(use, frozenset())
    rejected = [k for k in extra_kwargs if k not in allowed]
    if rejected:
        raise ExtraKwargsForbiddenError(
            f"[ERREUR] Modele '{model_name}' : extra_kwargs contient les cles "
            f"interdites {sorted(rejected)}. Provider {use} — whitelist : "
            f"{sorted(allowed)}. Retirer les cles interdites."
        )
