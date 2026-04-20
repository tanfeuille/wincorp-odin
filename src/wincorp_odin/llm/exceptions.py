"""Hierarchie des exceptions Odin LLM — messages FR actionnables.

@spec specs/llm-factory.spec.md v1.2

Regles de redaction R10/R10b/R10c : toute exception qui peut wrapper un appel
reseau/instanciation strip la cle API des `args` et de la chaine `__cause__`.
"""
from __future__ import annotations

import contextlib
import re

# Regex de detection multi-providers (R10d). Couvre Anthropic, OpenAI, AWS.
# Tout nouveau provider doit ajouter son pattern ici avant Phase 1.x.
_API_KEY_PATTERN = re.compile(
    r"(?:"
    r"sk-ant-(?:api\d+-)?[A-Za-z0-9_\-]{20,}"   # Anthropic (sk-ant-*, sk-ant-api03-*)
    r"|sk-proj-[A-Za-z0-9_\-]{20,}"              # OpenAI project keys
    r"|sk-[A-Za-z0-9]{32,}"                      # OpenAI/DeepSeek style generique
    r"|AKIA[0-9A-Z]{16}"                         # AWS access key
    r")"
)
_REDACTED = "***REDACTED***"


def _redact(value: object) -> object:
    """Remplace toute occurence de cle API dans une valeur quelconque.

    Appliquee recursivement aux tuples/dict/str. Les types non-strings passent
    tels quels sauf si ce sont des containers simples.
    """
    if isinstance(value, str):
        return _API_KEY_PATTERN.sub(_REDACTED, value)
    if isinstance(value, tuple):
        return tuple(_redact(v) for v in value)
    if isinstance(value, list):
        return [_redact(v) for v in value]
    if isinstance(value, dict):
        return {k: _redact(v) for k, v in value.items()}
    return value


class OdinLlmError(Exception):
    """Racine de la hierarchie — remplace MimirLlmError (PB-011)."""


class ModelConfigError(OdinLlmError):
    """Erreur config : YAML absent/invalide/doublon/conflit/taille/path interdit."""


class ModelConfigSchemaError(ModelConfigError):
    """Type incorrect ou champ obligatoire manquant — JSONPath dans le message."""


class SecretMissingError(ModelConfigError):
    """Variable d'environnement ${VAR} absente ou vide (EC7, EC8, EC9)."""


class ProviderNotInstalledError(OdinLlmError):
    """`use:` pointe vers pkg non installe / classe inexistante / non-callable."""


class ExtraKwargsForbiddenError(ModelConfigError):
    """`extra_kwargs` contient une cle hors whitelist provider (R13, EC23)."""


class ModelNotFoundError(OdinLlmError):
    """Nom logique inconnu ou modele disabled.

    PB-011 : ne derive PAS de KeyError.
    """


class CapabilityMismatchError(OdinLlmError):
    """`thinking_enabled=True` sur modele non-capable (EC14, R6).

    PB-011 : ne derive PAS de ValueError.
    """


class ModelAuthenticationError(OdinLlmError):
    """Authentification provider echouee (EC15).

    R10c : nettoie la cle API dans `args` ET dans la chaine `__cause__`.
    """

    def __init__(self, *args: object) -> None:
        cleaned = tuple(_redact(a) for a in args)
        super().__init__(*cleaned)
        # Nettoyage recursif de __cause__ au moment de l'init : la chaine
        # peut ne pas encore etre assignee (raise ... from e), on la traitera
        # aussi a la volee dans __str__.
        self._strip_cause_chain()

    def _strip_cause_chain(self) -> None:
        """Remonte la chaine __cause__ et nettoie args sur chaque maillon."""
        cursor: BaseException | None = self.__cause__
        depth = 0
        while cursor is not None and depth < 10:
            # Certaines exceptions C natives refusent la mutation — suppress
            with contextlib.suppress(AttributeError, TypeError):
                cursor.args = tuple(_redact(a) for a in cursor.args)
            cursor = cursor.__cause__
            depth += 1
