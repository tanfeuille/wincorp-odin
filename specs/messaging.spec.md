# messaging — Specification

> **Statut :** IMPLEMENTED
> **Version :** 1.0
> **Niveau :** 2 (standard)
> **Auteur :** Tan Phi HUYNH
> **Date de creation :** 2026-04-22
> **Implementation :** 81 tests verts / 100% branch coverage / Ruff + MyPy strict OK / 2026-04-22

---

## 1. Objectif

Fournir un **MessageBus asyncio** et deux **canaux concrets** (Telegram + WhatsApp) pour permettre aux agents Odin de :
1. **Recevoir** des messages utilisateur (webhook inbound traduit en `InboundMessage`).
2. **Émettre** des notifications / réponses (`OutboundMessage`).
3. **Dispatcher** des commandes (`/new`, `/status`, etc.) via une allowlist centralisée.

Debloque le backlog **[[project_backlog_notif_telephone]]** (attendu depuis 19/04) : alerter tp quand un sub-agent Claude Code a besoin d'intervention OU qu'une session d'autonomie est terminée.

Port partiel du pattern DeerFlow Phase 6 (cf `memory/project_deerflow_inspiration_plan.md`). Extension WhatsApp ajoutée hors-plan à la demande user 2026-04-22.

---

## 2. Perimetre

### IN

- ABC `Channel` + dataclasses `InboundMessage` / `OutboundMessage`.
- `MessageBus` asyncio FIFO inbound + fan-out outbound par canal.
- Registry lazy `_CHANNEL_REGISTRY` avec flag `enabled` YAML (diff tanfeuille / tanph).
- Commands frozenset `KNOWN_COMMANDS` + parser central.
- Rate limiter token bucket (anti-spam, anti-429 upstream).
- Security helper `safe_download_path` (path traversal guard pour fichiers sortants).
- `TelegramChannel` — send texte via Telegram Bot API + parser payload webhook.
- `WhatsAppChannel` — send texte via Meta WhatsApp Cloud API + parser payload webhook.

### OUT

- Serveur HTTP / endpoint webhook → vit dans `wincorp-heimdall` (FastAPI). Odin fournit juste le parser.
- Persistance Supabase (`thread_mapping`) → Phase 6.7 différée.
- Streaming tokens 350ms (`CHANNEL_CAPABILITIES`) → Phase 6.8 différée.
- Auth allowlist par canal → Phase 6.5 présente mais minimale (allowlist user_ids en config YAML, pas de JWT).
- Canaux Discord, mail, Feishu, Slack → Phase 6.10 différée.
- Tests live (nécessitent tokens réels Telegram/Meta) — tests unitaires mockés uniquement.

---

## 3. Interface

### ABC + Dataclasses

```python
class Channel(ABC):
    name: str
    @abstractmethod
    async def start(self) -> None: ...
    @abstractmethod
    async def stop(self) -> None: ...
    @abstractmethod
    async def send(self, message: OutboundMessage) -> None: ...

@dataclass(frozen=True)
class InboundMessage:
    channel_name: str  # "telegram" | "whatsapp" | ...
    sender_id: str     # user_id Telegram ou phone WhatsApp
    chat_id: str       # conversation context (même que sender_id en 1-1)
    text: str
    timestamp: datetime
    thread_id: str | None = None  # parent thread si applicable
    raw_payload: dict = field(default_factory=dict)

@dataclass(frozen=True)
class OutboundMessage:
    channel_name: str
    recipient_id: str
    text: str
    reply_to_message_id: str | None = None
```

### MessageBus

```python
class MessageBus:
    def __init__(self) -> None: ...
    def register_channel(self, channel: Channel) -> None: ...
    def register_handler(self, handler: Callable[[InboundMessage], Awaitable[None]]) -> None: ...
    async def publish_inbound(self, msg: InboundMessage) -> None:
        """Appelle tous les handlers enregistrés (fan-in)."""
    async def publish_outbound(self, msg: OutboundMessage) -> None:
        """Route vers le channel dont name == msg.channel_name. Rate-limit applied."""
    async def start_all(self) -> None: ...
    async def stop_all(self) -> None: ...
```

### Registry lazy

```python
_CHANNEL_REGISTRY: dict[str, str] = {
    "telegram": "wincorp_odin.messaging.channels.telegram:TelegramChannel",
    "whatsapp": "wincorp_odin.messaging.channels.whatsapp:WhatsAppChannel",
}

def load_channel(name: str, config: dict) -> Channel:
    """Import dynamique via use: path."""
```

### Commands

```python
KNOWN_COMMANDS: frozenset[str] = frozenset({"/new", "/status", "/memory", "/models", "/help"})

def parse_command(text: str) -> tuple[str, list[str]] | None:
    """Retourne ('/cmd', [arg1, arg2]) si text.startswith('/cmd '), sinon None."""
```

### Rate limiter

```python
class TokenBucket:
    def __init__(self, rate_per_second: float, capacity: int) -> None: ...
    async def acquire(self) -> None:
        """Bloque asyncio jusqu'à un token dispo. Respect rate_per_second."""
```

### Security

```python
_SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9_\-.]+$")

def safe_download_path(filename: str, base_dir: Path) -> Path:
    """Retourne base_dir/filename validé, raise ValueError si path traversal."""
```

### Channels

```python
class TelegramChannel(Channel):
    def __init__(self, *, bot_token: str, allowed_user_ids: set[int] | None = None,
                 rate_limit: TokenBucket | None = None) -> None: ...
    @classmethod
    def parse_webhook(cls, payload: dict) -> InboundMessage | None: ...

class WhatsAppChannel(Channel):
    def __init__(self, *, phone_number_id: str, access_token: str,
                 allowed_phone_numbers: set[str] | None = None,
                 rate_limit: TokenBucket | None = None) -> None: ...
    @classmethod
    def parse_webhook(cls, payload: dict) -> InboundMessage | None: ...
```

### Erreurs

| Type | Condition | Comportement |
|------|-----------|--------------|
| `ValueError` | `safe_download_path` : filename avec `..` ou `/` | Lever |
| `ValueError` | `load_channel` : nom inconnu dans registry | Lever avec message FR |
| `ChannelSendError` | `send()` : réponse HTTP non-2xx | Lever avec status_code + body |
| `ChannelAuthError` | `send()` : 401/403 | Lever (sous-classe ChannelSendError) |

---

## 4. Regles metier

- **R1** — `MessageBus.publish_outbound` route le message vers le channel dont `name == msg.channel_name`. Si inconnu → `ChannelNotFoundError`.
- **R2** — `MessageBus.publish_inbound` appelle tous les handlers enregistrés (fan-out). Si un handler raise → log + continuer (non bloquant).
- **R3** — Le registry `_CHANNEL_REGISTRY` est lazy : l'import n'a lieu que lors de `load_channel(name)`. Permet d'avoir des canaux optionnels sans imposer de dep.
- **R4** — `TokenBucket` respecte `rate_per_second` (ex: 25 msg/s Telegram, 80 msg/s WhatsApp). `acquire()` est idempotent : n appels consécutifs attendent jusqu'à n tokens disponibles.
- **R5** — `safe_download_path(filename, base_dir)` :
  - `filename` doit matcher `^[A-Za-z0-9_\-.]+$` (pas de `/`, pas de `..`).
  - Le path résolu doit être strictement dans `base_dir.resolve()`.
  - Rejet → `ValueError` avec message FR.
- **R6** — `parse_command(text)` :
  - `text` doit commencer par `/` suivi d'un nom de commande dans `KNOWN_COMMANDS`.
  - Retour `(command, args)` où args est la liste des tokens après la commande.
  - Commande inconnue (ex `/drop`) → `None` (le caller décide quoi faire).
- **R7** — `TelegramChannel.send` POST sur `https://api.telegram.org/bot{token}/sendMessage` avec body JSON. Status 2xx → OK. Sinon → `ChannelSendError`.
- **R8** — `WhatsAppChannel.send` POST sur `https://graph.facebook.com/v21.0/{phone_number_id}/messages` avec body JSON Meta Cloud API. Header `Authorization: Bearer {access_token}`.
- **R9** — `TelegramChannel.parse_webhook(payload)` lit `payload['message']['text']` + `from.id` + `chat.id`. Retour `None` si payload ne ressemble pas à un message texte (ex: update de callback_query). Jamais d'exception.
- **R10** — `WhatsAppChannel.parse_webhook(payload)` lit `payload['entry'][0]['changes'][0]['value']['messages'][0]` pour extraire `from` (phone), `text.body`. Retour `None` si pas un message texte (ex: status delivery).
- **R11** — Auth allowlist : si `allowed_user_ids` (Telegram) ou `allowed_phone_numbers` (WhatsApp) est non-None, `parse_webhook` **filtre** les messages et retourne `None` pour senders non autorisés.
- **R12** — Aucun log du **contenu complet** des messages (anti-fuite client). Logs en DEBUG uniquement, troncature 100 chars max.

---

## 5. Edge cases

- **EC1** — `MessageBus.publish_outbound` sans channel enregistré → `ChannelNotFoundError`, pas de fallback silencieux.
- **EC2** — Handler inbound qui raise exception → log warning, les autres handlers exécutés.
- **EC3** — Rate limiter `TokenBucket(0, 1)` (rate 0) → bloque indéfiniment (détecté en tests via timeout).
- **EC4** — `safe_download_path("../etc/passwd", Path("/tmp"))` → `ValueError`.
- **EC5** — `safe_download_path("file.txt", Path("/tmp"))` → `Path("/tmp/file.txt")`.
- **EC6** — `parse_command("")` → `None`.
- **EC7** — `parse_command("/unknown foo")` → `None`.
- **EC8** — `parse_command("/status")` → `("/status", [])`.
- **EC9** — `parse_command("/new mon sujet long")` → `("/new", ["mon", "sujet", "long"])`.
- **EC10** — `TelegramChannel.parse_webhook({})` → `None` (payload vide).
- **EC11** — `TelegramChannel.parse_webhook({"callback_query": ...})` → `None` (pas un message texte).
- **EC12** — `WhatsAppChannel.parse_webhook` avec `statuses` au lieu de `messages` → `None` (delivery notification, pas un message).
- **EC13** — `allowed_user_ids={42}` et webhook de user `99` → `parse_webhook` retourne `None` (filtré).
- **EC14** — `TelegramChannel.send` timeout httpx → propagé comme `ChannelSendError` avec cause.
- **EC15** — `send` avec réponse 401 → `ChannelAuthError` (sous-classe).

---

## 6. Exemples concrets

### Setup MessageBus avec Telegram + WhatsApp

```python
from wincorp_odin.messaging import MessageBus, TelegramChannel, WhatsAppChannel, TokenBucket

bus = MessageBus()

tg = TelegramChannel(
    bot_token=os.environ["TELEGRAM_BOT_TOKEN"],
    allowed_user_ids={123456, 789012},
    rate_limit=TokenBucket(rate_per_second=25, capacity=25),
)
wa = WhatsAppChannel(
    phone_number_id=os.environ["WHATSAPP_PHONE_ID"],
    access_token=os.environ["WHATSAPP_ACCESS_TOKEN"],
    allowed_phone_numbers={"+33671210925"},
    rate_limit=TokenBucket(rate_per_second=80, capacity=80),
)

bus.register_channel(tg)
bus.register_channel(wa)

async def on_message(msg: InboundMessage) -> None:
    cmd = parse_command(msg.text)
    if cmd:
        # dispatch...

bus.register_handler(on_message)
await bus.start_all()
```

### Envoyer une notif user

```python
await bus.publish_outbound(OutboundMessage(
    channel_name="telegram",
    recipient_id="123456",
    text="Session Claude Code terminée — 14 commits pushés.",
))
```

### Parser webhook Telegram

```python
payload = {"update_id": 1, "message": {"from": {"id": 42}, "chat": {"id": 42}, "text": "/status"}}
msg = TelegramChannel.parse_webhook(payload)
# InboundMessage(channel_name="telegram", sender_id="42", chat_id="42", text="/status", ...)
```

### Parser webhook WhatsApp

```python
payload = {"entry": [{"changes": [{"value": {"messages": [
    {"from": "33671210925", "text": {"body": "status ?"}, "timestamp": "1729..."}
]}}]}]}
msg = WhatsAppChannel.parse_webhook(payload)
# InboundMessage(channel_name="whatsapp", sender_id="33671210925", ...)
```

---

## 7. Dependances & contraintes

### Techniques

- Runtime : Python >= 3.12.
- Deps : `httpx` (déjà présent dans pyproject odin).
- Pas de python-telegram-bot, pas de whatsapp-api-client (lib non-officielles = risque ban WhatsApp).
- Pas de FastAPI ici (vit dans heimdall).

### Performance

- `TokenBucket.acquire()` : < 10 µs si token dispo.
- Envoi Telegram : 25 msg/s max (officiel Bot API).
- Envoi WhatsApp Cloud API : 80 msg/s recommandé (peut monter à 250 Tier 2+).

### Securite

- Tokens (bot_token, access_token) **jamais** en clair dans les logs.
- Messages : troncature 100 chars dans les logs DEBUG (R12).
- Path traversal guard obligatoire sur filenames sortants.
- Auth allowlist recommandée (R11).

---

## 8. Changelog

| Version | Date | Modification |
|---------|------|--------------|
| 1.0 | 2026-04-22 | Creation initiale — Phase 6 DeerFlow (partielle) + extension WhatsApp user. |
