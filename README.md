# trigger-reply-engine

The **World Simulator** for the RL gyms. A stateless FastAPI service that
simulates realistic prospect behavior — email replies, chat replies,
unsubscribes, meeting responses — by calling a target gym's API. The gym backend
stays the single owner of business logic and the database, so simulator-driven
and user-driven changes behave identically and both show up in the gym UI.

It is **manifest-driven**: each gym is described by a YAML file under
[`manifests/`](manifests/). Adding a new gym means adding a file — no code change.

## How it works

```
POST /api/trigger {gym_name, action, session_id, params}
        │
        ▼
   Engine:  validate → generate (LLM) → render steps → execute each in order → normalize
        │                     │                  │                   │
   Registry (manifests)   Generator (LLM/stub)  shared now/          HTTP client (Cookie:
        │                                        correlation_id       user_id=<session_id>)
   manifests/*.yaml  ──►  target gym API (1 call for most actions, N for a multi-step one)
```

- **Called, not automatic.** The service reacts when something calls `POST /api/trigger`
  (e.g. an RL harness right after the agent acts). It does not watch the gym.
- **Session binding.** Every gym isolates state by a `user_id` cookie and 404s
  without it. The engine sends the request's `session_id` as that cookie, so the
  write lands in the correct session's database. The engine never creates
  sessions — the harness does that once (a cookieless `GET /api/state` on the gym)
  and shares the `user_id` with both the agent and this service.
- **Replies are generated.** Fields listed as `generates` in a manifest are filled
  by the reply generator: a deterministic template stub by default, or OpenAI when
  `USE_LLM=1`. LLM failures degrade to the stub, so a trigger never hard-fails on
  the model.
- **Stateless.** No database; the event log is structured stdout logging keyed by
  `trigger_id` (and the caller's `episode_id`).
- **Multi-step actions.** An action is one HTTP call by default, but a manifest can
  declare `steps: [...]` instead of `request` when a UI action really issues several
  calls (e.g. sales_explorer's `chat.send`: create a message, patch the conversation,
  log a timeline activity). All steps in one trigger share a single `{{ now }}` and
  `{{ correlation_id }}`, so ids/timestamps referenced by more than one step match —
  the caller doesn't need to invent or repeat them. Steps run in order and stop at the
  first failure (no rollback); see [Adding a new gym](#adding-a-new-gym-no-code-change).

## API

Every route lives under `/api`. `POST /api/trigger` and the `/api/gyms*`
introspection routes require the `X-API-Key` header when `API_KEY` is set
(see [Configuration](#configuration)); `/api/health` and the docs do not.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/trigger` | Simulate one world reaction against a gym |
| `GET`  | `/api/gyms` | List available gyms |
| `GET`  | `/api/gyms/{name}/actions` | List a gym's supported actions |
| `GET`  | `/api/gyms/{name}/actions/{action}/schema` | Inputs, required fields, generated fields |
| `GET`  | `/api/health` | Liveness probe |
| `GET`  | `/api/docs` | Swagger UI (`/api/redoc`, `/api/openapi.json`) |

### `POST /api/trigger`

```jsonc
{
  "gym_name": "sales_explorer",
  "action": "chat.reply",
  "session_id": "8f3c...ab",        // REQUIRED — sent as the gym's user_id cookie
  "params": {
    "conversation_id": "c1", "sender": "lead-9",
    // Reserved key: per-step JSON body overrides for a multi-step action,
    // keyed by step name (e.g. sales_explorer chat.send: message/conversation/
    // timeline). Ignored (harmless) for single-request actions. Merges onto
    // the manifest-rendered body — it doesn't replace it — so this is only
    // needed for fields the manifest doesn't already template.
    "steps": { "conversation": { "contactName": "Anna Rogers" } }
  },
  "idempotency_key": "optional-unique-key",
  "dry_run": false,                  // true = validate + build request(s), no LLM/gym call
  "seed": 42,                        // seeds deterministic reply generation
  "episode_id": "optional-correlation-id"
}
```

Response — `results`/`request_previews` are always lists, one entry per step of the
action (length 1 for an ordinary single-request action, N for a multi-step one):

```jsonc
{
  "status": "ok",
  "gym_name": "sales_explorer",
  "action": "chat.reply",
  "trigger_id": "46ad2c7f...",
  "generated": { "content": "Thanks for the message! Happy to connect..." },
  "results": [
    { "step": "request", "method": "POST", "gym_status": 200, "body": { "id": "m5" } }
  ]
}
```

Errors use a uniform envelope: `{"status":"error","error":{"code","message","details"}}`.
`unknown_gym`/`unknown_action` → 404, `param_validation_error` → 422, `gym_error` → 502,
`gym_unavailable` → 504. A `gym_error` includes the gym's `gym_status`/`gym_body`; for a
multi-step action it also names the failed `step`/`step_index` and lists `completed`
(the `results` of steps that already succeeded — there is no rollback, matching the real
UI's own fire-and-forget multi-call flows). Pass an `idempotency_key` on multi-step
actions so a client retry after a partial failure replays the first attempt's outcome
instead of re-running already-succeeded steps.

### Example (curl)

```bash
# Dry-run: see exactly what would be sent, without calling the gym or the LLM.
curl -sS localhost:8000/api/trigger -H 'content-type: application/json' \
  -H 'X-API-Key: your-key' -d '{
  "gym_name":"spothub","action":"email.reply","session_id":"sess-1",
  "params":{"contact_id":"123"},"dry_run":true }' | jq
# The -H 'X-API-Key' header is only required once API_KEY is set (see Configuration).
```

## Adding a new gym (no code change)

1. Create `manifests/<gym>.yaml`, with `base_url` as a literal default (the gym
   works with zero configuration — no env var required):

```yaml
name: my_gym
base_url: https://my-gym.example.com   # default; see the override note below
session:
  cookie: user_id
actions:
  # The common case: one action = one HTTP call.
  chat.reply:
    description: A prospect replies in a thread.
    generates: [content]         # filled by the LLM/stub when omitted
    inputs:
      conversation_id: { type: string, required: true }
      sender:          { type: string, required: true }
      content:         { type: string, required: false }
    request:
      method: POST
      path: /api/conversations/{{ conversation_id }}/messages
      json:
        sender:  "{{ sender }}"
        content: "{{ content }}"
        sentAt:  "{{ now }}"     # helpers: {{ now }}, {{ today }}, {{ correlation_id }}
        isRead:  false

  # Use `steps` instead of `request` when the real UI issues several calls
  # for one action. `{{ now }}`/`{{ correlation_id }}` are identical across
  # every step of one trigger, so ids/timestamps that must match (e.g. a
  # message id echoed into a later timeline call) don't need the caller's
  # help. `params.steps.<name>` (see the API example above) can add fields
  # to any one step's body without the manifest templating every field.
  chat.send:
    description: Send a message; log it as a timeline activity too.
    generates: [content]
    inputs:
      lead_id: { type: string, required: true }
      content: { type: string, required: false }
    steps:
      - name: message
        method: POST
        path: /api/conversations/{{ lead_id }}/messages
        json: { id: "{{ correlation_id }}", content: "{{ content }}", sentAt: "{{ now }}" }
      - name: timeline
        method: POST
        path: /api/timeline/{{ lead_id }}
        json:
          type: message_sent
          metadata: { messageId: "{{ correlation_id }}", messagePreview: "{{ content }}" }
```

2. Restart the service. It appears in `GET /api/gyms` automatically — no env
   var, no code change.
3. (Optional) To redirect this gym elsewhere (e.g. local/dev) without editing
   the manifest, set its override env var. The name is derived from `name` by
   convention — uppercase, non-alphanumerics become `_`, suffixed `_BASE_URL`
   (`app/registry.py:base_url_env_var`) — so `my_gym` is overridable via
   `MY_GYM_BASE_URL`. An explicitly-set env var always wins over the manifest
   default.

For a gym whose request templating can't express, subclass `ManifestAdapter`
and override `build_requests` (see [`app/adapters/base.py`](app/adapters/base.py)).

The two bundled manifests map to the real gym fields:
[`manifests/spothub.yaml`](manifests/spothub.yaml) (`POST /crm/v3/objects/emails`) and
[`manifests/sales_explorer.yaml`](manifests/sales_explorer.yaml) — whose `chat.send` is
the bundled multi-step example, mirroring the UI's create-message + patch-conversation +
log-timeline-activity sequence.

## Configuration

All via environment variables (see [`.env.example`](.env.example)):

| Var | Default | Purpose |
|---|---|---|
| `<GYM_NAME>_BASE_URL`, e.g. `SPOTHUB_BASE_URL`, `SALES_EXPLORER_BASE_URL` | manifest's `base_url` | **Optional override** for that gym's base URL. Derived from the manifest's `name` by convention (`app/registry.py:base_url_env_var`) — every gym, including future ones, gets one automatically. Unset = use the manifest's default. |
| `USE_LLM` | `0` | `1` = generate replies with OpenAI; else deterministic stub |
| `OPENAI_API_KEY`, `OPENAI_MODEL` | `""`, `gpt-4o-mini` | OpenAI credentials/model |
| `LLM_TIMEOUT` | `20` | Seconds before an OpenAI call times out and falls back to the stub |
| `REQUEST_TIMEOUT`, `MAX_RETRIES`, `RETRY_BACKOFF` | `15`, `2`, `0.5` | Outbound HTTP behavior |
| `IDEMPOTENCY_TTL` | `900` | Seconds an idempotent result is remembered (in-memory) |
| `IDEMPOTENCY_MAX_ENTRIES` | `10000` | Oldest keys are evicted past this, bounding cache memory |
| `API_KEY` | `""` | Shared secret required on `X-API-Key` for `/api/trigger` + introspection. Empty = no auth (local dev only) |
| `MANIFESTS_DIR` | `./manifests` | Manifest directory (set to `/app/manifests` in Docker) |

**Idempotency is per-instance.** The cache (and the lock that serializes
concurrent replays of the same key) lives in process memory — it is not
shared across instances. `cloudbuild.yaml` therefore pins `--max-instances=1`.
Scaling out safely requires moving the cache to a shared store (e.g. Redis)
first. It's especially useful for multi-step actions: a retried request after
a partial failure replays the whole trigger's original outcome rather than
re-running steps that already succeeded against the gym.

The two bundled gyms default to the hosted v1 environments (baked into their
manifests): `spothub` → `https://spothub-v5-be.rlgym.turing.com`, `sales_explorer`
→ `https://sales-explorer-v1-be.rlgym.turing.com`. Set `SPOTHUB_BASE_URL` /
`SALES_EXPLORER_BASE_URL` only to redirect them elsewhere.

## Run locally

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows; use bin/activate on *nix
pip install -e ".[dev]"        # add ",llm" too if you want USE_LLM=1 locally
uvicorn app.main:app --reload --port 8000
# Each gym's manifest default is used unless overridden. To target a local
# gym instead:
#   export SALES_EXPLORER_BASE_URL=http://localhost:3001
```

`openai` is an optional extra (`[llm]`, see [`pyproject.toml`](pyproject.toml)) —
only needed for `USE_LLM=1`; the default install/image works without it via
the deterministic template stub. [`requirements.txt`](requirements.txt) pins
exact versions (core + `llm`) for reproducible Docker builds; regenerate it
per the instructions at the top of that file after bumping a bound.

### Docker

```bash
cp .env.example .env    # gym base URLs are optional overrides; leave blank to use manifest defaults
docker compose up --build
# → http://localhost:8000/api/health
```

On Docker Desktop, reach gyms running on the host via `http://host.docker.internal:<port>`.

### Deploy (Cloud Run)

[`cloudbuild.yaml`](cloudbuild.yaml) builds, pushes, and deploys. Gym base URLs
use their manifest defaults unless overridden via the `_SPOTHUB_BASE_URL` /
`_SALES_EXPLORER_BASE_URL` substitutions; (only if `USE_LLM=1`) create a Secret
Manager secret `openai-api-key`.

## Development

```bash
pytest        # 37 tests: manifests, adapter validation/encoding/multi-step, auth, dry-run,
              #           engine vs. a mocked gym (incl. multi-step partial-failure + overrides)
ruff check app tests
mypy app
```

## Layout

```
app/
  main.py            # FastAPI app: factory + lifespan (loads registry, builds engine)
  config.py          # pydantic-settings
  schemas.py         # manifest models + API request/response models
  registry.py        # loads manifests/*.yaml -> adapters; resolves base_url
                     #   (convention-derived <NAME>_BASE_URL override, else
                     #   the manifest default) + a generic ${ENV} escape hatch
  adapters/base.py   # GymAdapter protocol + ManifestAdapter (Jinja2 rendering)
  session.py         # SessionContext + resilient execute() (cookie, retries)
  generation/        # policy.py (canned/seeded) + llm.py (stub, OpenAI, fallback)
  engine.py          # orchestration: validate → generate → render → execute → normalize
  routers/           # trigger.py, gyms.py, health.py
manifests/           # one YAML per gym  (spothub, sales_explorer)
tests/
```
