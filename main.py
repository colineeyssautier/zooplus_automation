"""Petit relais entre l'Article Admin (navigateur) et Groq.

Le navigateur envoie un texte ici, ce serveur appelle Groq avec la clé (qui reste
côté serveur) et renvoie la réponse. Il autorise uniquement l'origine de l'Article Admin
(CORS) et exige un jeton secret.

Variables d'environnement :
  GROQ_API_KEY     clé Groq (obligatoire, la première)
  GROQ_API_KEY_2, GROQ_API_KEY_3 ... GROQ_API_KEY_9   clés de secours (facultatif) : dès qu'une
                   clé n'a plus de crédit ou atteint sa limite, le relais passe à la suivante
  GROQ_API_KEYS    autre façon de donner plusieurs clés, séparées par des virgules (facultatif)
  RELAY_TOKEN      jeton secret partagé avec l'outil du navigateur (obligatoire)
  ALLOWED_ORIGINS  origines autorisées, séparées par des virgules
                   (défaut : https://article-admin.private.zooplus.net)
  GROQ_MODEL       modèle par défaut (défaut : openai/gpt-oss-120b)
  GROQ_REASONING_EFFORT  facultatif : réglage de la réflexion du modèle (ex. low),
                   utilisé quand la requête n'en précise pas
"""
from __future__ import annotations

import hmac
import os
import time
from typing import Literal

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

GROQ_BASE = "https://api.groq.com/openai/v1"

ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get("ALLOWED_ORIGINS", "https://article-admin.private.zooplus.net").split(",")
    if o.strip()
]

app = FastAPI(title="Groq relay", docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Relay-Token"],
    allow_credentials=False,
    max_age=600,
)


class Msg(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(max_length=20000)


class ChatIn(BaseModel):
    messages: list[Msg] = Field(min_length=1, max_length=10)
    model: str | None = Field(default=None, max_length=80)
    temperature: float = Field(default=0.4, ge=0, le=1.5)
    max_tokens: int = Field(default=3000, ge=1, le=8000)
    reasoning_effort: str | None = Field(default=None, max_length=20)


def check_token(token: str | None) -> None:
    expected = os.environ.get("RELAY_TOKEN")
    if not expected:
        raise HTTPException(500, "RELAY_TOKEN non configuré sur le serveur")
    if not token or not hmac.compare_digest(token, expected):
        raise HTTPException(401, "jeton invalide")


def load_keys() -> list[str]:
    """Toutes les clés Groq configurées, dans l'ordre, sans doublon."""
    raw: list[str] = []
    raw += [k for k in os.environ.get("GROQ_API_KEYS", "").replace("\n", ",").split(",")]
    raw.append(os.environ.get("GROQ_API_KEY", ""))
    for i in range(2, 10):
        raw.append(os.environ.get(f"GROQ_API_KEY_{i}", ""))
    keys: list[str] = []
    for k in raw:
        k = k.strip()
        if k and k not in keys:
            keys.append(k)
    return keys


# clé -> instant (time.time) jusqu'auquel on évite de l'utiliser
_cooldown: dict[str, float] = {}


def _pause_for(status: int, headers: dict, text: str) -> float:
    """Durée de mise en pause d'une clé selon la réponse de Groq."""
    low = (text or "").lower()
    if status in (401, 403):
        return 24 * 3600  # clé refusée ou révoquée
    if status == 402:
        return 6 * 3600  # plus de crédit
    # 429 : limite atteinte
    retry = headers.get("retry-after")
    try:
        wait = float(retry) if retry is not None else 0.0
    except ValueError:
        wait = 0.0
    if "per day" in low or "tokens per day" in low or "requests per day" in low or "quota" in low or "billing" in low:
        return max(wait, 1800.0)
    return max(wait, 60.0)


def key_order(now: float | None = None) -> list[str]:
    """Clés prêtes d'abord (dans l'ordre configuré), puis celles en pause (la plus proche de la reprise d'abord)."""
    now = time.time() if now is None else now
    keys = load_keys()
    ready = [k for k in keys if _cooldown.get(k, 0) <= now]
    waiting = sorted((k for k in keys if _cooldown.get(k, 0) > now), key=lambda k: _cooldown[k])
    return ready + waiting


async def _post_groq(key: str, payload: dict) -> tuple[int, dict, str]:
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            f"{GROQ_BASE}/chat/completions", json=payload, headers={"Authorization": f"Bearer {key}"}
        )
    return r.status_code, {k.lower(): v for k, v in r.headers.items()}, r.text


async def _get_groq_models(key: str) -> tuple[int, dict, str]:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{GROQ_BASE}/models", headers={"Authorization": f"Bearer {key}"})
    return r.status_code, {k.lower(): v for k, v in r.headers.items()}, r.text


async def _with_rotation(send):
    """Appelle `send(clé)` avec chaque clé à tour de rôle jusqu'à une réponse 200.

    Une clé sans crédit (402), refusée (401/403) ou limitée (429) est mise en pause et on passe à la suivante.
    Les autres erreurs de Groq sont renvoyées telles quelles.
    """
    keys = key_order()
    if not keys:
        raise HTTPException(500, "aucune clé Groq configurée sur le serveur (GROQ_API_KEY)")
    tried: list[str] = []
    for n, key in enumerate(keys, start=1):
        status, headers, text = await send(key)
        if status == 200:
            return text
        if status in (401, 402, 403, 429):
            _cooldown[key] = time.time() + _pause_for(status, headers, text)
            tried.append(f"clé {n} : {status}")
            continue
        raise HTTPException(502, f"Groq a répondu {status} : {text[:300]}")
    raise HTTPException(
        503,
        "toutes les clés Groq sont épuisées, limitées ou refusées (" + " ; ".join(tried) + "). "
        "Réessaie dans quelques minutes ou ajoute une clé.",
    )


async def _call_groq_real(payload: dict) -> dict:
    import json as _json

    text = await _with_rotation(lambda key: _post_groq(key, payload))
    return _json.loads(text)


async def _list_groq_models_real() -> list[str]:
    import json as _json

    text = await _with_rotation(_get_groq_models)
    return sorted(m["id"] for m in _json.loads(text).get("data", []))


call_groq = _call_groq_real
list_groq_models = _list_groq_models_real


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/keys")
async def keys_status(x_relay_token: str | None = Header(default=None)):
    """État des clés (sans jamais les afficher) : combien sont prêtes, lesquelles sont en pause."""
    check_token(x_relay_token)
    now = time.time()
    out = []
    for n, k in enumerate(load_keys(), start=1):
        wait = max(0, int(_cooldown.get(k, 0) - now))
        out.append({"cle": n, "etat": "prete" if wait == 0 else "en pause", "reprise_dans_secondes": wait})
    return {"nombre": len(out), "cles": out}


@app.get("/models")
async def models(x_relay_token: str | None = Header(default=None)):
    check_token(x_relay_token)
    return {"models": await list_groq_models()}


@app.post("/chat")
async def chat(body: ChatIn, x_relay_token: str | None = Header(default=None)):
    check_token(x_relay_token)
    payload = {
        "model": body.model or os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"),
        "messages": [m.model_dump() for m in body.messages],
        "temperature": body.temperature,
        "max_tokens": body.max_tokens,
    }
    effort = body.reasoning_effort or os.environ.get("GROQ_REASONING_EFFORT")
    if effort:
        payload["reasoning_effort"] = effort
    data = await call_groq(payload)
    try:
        choice = data["choices"][0]
        content = choice["message"].get("content") or ""
        finish = choice.get("finish_reason")
    except (KeyError, IndexError, AttributeError):
        raise HTTPException(502, "réponse Groq inattendue")
    if not content.strip():
        raise HTTPException(
            502,
            "réponse vide (finish_reason=%s) : le modèle a probablement utilisé tous ses "
            "tokens pour réfléchir. Augmenter max_tokens ou baisser reasoning_effort." % finish,
        )
    return {
        "content": content,
        "model": data.get("model", payload["model"]),
        "finish_reason": finish,
        "truncated": finish == "length",
        "usage": data.get("usage"),
    }