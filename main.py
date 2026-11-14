"""Petit relais entre l'Article Admin (navigateur) et Groq.

Le navigateur envoie un texte ici, ce serveur appelle Groq avec la clé (qui reste
côté serveur) et renvoie la réponse. Il autorise uniquement l'origine de l'Article Admin
(CORS) et exige un jeton secret.

Variables d'environnement :
  GROQ_API_KEY     clé Groq (obligatoire)
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


def groq_headers() -> dict:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise HTTPException(500, "GROQ_API_KEY non configuré sur le serveur")
    return {"Authorization": f"Bearer {key}"}


async def call_groq(payload: dict) -> dict:
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(f"{GROQ_BASE}/chat/completions", json=payload, headers=groq_headers())
    if r.status_code != 200:
        raise HTTPException(502, f"Groq a répondu {r.status_code} : {r.text[:300]}")
    return r.json()


async def list_groq_models() -> list[str]:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{GROQ_BASE}/models", headers=groq_headers())
    if r.status_code != 200:
        raise HTTPException(502, f"Groq a répondu {r.status_code} : {r.text[:300]}")
    return sorted(m["id"] for m in r.json().get("data", []))


@app.get("/health")
async def health():
    return {"ok": True}


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
