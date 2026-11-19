"""Configuración del RAG, leída del archivo .env en la raíz del proyecto."""

from __future__ import annotations

import os
from dataclasses import dataclass

import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(RAIZ, "src"))
from entorno import cargar_entorno  # noqa: E402  (.env de la raíz o de src/)

cargar_entorno()


def _ruta(valor: str) -> str:
    return valor if os.path.isabs(valor) else os.path.join(RAIZ, valor)


def _num(nombre: str, defecto, tipo=int):
    try:
        return tipo(os.environ.get(nombre, defecto))
    except ValueError:
        return tipo(defecto)


@dataclass(frozen=True)
class Config:
    # documentos e índice
    carpeta_documentos: str
    carpeta_indice: str
    # embeddings
    modelo_embeddings: str
    motor_embeddings: str       # auto | sentence-transformers | fastembed
    tam_fragmento: int          # caracteres por fragmento
    solape: int                 # caracteres repetidos entre fragmentos vecinos
    # recuperación
    top_k: int
    similitud_minima: float     # por debajo de esto un fragmento no se usa como contexto
    # LLM
    anthropic_api_key: str
    modelo_llm: str
    max_tokens: int
    temperatura: float

    @property
    def hay_llm(self) -> bool:
        k = self.anthropic_api_key
        return bool(k) and "REEMPLAZAR" not in k.upper()


def cargar_config() -> Config:
    return Config(
        carpeta_documentos=_ruta(os.environ.get("RAG_CARPETA_DOCUMENTOS", "data/resoluciones")),
        carpeta_indice=_ruta(os.environ.get("RAG_CARPETA_INDICE", "out/rag")),
        modelo_embeddings=os.environ.get("RAG_MODELO_EMBEDDINGS", "intfloat/multilingual-e5-small"),
        motor_embeddings=os.environ.get("RAG_MOTOR_EMBEDDINGS", "auto"),
        tam_fragmento=_num("RAG_TAM_FRAGMENTO", 1200),
        solape=_num("RAG_SOLAPE", 200),
        top_k=_num("RAG_TOP_K", 5),
        similitud_minima=_num("RAG_SIMILITUD_MINIMA", 0.70, float),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", "").strip().strip('"').strip("'"),
        modelo_llm=os.environ.get("ANTHROPIC_MODELO", "claude-sonnet-5-5"),
        max_tokens=_num("ANTHROPIC_MAX_TOKENS", 1024),
        temperatura=_num("ANTHROPIC_TEMPERATURA", 0.2, float),
    )
