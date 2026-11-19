"""
5. LLM (Claude)
Recibe la pregunta y los fragmentos recuperados y redacta la respuesta en
español, usando SOLO ese contexto y citando cada dato con [n]. Si el
contexto no alcanza, lo dice en vez de inventar.

Sin ANTHROPIC_API_KEY funciona en modo extractivo: devuelve los fragmentos
más relevantes tal cual, para poder probar la recuperación sin gastar API.
"""

from __future__ import annotations

from .fragmentos import Fragmento

INSTRUCCIONES = """Eres el asistente normativo de VíaSegura IA. Respondes preguntas sobre normas de tránsito y \
seguridad vial de Bogotá usando ÚNICAMENTE los fragmentos de normas que se te entregan.

Reglas:
- Responde en español, claro y breve (máximo 2 o 3 párrafos o una lista corta).
- Cada afirmación debe llevar la cita del fragmento del que sale, así: [1], [2].
- Si los fragmentos no contienen la respuesta, dilo literalmente: «No encuentro eso en las normas cargadas» y \
sugiere qué norma habría que cargar. No uses conocimiento externo ni inventes artículos, fechas o cifras.
- Si un fragmento indica que la norma está derogada o compilada en otra, adviértelo.
- No des asesoría legal: es información de apoyo."""


def armar_contexto(recuperados: list[tuple[Fragmento, float]]) -> str:
    bloques = []
    for n, (f, _) in enumerate(recuperados, 1):
        m = f.metadatos
        cab = f"[{n}] {f.titulo} — {f.seccion}"
        extra = [x for x in (m.get("fecha_expedicion") and f"expedida {m['fecha_expedicion']}", m.get("estado")) if x]
        if extra:
            cab += " (" + "; ".join(extra) + ")"
        bloques.append(f"{cab}\n{f.texto}")
    return "\n\n---\n\n".join(bloques)


class GeneradorClaude:
    def __init__(self, api_key: str, modelo: str, max_tokens: int = 1024, temperatura: float = 0.2):
        import anthropic
        self.cliente = anthropic.Anthropic(api_key=api_key)
        self.modelo, self.max_tokens, self.temperatura = modelo, max_tokens, temperatura
        try:                                # el SDK más reciente de anthropic no tiene el parámetro «temperature»
            import inspect
            parametros = inspect.signature(self.cliente.messages.create).parameters
            self._acepta_temperatura = "temperature" in parametros or any(
                p.kind == p.VAR_KEYWORD for p in parametros.values())
        except (TypeError, ValueError):
            self._acepta_temperatura = True

    def responder(self, pregunta: str, recuperados: list[tuple[Fragmento, float]]) -> str:
        if not recuperados:
            return "No encuentro eso en las normas cargadas."
        mensaje = (f"<fragmentos>\n{armar_contexto(recuperados)}\n</fragmentos>\n\n"
                   f"Pregunta: {pregunta}")
        args = dict(model=self.modelo, max_tokens=self.max_tokens, system=INSTRUCCIONES,
                    messages=[{"role": "user", "content": mensaje}])
        if self._acepta_temperatura:
            args["temperature"] = self.temperatura
        try:
            r = self.cliente.messages.create(**args)
        except TypeError as e:              # versiones del SDK que ya no reciben «temperature»
            if "temperature" not in str(e):
                raise
            self._acepta_temperatura = False
            args.pop("temperature", None)
            r = self.cliente.messages.create(**args)
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text").strip()


def explicar_error(e: Exception) -> str:
    """Mensaje claro para los errores más comunes al llamar a Claude."""
    nombre, texto = type(e).__name__, str(e)
    if nombre == "AuthenticationError":
        return "la clave ANTHROPIC_API_KEY no es válida (revísela en console.anthropic.com)."
    if nombre == "PermissionDeniedError" or "credit balance" in texto.lower():
        return "la cuenta de Anthropic no tiene saldo o permiso para este modelo."
    if nombre == "NotFoundError":
        return "el modelo de ANTHROPIC_MODELO no existe o su cuenta no tiene acceso (revise el nombre en console.anthropic.com)."
    if nombre == "RateLimitError":
        return "se superó el límite de solicitudes; espere un momento."
    if nombre in ("APIConnectionError", "APITimeoutError"):
        return "no hay conexión con la API de Anthropic."
    if nombre == "TypeError" and "unexpected keyword argument" in texto:
        return f"la versión instalada de la librería anthropic no reconoce un parámetro ({texto[:160]})."
    if nombre == "TypeError":
        return ("incompatibilidad de librerías con su versión de Python. Ejecute: "
                "python -m pip install -U anthropic pydantic httpx  —  y reinicie la app. "
                f"(Detalle: {texto[:160]})")
    return f"{nombre}: {texto[:200]}"


class GeneradorExtractivo:
    """Respaldo sin API: muestra los fragmentos encontrados, sin redactar."""

    def __init__(self, nota: str | None = None):
        self.nota = nota or "(Modo sin LLM: configure ANTHROPIC_API_KEY en .env para obtener una respuesta redactada.)"

    def responder(self, pregunta: str, recuperados: list[tuple[Fragmento, float]]) -> str:
        if not recuperados:
            return "No encuentro eso en las normas cargadas."
        partes = [self.nota, "Fragmentos más relevantes:"]
        for n, (f, s) in enumerate(recuperados[:3], 1):
            texto = f.texto if len(f.texto) <= 500 else f.texto[:500].rsplit(" ", 1)[0] + "…"
            partes.append(f"[{n}] {f.titulo} — {f.seccion} (similitud {s:.2f})\n{texto}")
        return "\n\n".join(partes)
