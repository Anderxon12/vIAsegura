"""
2. FRAGMENTACIÓN
Cada norma se parte primero por artículos (así una cita apunta a un artículo
concreto) y, si un artículo es largo, en ventanas de `tam` caracteres con
`solape` caracteres repetidos para no cortar ideas a la mitad.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .documentos import Documento

# «Artículo 5.», «ARTÍCULO 5°.-»; no «artículo 2.2.22.3.8» (citas de otras normas)
RE_ARTICULO = re.compile(r"(?=\b(?:ART[IÍ]CULO|Art[ií]culo)\s+\d+(?!\d|\.\d)\s*[°º]?)")


@dataclass
class Fragmento:
    id: str
    doc_id: str
    titulo: str
    seccion: str          # «Considerandos», «Artículo 5»…
    texto: str
    metadatos: dict

    def a_dict(self) -> dict:
        return asdict(self)


def _ventanas(texto: str, tam: int, solape: int) -> list[str]:
    if len(texto) <= tam:
        return [texto]
    partes, i = [], 0
    while i < len(texto):
        fin = min(i + tam, len(texto))
        if fin < len(texto):                       # cortar en el último punto o espacio
            corte = max(texto.rfind(". ", i, fin), texto.rfind(" ", i, fin))
            if corte > i + tam // 2:
                fin = corte + 1
        partes.append(texto[i:fin].strip())
        if fin >= len(texto):
            break
        i = max(fin - solape, i + 1)
    return partes


def fragmentar(doc: Documento, tam: int = 1200, solape: int = 200) -> list[Fragmento]:
    bloques = [b.strip() for b in RE_ARTICULO.split(doc.texto) if b.strip()]
    salida = []
    for bloque in bloques:
        m = re.match(r"(?:ART[IÍ]CULO|Art[ií]culo)\s+(\d+)(?!\.\d)", bloque)
        seccion = f"Artículo {m.group(1)}" if m else ("Encabezado y considerandos" if not salida else "Texto")
        for j, parte in enumerate(_ventanas(bloque, tam, solape)):
            salida.append(Fragmento(
                id=f"{doc.id}#{len(salida)}", doc_id=doc.id, titulo=doc.titulo,
                seccion=seccion + (f" (parte {j + 1})" if j else ""), texto=parte, metadatos=doc.metadatos))
    return salida
