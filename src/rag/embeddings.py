"""
3. EMBEDDINGS
Convierte textos en vectores normalizados (largo 1), así el producto punto es
la similitud coseno. Hay dos motores, se elige con RAG_MOTOR_EMBEDDINGS en .env:

  sentence-transformers  PyTorch. Modelo por defecto intfloat/multilingual-e5-small
                         (pide prefijos «passage: » / «query: »). Necesita ~2 GB de
                         memoria libre para cargar.
  fastembed              ONNX Runtime, sin PyTorch, modelo cuantizado (int8):
                         sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2,
                         ~220 MB. Para equipos con poca memoria.
  auto (por defecto)     usa sentence-transformers si hay memoria libre suficiente
                         (≥ 2,5 GB de RAM y ≥ 3 GB de memoria virtual); si no, o si
                         Windows falla al cargarlo (error 1455 «archivo de paginación
                         demasiado pequeño»), usa fastembed. Así la app no deja el
                         equipo sin memoria.

La primera vez el modelo se descarga de Hugging Face y queda en caché.
"""

from __future__ import annotations

import os

import numpy as np

from .config import RAIZ

MODELO_LIGERO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CACHE_FASTEMBED = os.path.join(RAIZ, "out", "modelos_embeddings")


def memoria_libre_gb() -> tuple[float, float] | None:
    """(RAM disponible, memoria virtual disponible) en GB, o None si no se puede saber."""
    try:
        import psutil
        vm, sw = psutil.virtual_memory(), psutil.swap_memory()
        return vm.available / 1e9, (vm.available + sw.free) / 1e9
    except ImportError:
        pass
    try:
        if os.name == "nt":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return st.ullAvailPhys / 1e9, st.ullAvailPageFile / 1e9
        libre = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9
        return libre, libre
    except Exception:
        return None


def _sin_memoria(e: Exception) -> bool:
    texto = str(e).lower()
    return isinstance(e, MemoryError) or getattr(e, "winerror", None) == 1455 or "1455" in texto or \
        "paginaci" in texto or "paging file" in texto or "not enough memory" in texto or "out of memory" in texto


class _MotorST:
    def __init__(self, modelo: str):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(modelo, device="cpu")
        self.dimension = self.m.get_sentence_embedding_dimension()

    def codificar(self, textos: list[str]) -> np.ndarray:
        return self.m.encode(textos, normalize_embeddings=True, batch_size=16,
                             show_progress_bar=len(textos) > 50, convert_to_numpy=True)


class _MotorFastEmbed:
    def __init__(self, modelo: str):
        from fastembed import TextEmbedding
        os.makedirs(CACHE_FASTEMBED, exist_ok=True)
        self.m = TextEmbedding(model_name=modelo, cache_dir=CACHE_FASTEMBED, threads=None)
        self.dimension = int(self.codificar(["prueba"]).shape[1])

    def codificar(self, textos: list[str]) -> np.ndarray:
        v = np.asarray(list(self.m.embed(textos, batch_size=16)), dtype="float32")
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


class GeneradorEmbeddings:
    def __init__(self, modelo: str, motor: str = "auto"):
        motor = (motor or "auto").lower()
        self.aviso = ""
        mem = memoria_libre_gb() if motor == "auto" else None
        if mem and (mem[0] < 2.5 or mem[1] < 3.0):
            self.aviso = (f"Hay poca memoria libre ({mem[0]:.1f} GB de RAM); se usa el modelo ligero "
                          f"{MODELO_LIGERO} con fastembed (ONNX) en lugar de PyTorch.")
            print("  ℹ️ " + self.aviso)
            motor = "fastembed"
            modelo = MODELO_LIGERO
        if motor == "fastembed":
            modelo = modelo if "e5-small" not in modelo else MODELO_LIGERO     # e5-small no existe en fastembed
            self._motor, self.motor = _MotorFastEmbed(modelo), "fastembed"
        else:
            try:
                self._motor, self.motor = _MotorST(modelo), "sentence-transformers"
            except (OSError, MemoryError, RuntimeError) as e:
                if motor != "auto" or not _sin_memoria(e):
                    raise
                self.aviso = ("No hubo memoria para cargar el modelo con PyTorch; se usa el modelo ligero "
                              f"{MODELO_LIGERO} con fastembed (ONNX).")
                print("  ⚠️ " + self.aviso)
                modelo = MODELO_LIGERO
                self._motor, self.motor = _MotorFastEmbed(modelo), "fastembed"
        self.modelo = modelo
        self.nombre = f"{self.motor}:{modelo}"          # se guarda en el índice para no mezclar modelos
        self.e5 = "e5" in modelo.lower()
        self.dimension = self._motor.dimension

    def _codificar(self, textos: list[str], prefijo: str) -> np.ndarray:
        if not textos:
            return np.zeros((0, self.dimension), "float32")
        if self.e5:
            textos = [prefijo + t for t in textos]
        return np.asarray(self._motor.codificar(textos), dtype="float32")

    def documentos(self, textos: list[str]) -> np.ndarray:
        return self._codificar(textos, "passage: ")

    def consulta(self, texto: str) -> np.ndarray:
        return self._codificar([texto], "query: ")[0]
