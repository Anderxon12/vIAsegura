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


class _MotorLigero:
    """El mismo modelo ligero que fastembed (mismo ONNX, mismos vectores), pero con mucha menos
    memoria: ~300 MB en lugar de ~640 MB. Tokeniza con sentencepiece (~45 MB; el tokenizador de
    fastembed ocupa ~260 MB con este vocabulario de 250 000 piezas) y abre la sesión de
    onnxruntime sin la reserva de memoria por adelantado. Pensado para servidores pequeños
    (Render gratis: 512 MB)."""
    REPO_ONNX, ARCHIVO_ONNX = "Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q", "model_optimized.onnx"
    REPO_SP, ARCHIVO_SP = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "sentencepiece.bpe.model"
    MAX_TOKENS = 128                    # el mismo truncado que fastembed para este modelo

    DIR_EXTERNO = os.path.join(CACHE_FASTEMBED, "ligero_pesos_externos")

    @classmethod
    def descargar(cls, separar_pesos: bool = True) -> tuple[str, str]:
        """Baja el modelo (si falta) y, si está la librería onnx, guarda una copia con los pesos en un
        archivo aparte. onnxruntime mapea ese archivo en memoria en vez de copiarlo: la tabla de
        embeddings (192 MB) deja de contar como memoria propia del proceso y solo se leen las filas
        de las palabras que aparecen. Devuelve (ruta del ONNX a usar, ruta del sentencepiece)."""
        from huggingface_hub import hf_hub_download
        os.makedirs(CACHE_FASTEMBED, exist_ok=True)
        ruta_sp = hf_hub_download(cls.REPO_SP, cls.ARCHIVO_SP, cache_dir=CACHE_FASTEMBED)
        externo = os.path.join(cls.DIR_EXTERNO, "modelo.onnx")
        if os.path.exists(externo) and os.path.exists(externo + ".data"):
            return externo, ruta_sp
        ruta_onnx = hf_hub_download(cls.REPO_ONNX, cls.ARCHIVO_ONNX, cache_dir=CACHE_FASTEMBED)
        if separar_pesos:
            try:
                import onnx
            except ImportError:
                return ruta_onnx, ruta_sp
            os.makedirs(cls.DIR_EXTERNO, exist_ok=True)
            m = onnx.load(ruta_onnx)
            onnx.save_model(m, externo + ".tmp", save_as_external_data=True, all_tensors_to_one_file=True,
                            location="modelo.onnx.data", size_threshold=1024)
            del m
            os.replace(externo + ".tmp", externo)
            return externo, ruta_sp
        return ruta_onnx, ruta_sp

    def __init__(self):
        import onnxruntime as ort
        import sentencepiece as spm
        # La conversión (una sola vez) usa ~500 MB por un momento; en Render se hace en el build
        # (src/preparar_render.py), así que al arrancar el servidor ya está lista.
        ruta_onnx, ruta_sp = self.descargar()
        self.sp = spm.SentencePieceProcessor(model_file=ruta_sp)
        o = ort.SessionOptions()
        o.enable_cpu_mem_arena = False
        o.enable_mem_pattern = False
        o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL   # el modelo ya viene optimizado
        o.intra_op_num_threads = int(os.environ.get("RAG_HILOS_ONNX", 1))
        self.s = ort.InferenceSession(ruta_onnx, o, providers=["CPUExecutionProvider"])
        self.entradas = {i.name for i in self.s.get_inputs()}
        self.dimension = int(self.codificar(["prueba"]).shape[1])

    def _ids(self, texto: str) -> list[int]:
        # Igual que el tokenizador XLM-R: <s>=0 <pad>=1 </s>=2 <unk>=3; las piezas se corren 1 lugar.
        piezas = [i + 1 if i else 3 for i in self.sp.encode(texto)][: self.MAX_TOKENS - 2]
        return [0] + piezas + [2]

    def codificar(self, textos: list[str]) -> np.ndarray:
        salida = []
        for i in range(0, len(textos), 16):
            lote = [self._ids(t) for t in textos[i:i + 16]]
            largo = max(len(x) for x in lote)
            ids = np.ones((len(lote), largo), np.int64)               # 1 = <pad>
            mascara = np.zeros((len(lote), largo), np.int64)
            for j, x in enumerate(lote):
                ids[j, :len(x)], mascara[j, :len(x)] = x, 1
            feed = {"input_ids": ids, "attention_mask": mascara}
            if "token_type_ids" in self.entradas:
                feed["token_type_ids"] = np.zeros_like(ids)
            h = self.s.run(None, feed)[0]
            m = mascara[..., None].astype(h.dtype)
            salida.append((h * m).sum(1) / np.maximum(m.sum(1), 1e-9))   # promedio de los tokens reales
        v = np.concatenate(salida).astype("float32")
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


class _MotorFastEmbed:
    def __init__(self, modelo: str):
        if modelo == MODELO_LIGERO:
            try:
                self.m = _MotorLigero()
                self.dimension = self.m.dimension
                return
            except ImportError:             # falta sentencepiece u onnxruntime: se usa fastembed tal cual
                pass
        from fastembed import TextEmbedding
        os.makedirs(CACHE_FASTEMBED, exist_ok=True)
        self.m = TextEmbedding(model_name=modelo, cache_dir=CACHE_FASTEMBED, threads=None)
        self.dimension = int(self.codificar(["prueba"]).shape[1])

    def codificar(self, textos: list[str]) -> np.ndarray:
        if isinstance(self.m, _MotorLigero):
            return self.m.codificar(textos)
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
