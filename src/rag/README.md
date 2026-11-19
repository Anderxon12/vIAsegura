# RAG de normas viales de Bogotá

Responde preguntas sobre las normas cargadas en `data/resoluciones/` citando
la norma y el artículo de donde sale cada dato.

```
                    ┌──────────────── INDEXAR (una vez) ────────────────┐
data/resoluciones → │ documentos.py → fragmentos.py → embeddings.py → FAISS │ → out/rag/
   (.txt .pdf .docx)│   limpia SISJUR   por artículos    multilingual-e5    │   indice.faiss
                    └───────────────────────────────────────────────────┘   fragmentos.jsonl

                    ┌──────────────── PREGUNTAR ─────────────────────────┐
pregunta ─────────→ │ embeddings.py → FAISS (top-k, similitud ≥ mínima) → │
                    │ llm.py: Claude redacta SOLO con esos fragmentos [n]   │ → respuesta + fuentes
                    └───────────────────────────────────────────────────┘
```

| Parte | Archivo | Qué usa |
|---|---|---|
| Carga y limpieza | `documentos.py` | quita el encabezado de SISJUR, extrae fechas y si la norma está derogada; omite los archivos vacíos |
| Fragmentación | `fragmentos.py` | un fragmento por artículo; si es largo, ventanas de 1.200 caracteres con 200 de solape |
| **Embeddings** | `embeddings.py` | `sentence-transformers` con `intfloat/multilingual-e5-small` (384 dimensiones, prefijos `passage:` / `query:`) |
| **Índice vectorial** | `almacen_vectorial.py` | **FAISS** `IndexFlatIP` con vectores normalizados = similitud coseno exacta |
| **LLM** | `llm.py` | **Claude** por la API de Anthropic; sin clave, modo extractivo |
| Orquestación | `pipeline.py`, `cli.py` | `indexar`, `buscar`, `preguntar`, `chat` |

## Uso

1. `rag.bat` crea `.env` a partir de `plantilla_env.txt` la primera vez; ponga ahí su clave (`ANTHROPIC_API_KEY=...`).
2. Doble clic en `rag.bat` (instala, indexa la primera vez y abre el chat), o:

```
pip install -r requirements_rag.txt
python src/rag/cli.py indexar
python src/rag/cli.py buscar "zonas escolares"            # solo embeddings + FAISS
python src/rag/cli.py preguntar "¿Qué programas tiene el Plan Distrital de Seguridad Vial?"
python src/rag/cli.py chat
```

### Desde la app (pestaña **Normas**)
1. Arrastre los PDF (o .docx / .txt): se guardan en `data/resoluciones/subidos/` y quedan como **Pendientes**.
2. Pulse **Indexar pendientes** (o **Indexar** en un documento): lee el texto, lo parte por artículos, calcula
   los embeddings y los agrega al índice FAISS sin recalcular los demás documentos.
3. **Reindexar** vuelve a procesar un documento modificado; **Quitar** lo saca del índice y borra el archivo.

Los PDF escaneados (solo imagen) aparecen como «Sin texto»: necesitan OCR antes de subirlos.

También puede copiar los archivos en `data/resoluciones/` y ejecutar `rag.bat indexar`.

## Por qué FAISS y no Chroma
Con unos cientos de fragmentos, FAISS hace una búsqueda exacta en milisegundos, guarda todo en dos
archivos y no necesita servidor ni base de datos. Chroma conviene cuando hay que filtrar por metadatos
o actualizar documentos uno por uno.
