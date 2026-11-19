"""
RAG de normas de tránsito y seguridad vial de Bogotá (VíaSegura IA).

    documentos.py        1. CARGA      lee .txt / .pdf / .docx de data/resoluciones y limpia el texto
    fragmentos.py        2. FRAGMENTOS parte cada norma por artículos y en ventanas con solape
    embeddings.py        3. EMBEDDINGS sentence-transformers (multilingual-e5) → vectores normalizados
    almacen_vectorial.py 4. ÍNDICE     FAISS (similitud coseno) + metadatos en disco (out/rag/)
    llm.py               5. LLM        Claude (API de Anthropic) redacta la respuesta citando fuentes
    pipeline.py          orquesta todo: indexar() y preguntar()
    cli.py               línea de comandos: indexar · buscar · preguntar · chat
"""
