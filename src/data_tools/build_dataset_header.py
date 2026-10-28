#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gera e SALVA um header com estatísticas de slots por intent.

Entrada esperada (exemplo):
{
  "AddToPlaylist": [
    {"data": [{"text": "add "}, {"text": "album", "entity": "music_item"}, ...]},
    ...
  ],
  "OutroIntent": [ ... ]
}

Saída (arquivo .txt):
================================================================
Intent: AddToPlaylist
Total de sentenças: 123
Slots (presença por sentença):
  - playlist: 78.9% (97/123)
  - artist: 35.0% (43/123)
  ...

Opções:
    --out <ARQUIVO>   caminho do .txt de saída (padrão: <dataset_stem>_header.txt)
    --no-stdout       não imprimir o conteúdo no terminal

Exemplo:

python dataset_header.py train_AddToPlaylist_full.json --out  train_AddToPlaylist_header.txt
python dataset_header.py train_BookRestaurant_full.json --out  train_BookRestaurant__header.txt
python dataset_header.py train_PlayMusic_full.json --out  train_PlayMusic_header.txt


"""

from __future__ import annotations
import argparse
import json
import codecs
from json import JSONDecodeError
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

SEPARATOR = "=" * 64

def load_json_any_encoding(path: Path):
    """
    Lê bytes e tenta decodificar em ordem: utf-8, utf-8-sig, cp1252, latin-1.
    Retorna (data, encoding_usado).
    """
    raw = path.read_bytes()
    tried = []
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            tried.append(enc)
            text = raw.decode(enc)
            return json.loads(text), enc
        except (UnicodeDecodeError, JSONDecodeError):
            continue
    # último recurso: latin-1 nunca quebra; se quebrar no JSON, re-raise com contexto
    try:
        text = raw.decode("latin-1")
        return json.loads(text), "latin-1"
    except JSONDecodeError as e:
        raise JSONDecodeError(
            f"Falha ao decodificar/parsing JSON. Tentativas: {tried}. "
            f"Última mensagem: {e.msg}", e.doc, e.pos
        )
    

def compute_presence_counts(examples: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Retorna dict {slot: num_sentencas_com_ao_menos_uma_ocorrencia}.
    """
    counts = defaultdict(int)
    for ex in examples:
        data = ex.get("data", [])
        slots_in_sentence = {span.get("entity") for span in data if span.get("entity")}
        for ent in slots_in_sentence:
            counts[ent] += 1
    return dict(counts)

def format_header(intent: str, examples: List[Dict[str, Any]]) -> str:
    total = len(examples)
    counts = compute_presence_counts(examples)
    lines = [SEPARATOR, f"Intent: {intent}", f"Total de sentenças: {total}", "Slots (presença por sentença):"]
    if not counts:
        lines.append("  — (nenhum slot encontrado)")
    else:
        # ordenar por % desc e depois por nome
        ordered = sorted(counts.items(),
                         key=lambda kv: (-(kv[1] / total if total else 0.0), kv[0].lower()))
        for slot, c in ordered:
            pct = (c / total) * 100.0 if total else 0.0
            lines.append(f"  - {slot}: {pct:.1f}% ({c}/{total})")
    lines.append(SEPARATOR)
    return "\n".join(lines)

def parse_top(data: Any):
    """
    Aceita:
      - dict {intent: [examples]}
      - list de dicts [{intent: [examples]}, ...] (agrega)
    """
    if isinstance(data, dict):
        return list(data.items())
    if isinstance(data, list):
        merged = {}
        for block in data:
            if isinstance(block, dict):
                for k, v in block.items():
                    merged.setdefault(k, []).extend(v if isinstance(v, list) else [])
        return list(merged.items())
    raise ValueError("Formato de arquivo não reconhecido. Esperado dict ou lista de dicts.")

def main():
    ap = argparse.ArgumentParser(description="Gera e salva header de intent/slots com frequência por sentença.")
    ap.add_argument("dataset_path", help="Caminho para o arquivo JSON do dataset.")
    ap.add_argument("--out", help="Caminho do .txt de saída (default: <dataset_stem>_header.txt).")
    ap.add_argument("--no-stdout", action="store_true", help="Não imprimir o header no terminal.")
    args = ap.parse_args()

    in_path = Path(args.dataset_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {in_path}")

    data, used_enc = load_json_any_encoding(in_path)

    intents = parse_top(data)
    if not intents:
        raise ValueError("Nenhum intent/sentença válido encontrado no arquivo.")

    # Monta todo o conteúdo
    headers = []
    for intent, examples in intents:
        if isinstance(examples, list):
            headers.append(format_header(intent, examples))

    content = "\n\n".join(headers)

    # Define saída
    out_path = Path(args.out) if args.out else in_path.with_name(in_path.stem + "_header.txt")
    out_path.write_text(content + "\n", encoding="utf-8")

    
    print(content)
    print(f"\n[✔] Header salvo em: {out_path}")

if __name__ == "__main__":
    main()
