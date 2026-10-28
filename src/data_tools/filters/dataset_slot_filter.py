#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filtra um dataset de intent/slot removendo sentenças que possuem certos slots.

Entrada suportada:
1) Dict no topo:
{
  "AddToPlaylist": [
    {"data": [{"text": "..."}, {"text": "X", "entity": "slot_a"}]},
    ...
  ],
  "OutroIntent": [ ... ]
}

2) Lista de blocos (cada bloco é {intent: [exemplos]}):
[
  {"AddToPlaylist": [ ... ]},
  {"OutroIntent": [ ... ]}
]

Opções principais:
  --slots SLOTS...        lista de slots a excluir (ex.: music_item playlist_owner)
  --mode {any,all}        'any' (padrão) exclui se tiver QUALQUER um dos slots;
                          'all' exclui apenas se tiver TODOS os slots listados
  --out PATH              caminho do JSON de saída
  --drop-empty-intents    remove intents que ficarem com 0 sentenças

Saída:
- Um novo arquivo JSON sem as sentenças filtradas.
- Estatísticas impressas no terminal (total antes/depois/removidas e por intent).
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

def collect_sentence_slots(example: Dict[str, Any]) -> set:
    """Retorna o conjunto de slots (entities) presentes na sentença."""
    data = example.get("data", [])
    return {span.get("entity") for span in data if isinstance(span, dict) and span.get("entity")}

def should_exclude(example: Dict[str, Any], targets: set, mode: str) -> bool:
    """Decide se exclui a sentença conforme o modo ('any' ou 'all')."""
    present = collect_sentence_slots(example)
    if mode == "all":
        return targets.issubset(present)
    # modo 'any'
    return len(present & targets) > 0

def filter_examples(examples: List[Dict[str, Any]], targets: set, mode: str) -> Tuple[List[Dict[str, Any]], int]:
    """Filtra uma lista de exemplos e retorna (filtrados, removidos)."""
    filtered = []
    removed = 0
    for ex in examples:
        if should_exclude(ex, targets, mode):
            removed += 1
        else:
            filtered.append(ex)
    return filtered, removed

def sanitize_suffix(slots: List[str]) -> str:
    """Gera um sufixo amigável para nome de arquivo a partir da lista de slots."""
    if not slots:
        return "no-slots"
    return "excluding_" + "+".join(s.replace("/", "-") for s in slots)

def process_top_level_keep_shape(
    data: Any,
    targets: set,
    mode: str,
    drop_empty: bool,
) -> Tuple[Any, Dict[str, Dict[str, int]]]:
    """
    Filtra preservando o shape original.
    Retorna (novo_data, stats_por_intent) onde stats_por_intent[intent] = {before, after, removed}.
    """
    stats: Dict[str, Dict[str, int]] = {}

    # Caso 1: dict {intent: [examples]}
    if isinstance(data, dict):
        new_data = {}
        for intent, examples in data.items():
            if not isinstance(examples, list):
                continue
            before = len(examples)
            filtered, removed = filter_examples(examples, targets, mode)
            after = len(filtered)
            if after > 0 or not drop_empty:
                new_data[intent] = filtered
            stats[intent] = {"before": before, "after": after, "removed": removed}
        return new_data, stats

    # Caso 2: lista de blocos [{intent: [examples]}, ...]
    if isinstance(data, list):
        new_list = []
        for block in data:
            if not isinstance(block, dict):
                continue
            new_block = {}
            for intent, examples in block.items():
                if not isinstance(examples, list):
                    continue
                before = len(examples)
                filtered, removed = filter_examples(examples, targets, mode)
                after = len(filtered)
                if after > 0 or not drop_empty:
                    new_block[intent] = filtered
                # agrega stats por intent (somando ao longo de blocos com mesma chave)
                s = stats.setdefault(intent, {"before": 0, "after": 0, "removed": 0})
                s["before"] += before
                s["after"] += after
                s["removed"] += removed
            # Só adiciona bloco se houver pelo menos um intent (ou se drop_empty=False com intents vazios mantidos)
            if new_block or not drop_empty:
                new_list.append(new_block)
        return new_list, stats

    raise ValueError("Formato de arquivo não reconhecido. Esperado dict ou lista de dicts.")

def print_stats(stats: Dict[str, Dict[str, int]]) -> None:
    """Imprime estatísticas por intent e totais."""
    total_before = sum(s["before"] for s in stats.values())
    total_after  = sum(s["after"]  for s in stats.values())
    total_removed = sum(s["removed"] for s in stats.values())
    pct_removed = (total_removed / total_before * 100.0) if total_before else 0.0

    print("=" * 64)
    print("Resumo geral")
    print(f"  Sentenças totais (antes):   {total_before}")
    print(f"  Sentenças restantes (depois): {total_after}")
    print(f"  Sentenças removidas:          {total_removed} ({pct_removed:.1f}%)")
    print("=" * 64)
    print("Por intent:")
    # ordena intents por removidas desc
    for intent, s in sorted(stats.items(), key=lambda kv: (-kv[1]["removed"], kv[0].lower())):
        b, a, r = s["before"], s["after"], s["removed"]
        pct = (r / b * 100.0) if b else 0.0
        print(f"  - {intent}: removidas {r}/{b} ({pct:.1f}%), ficaram {a}")
    print("=" * 64)

def main():
    parser = argparse.ArgumentParser(
        description="Filtra um dataset removendo sentenças que contenham certos slots."
    )
    parser.add_argument("dataset_path", help="Caminho do arquivo JSON do dataset.")
    parser.add_argument(
        "--slots", nargs="+", required=True,
        help="Lista de slots a excluir (ex.: music_item playlist_owner)."
    )
    parser.add_argument(
        "--mode", choices=["any", "all"], default="any",
        help="Regra de exclusão: 'any' (qualquer um dos slots) ou 'all' (todos os slots)."
    )
    parser.add_argument(
        "--out", help="Caminho do arquivo JSON de saída."
    )
    parser.add_argument(
        "--drop-empty-intents", action="store_true",
        help="Remove intents que ficarem com 0 sentenças após o filtro."
    )
    args = parser.parse_args()

    in_path = Path(args.dataset_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {in_path}")

    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    targets = set(args.slots)
    new_data, stats = process_top_level_keep_shape(
        data=data,
        targets=targets,
        mode=args.mode,
        drop_empty=args.drop_empty_intents,
    )

    # Define arquivo de saída
    if args.out:
        out_path = Path(args.out)
    else:
        suffix = sanitize_suffix(list(targets))
        out_path = in_path.with_name(f"{in_path.stem}_{suffix}.json")

    # Escrever JSON
    out_path.write_text(json.dumps(new_data, ensure_ascii=False, indent=2), encoding="utf-8")

    # Estatísticas
    print_stats(stats)
    print(f"\n[✔] Dataset filtrado salvo em: {out_path}")

if __name__ == "__main__":
    main()
