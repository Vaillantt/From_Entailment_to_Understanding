#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set


def iter_examples(dataset_obj: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    out = []
    for intent, examples in dataset_obj.items():
        if isinstance(examples, list):
            for ex in examples:
                if isinstance(ex, dict):
                    out.append((intent, ex))
    return out


def example_entities(example: Dict[str, Any]) -> Set[str]:
    ents = set()
    data = example.get("data", [])
    if not isinstance(data, list):
        return ents
    for chunk in data:
        if isinstance(chunk, dict) and "entity" in chunk:
            ent = chunk.get("entity")
            if ent is not None and str(ent).strip() != "":
                ents.add(str(ent).strip())
    return ents


def count_slot_presence(dataset_obj: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {}
    for intent, ex in iter_examples(dataset_obj):
        counts.setdefault(intent, {})
        for ent in example_entities(ex):
            counts[intent][ent] = counts[intent].get(ent, 0) + 1
    return counts


def filter_dataset_obj(
    dataset_obj: Dict[str, Any],
    min_occ: int,
    drop_empty_examples: bool = True,     # <- DEFAULT TRUE
    drop_empty_intents: bool = True
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    counts = count_slot_presence(dataset_obj)

    keep_by_intent: Dict[str, Set[str]] = {}
    removed_by_intent: Dict[str, Set[str]] = {}

    for intent, slot_counts in counts.items():
        keep = {s for s, c in slot_counts.items() if c >= min_occ}
        removed = {s for s, c in slot_counts.items() if c < min_occ}
        keep_by_intent[intent] = keep
        removed_by_intent[intent] = removed

    out: Dict[str, Any] = {}
    stats: Dict[str, Any] = {"min_occ": min_occ, "intents": {}}

    for intent, examples in dataset_obj.items():
        if not isinstance(examples, list):
            continue

        keep_slots = keep_by_intent.get(intent, set())
        before_n = len(examples)
        kept_examples = []

        for ex in examples:
            if not isinstance(ex, dict):
                continue
            data = ex.get("data", [])
            if not isinstance(data, list):
                continue

            new_data = []
            has_any_slot = False

            for chunk in data:
                if not isinstance(chunk, dict):
                    new_data.append(chunk)
                    continue

                if "entity" in chunk:
                    ent = chunk.get("entity")
                    ent_str = str(ent).strip() if ent is not None else ""
                    if ent_str in keep_slots and ent_str != "":
                        has_any_slot = True
                        new_data.append(chunk)
                    else:
                        # remove the entity label but keep the text
                        new_chunk = dict(chunk)
                        new_chunk.pop("entity", None)
                        new_data.append(new_chunk)
                else:
                    new_data.append(chunk)

            if drop_empty_examples and not has_any_slot:
                continue  # <- remove sentence/example entirely

            new_ex = dict(ex)
            new_ex["data"] = new_data
            kept_examples.append(new_ex)

        after_n = len(kept_examples)

        if kept_examples or not drop_empty_intents:
            out[intent] = kept_examples

        stats["intents"][intent] = {
            "examples_before": before_n,
            "examples_after": after_n,
            "kept_slots": sorted(list(keep_by_intent.get(intent, set()))),
            "removed_slots": sorted(list(removed_by_intent.get(intent, set())))
        }

    return out, stats


def process_file(in_path: Path, out_path: Path, min_occ: int, drop_empty_intents: bool, write_stats: bool):
    with in_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise ValueError(f"JSON raiz precisa ser dict: {in_path}")

    filtered, stats = filter_dataset_obj(
        obj,
        min_occ=min_occ,
        drop_empty_examples=True,          # <- ALWAYS TRUE now
        drop_empty_intents=drop_empty_intents
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    if write_stats:
        stats_path = out_path.with_suffix("").with_name(out_path.stem + "_stats").with_suffix(".json")
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser(
        description="Filtra slots raros (<N ocorrências por sentença) e remove exemplos que ficarem sem slots."
    )
    ap.add_argument("input", help="Arquivo .json ou diretório contendo .json")
    ap.add_argument("--min-occ", type=int, default=10, help="Mínimo de ocorrências para manter o slot. Default: 10")
    ap.add_argument("--out-dir", default=None, help="Diretório de saída. Default: filtered_min<occ> ao lado do input.")
    ap.add_argument("--suffix", default=None, help="Sufixo do arquivo de saída. Default: _min<occ>.json")
    ap.add_argument("--keep-empty-intents", action="store_true", help="Mantém intents vazias após a filtragem.")
    ap.add_argument("--no-stats", action="store_true", help="Não gerar arquivo *_stats.json por dataset.")
    args = ap.parse_args()

    in_path = Path(args.input)
    min_occ = args.min_occ
    drop_empty_intents = not args.keep_empty_intents
    write_stats = not args.no_stats
    suffix = args.suffix or f"_min{min_occ}.json"

    if in_path.is_file():
        out_dir = Path(args.out_dir) if args.out_dir else in_path.parent / f"filtered_min{min_occ}"
        out_path = out_dir / (in_path.stem + suffix)
        process_file(in_path, out_path, min_occ, drop_empty_intents, write_stats)
        print(f"OK: {in_path} -> {out_path}")

    elif in_path.is_dir():
        out_dir = Path(args.out_dir) if args.out_dir else in_path / f"filtered_min{min_occ}"
        out_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(in_path.glob("*.json"))
        if not files:
            raise SystemExit(f"Nenhum .json em {in_path}")

        for fp in files:
            if fp.parent == out_dir:
                continue
            out_path = out_dir / (fp.stem + suffix)
            process_file(fp, out_path, min_occ, drop_empty_intents, write_stats)
            print(f"OK: {fp.name} -> {out_path.name}")

        print(f"Concluído. Saída em: {out_dir}")
    else:
        raise SystemExit(f"Caminho inválido: {in_path}")


if __name__ == "__main__":
    main()
