#!/usr/bin/env python3
# filter_with_slots.py

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def example_has_slot(example: Dict[str, Any]) -> bool:
    """
    Retorna True se o exemplo tiver ao menos um chunk em 'data' com a chave 'entity'
    (e valor não vazio).
    """
    data = example.get("data", [])
    if not isinstance(data, list):
        return False

    for chunk in data:
        if isinstance(chunk, dict) and "entity" in chunk:
            ent = chunk.get("entity")
            if ent is not None and str(ent).strip() != "":
                return True
    return False


def filter_dataset(obj: Dict[str, Any], drop_empty_intents: bool = True) -> Dict[str, Any]:
    """
    Mantém apenas exemplos com slots em cada intent.
    Se drop_empty_intents=True, remove intents sem nenhum exemplo restante.
    """
    if not isinstance(obj, dict):
        raise ValueError("JSON raiz precisa ser um objeto/dict (intent -> lista de exemplos).")

    out: Dict[str, Any] = {}
    for intent, examples in obj.items():
        if not isinstance(examples, list):
            # Se por acaso o formato vier diferente, preserva como está (ou ignore)
            continue

        kept = [ex for ex in examples if isinstance(ex, dict) and example_has_slot(ex)]

        if kept or not drop_empty_intents:
            out[intent] = kept

    return out


def process_file(in_path: Path, out_path: Path, drop_empty_intents: bool) -> None:
    with in_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    filtered = filter_dataset(obj, drop_empty_intents=drop_empty_intents)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Filtra JSONs mantendo apenas exemplos que possuem ao menos um slot (chunk com 'entity')."
    )
    parser.add_argument("input", help="Arquivo .json ou pasta contendo .json")
    parser.add_argument(
        "-o", "--output",
        help="Arquivo de saída (se input for arquivo) OU pasta de saída (se input for pasta). "
             "Se omitido, cria arquivos com sufixo .with_slots.json ao lado dos originais."
    )
    parser.add_argument(
        "--keep-empty-intents",
        action="store_true",
        help="Não remove intents que ficarem vazias após o filtro."
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    drop_empty_intents = not args.keep_empty_intents

    if in_path.is_file():
        if args.output:
            out_path = Path(args.output)
        else:
            out_path = in_path.with_suffix("")  # remove .json
            out_path = out_path.with_name(out_path.name + ".with_slots").with_suffix(".json")

        process_file(in_path, out_path, drop_empty_intents)
        print(f"OK: {in_path} -> {out_path}")

    elif in_path.is_dir():
        out_dir = Path(args.output) if args.output else in_path

        json_files = sorted(in_path.glob("*.json"))
        if not json_files:
            raise SystemExit(f"Nenhum .json encontrado em: {in_path}")

        for fp in json_files:
            if args.output:
                out_path = out_dir / fp.name
            else:
                stem = fp.stem  # sem .json
                out_path = fp.with_name(f"{stem}.with_slots.json")

            process_file(fp, out_path, drop_empty_intents)
            print(f"OK: {fp.name} -> {out_path.name}")

        print("Concluído.")
    else:
        raise SystemExit(f"Caminho inválido: {in_path}")


if __name__ == "__main__":
    main()
