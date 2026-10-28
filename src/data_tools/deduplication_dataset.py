import json
import os
import re
from pathlib import Path
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

REPO_ROOT = Path(__file__).resolve().parents[2]
FIELD_OPERATIONS_RAW_DIR = REPO_ROOT / "datasets" / "field_operations" / "raw"
FIELD_OPERATIONS_ARCHIVE_DIR = REPO_ROOT / "archive" / "datasets" / "field_operations" / "generated_dataset"

FILE_IN = FIELD_OPERATIONS_RAW_DIR / "confirm_movement_milestone_300.json"
FILE_OUT = FIELD_OPERATIONS_ARCHIVE_DIR / "confirm_movement_milestone_300_FINAL.json"
INTENT_KEY = "Confirm Movement Milestone"
TARGET = 300

# --------- Normalização / utilidades ---------

def full_utterance(example: dict) -> str:
    return "".join(seg["text"] for seg in example["data"]).strip()

def normalize(text: str) -> str:
    # normalização "leve" para dedupe exato
    return " ".join(text.split()).strip().lower()

def word_count(text: str) -> int:
    return len(text.split())

def is_valid_example(ex: dict, min_words=8, max_words=25) -> bool:
    utt = full_utterance(ex)
    wc = word_count(utt)
    if wc < min_words or wc > max_words:
        return False

    # no máximo 1 de cada entidade
    ents = [seg.get("entity") for seg in ex["data"] if "entity" in seg]
    if ents.count("milestone") > 1 or ents.count("location_ref") > 1:
        return False

    # spans rotulados curtos (1–4 palavras recomendado)
    for seg in ex["data"]:
        if "entity" in seg and len(seg["text"].split()) > 4:
            return False

    # garantir prefixo e sufixo não rotulados (heurística simples)
    if len(ex["data"]) < 3:
        return False
    if "entity" in ex["data"][0] or "entity" in ex["data"][-1]:
        return False

    return True

def dedupe_keep_first(examples: list) -> tuple[list, dict]:
    """
    Retorna (unique_examples, dup_map) onde dup_map[norm] = [idxs...]
    """
    seen = {}
    dup_map = {}
    unique = []
    for i, ex in enumerate(examples):
        utt = normalize(full_utterance(ex))
        if utt in seen:
            dup_map.setdefault(utt, [seen[utt]]).append(i)
        else:
            seen[utt] = i
            unique.append(ex)
    return unique, dup_map

# --------- Schema dinâmico para gerar exatamente N itens ---------

def build_segment_schema():
    plain = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "text": {"type": "string", "minLength": 1, "maxLength": 140},
        },
        "required": ["text"],
    }
    entity = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "text": {"type": "string", "minLength": 1, "maxLength": 30},
            "entity": {"type": "string", "enum": ["milestone", "location_ref"]},
        },
        "required": ["text", "entity"],
    }
    return {"anyOf": [plain, entity]}

def build_schema_for_n(n: int) -> dict:
    seg = build_segment_schema()

    # Para intent com 2 slots: permitir só 3 ou 5 segmentos (estrutura “limpa”)
    data_schema = {
        "anyOf": [
            {"type": "array", "minItems": 3, "maxItems": 3, "items": seg},
            {"type": "array", "minItems": 5, "maxItems": 5, "items": seg},
        ]
    }

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            INTENT_KEY: {
                "type": "array",
                "minItems": n,
                "maxItems": n,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"data": data_schema},
                    "required": ["data"],
                },
            }
        },
        "required": [INTENT_KEY],
    }

# --------- Prompt para delta ---------

INSTRUCTIONS = (
    "You are generating an English slot-filling dataset.\n"
    "Output MUST be valid JSON that strictly follows the provided JSON Schema.\n"
    "Do not include any extra keys, comments, markdown, or explanations.\n"
)

def build_user_prompt_delta(n: int, blacklist: list[str]) -> str:
    # blacklist como JSON para evitar problemas de aspas
    blacklist_json = json.dumps(blacklist, ensure_ascii=False)

    return f"""Generate EXACTLY {n} NEW unique examples for the intent "{INTENT_KEY}" in English.

Allowed entity labels: ["milestone","location_ref"] only.

Meaning constraints:
- This intent confirms a movement milestone: arrival/on-scene, departure/rolling out, or checkpoint/check-in.
- location_ref is a synthetic location reference (no real addresses).

CRITICAL LENGTH + LABELING RULES:
- Each utterance must be 8–25 words.
- NEVER label the whole utterance as an entity.
- If only ONE slot is tagged: data MUST have exactly 3 segments:
  [plain prefix] + [entity span] + [plain suffix]
- If TWO slots are tagged: data MUST have exactly 5 segments:
  [plain prefix] + [entity span #1] + [plain middle] + [entity span #2] + [plain suffix]
- At most ONE segment may have entity="milestone".
- At most ONE segment may have entity="location_ref".
- The entity span must be minimal (1–4 words).

Uniqueness:
- Do NOT repeat ANY utterance from the blacklist below (case-insensitive, ignoring extra whitespace).
- Also avoid near-duplicates: do not produce sentences that only change a callsign or one word.

DO_NOT_REPEAT = {blacklist_json}

Return ONLY the JSON object with the single top-level key "{INTENT_KEY}".
"""

# --------- Geração delta + loop ---------

def request_delta(n: int, blacklist: list[str], temperature=0.3) -> list[dict]:
    schema = build_schema_for_n(n)
    prompt = build_user_prompt_delta(n, blacklist)

    resp = client.responses.create(
        model="gpt-4.1",
        instructions=INSTRUCTIONS,
        input=prompt,
        temperature=temperature,
        max_output_tokens=32768,
        text={
            "format": {
                "type": "json_schema",
                "name": "delta_dataset",
                "strict": True,
                "schema": schema,
            }
        },
    )

    if getattr(resp, "status", None) == "incomplete":
        raise RuntimeError(f"Incomplete response: {resp.incomplete_details}")

    obj = json.loads(resp.output_text)
    return obj[INTENT_KEY]

def main():
    data = json.loads(FILE_IN.read_text(encoding="utf-8"))
    base = data[INTENT_KEY]

    # 1) filtra inválidos
    base_valid = [ex for ex in base if is_valid_example(ex)]
    print(f"Valid (before dedupe): {len(base_valid)} / {len(base)}")

    # 2) dedupe
    unique, dup_map = dedupe_keep_first(base_valid)
    print(f"Unique (after dedupe): {len(unique)}")
    if dup_map:
        print(f"Duplicate groups found: {len(dup_map)}")

    accepted = unique

    # Loop até completar TARGET
    attempt = 0
    while len(accepted) < TARGET:
        attempt += 1
        missing = TARGET - len(accepted)

        # dica prática: se faltar muito, peça em blocos menores (reduz risco de near-duplicate)
        batch_n = min(missing, 60)

        blacklist = [full_utterance(ex) for ex in accepted]
        print(f"\n[Attempt {attempt}] Missing={missing}, requesting delta={batch_n}...")

        delta = request_delta(batch_n, blacklist, temperature=0.3)

        # valida + dedupe contra accepted
        new_added = 0
        existing_norm = {normalize(full_utterance(ex)) for ex in accepted}

        for ex in delta:
            if not is_valid_example(ex):
                continue
            utt_norm = normalize(full_utterance(ex))
            if utt_norm in existing_norm:
                continue
            accepted.append(ex)
            existing_norm.add(utt_norm)
            new_added += 1

        print(f"Added {new_added} new valid unique examples.")

        # se adicionou 0, aumente temperatura um pouco na próxima tentativa
        if new_added == 0:
            print("No progress. Consider increasing temperature slightly (e.g., 0.4) or relaxing min words.")

    final_obj = {INTENT_KEY: accepted[:TARGET]}
    FIELD_OPERATIONS_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    FILE_OUT.write_text(json.dumps(final_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved final: {FILE_OUT} (count={len(final_obj[INTENT_KEY])})")

if __name__ == "__main__":
    main()
