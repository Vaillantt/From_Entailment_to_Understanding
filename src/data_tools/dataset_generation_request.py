import json
import os
from pathlib import Path
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
REPO_ROOT = Path(__file__).resolve().parents[2]
FIELD_OPERATIONS_RAW_DIR = REPO_ROOT / "datasets" / "field_operations" / "raw"
FILE_OUT = FIELD_OPERATIONS_RAW_DIR / "confirm_movement_milestone_300.json"

# ----------------------------
# 1) JSON Schema mais restritivo
# ----------------------------

# Segmentos sem entity (texto normal)
PLAIN_SEGMENT = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "text": {"type": "string", "minLength": 1, "maxLength": 140},
    },
    "required": ["text"],
}

# Segmentos com entity (span rotulado) - forçar span curto pra não rotular a frase toda
ENTITY_SEGMENT = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "text": {"type": "string", "minLength": 1, "maxLength": 30},  # ✅ span curto
        "entity": {"type": "string", "enum": ["milestone", "location_ref"]},
    },
    "required": ["text", "entity"],
}

SEGMENT_SCHEMA = {
    "anyOf": [
        PLAIN_SEGMENT,
        ENTITY_SEGMENT,
    ]
}

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "Confirm Movement Milestone": {
            "type": "array",
            "minItems": 300,
            "maxItems": 300,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "data": {
                        "type": "array",
                        "minItems": 3,   # ✅ impede exemplo com 1 segmento só
                        "maxItems": 5,   # ✅ permite 3 ou 5 (e alguns extras de pontuação/tempo)
                        "items": SEGMENT_SCHEMA
                    }
                },
                "required": ["data"],
            },
        }
    },
    "required": ["Confirm Movement Milestone"],
}

# ----------------------------
# 2) Prompts: instruções + regras mais fortes
# ----------------------------

instructions = (
    "You are generating an English slot-filling dataset.\n"
    "Output MUST be valid JSON that strictly follows the provided JSON Schema.\n"
    "Do not include any extra keys, comments, markdown, or explanations.\n"
    "Within the batch, do not repeat any full utterance (case-insensitive, ignoring extra whitespace).\n"
    "Avoid near-duplicates: do not produce sentences that only change a callsign or a single word.\n"
)

user_prompt = """Generate EXACTLY 300 unique examples for the intent "Confirm Movement Milestone" in English.

Allowed entity labels: ["milestone","location_ref"] only.

Meaning constraints:
- This intent confirms a movement milestone: arrival/on-scene, departure/rolling out, or checkpoint/check-in.
- The milestone entity span must vary a lot (e.g., "on scene", "arrived", "rolling out", "departing now", "checking in", "cleared the checkpoint", etc.).
- location_ref must be a synthetic location reference (no real addresses): "Base Alpha", "Checkpoint 3", "km 12", "Sector Bravo", "Staging Area", "Gate 2", "LZ Charlie", "north perimeter", "Hangar Line", etc.

Required distribution (exact):
- 100 examples: milestone only (tag milestone; do NOT tag location_ref)
- 100 examples: milestone + location_ref (tag both)
- 100 examples: location_ref only (tag only location_ref; milestone may be implied in untagged text but must NOT be tagged)

CRITICAL LENGTH + LABELING RULES (must follow):
- Every example must be a full human utterance (5–25 words).
- NEVER label the whole utterance as an entity.
- Use 3 to 5 segments in "data":
  - If only ONE slot is tagged: use EXACTLY 3 segments:
    [prefix (no entity), entity span, suffix (no entity)]
  - If TWO slots are tagged: use EXACTLY 5 segments:
    [prefix (no entity), entity span #1, middle (no entity), entity span #2, suffix (no entity)]
- At most ONE segment may have entity="milestone".
- At most ONE segment may have entity="location_ref".
- The entity span must be the minimal value phrase only (1–4 words). Examples:
  ✅ "on scene" / ✅ "Checkpoint 3" / ✅ "km 12"
  ❌ "Dispatch, we're on scene" (too long, includes non-value words)
- Make sure there is at least one non-entity prefix and one non-entity suffix in every example.

Style diversity requirements (critical):
- Mix short radio-style utterances, full sentences, confirmations, corrections ("... correction ..."),
  time mentions ("at 14:20"), and optional call signs ("Unit 12", "Eagle-3", "Dispatch", "Control").
- Vary order: milestone first vs location first vs embedded phrasing.
- Ensure strong variation INSIDE the tagged spans and across the untagged parts.
- No repeated full utterances; avoid templated repetition.

GOOD EXAMPLE (structure only):
{
  "data": [
    { "text": "Control, we are now " },
    { "text": "on scene", "entity": "milestone" },
    { "text": " at " },
    { "text": "Base Alpha", "entity": "location_ref" },
    { "text": "." }
  ]
}

Blacklist (optional):
DO_NOT_REPEAT = []

Return ONLY the JSON object with the single top-level key "Confirm Movement Milestone".
"""

# ----------------------------
# 3) Chamada à API
# ----------------------------

resp = client.responses.create(
    model="gpt-4.1",
    instructions=instructions,
    input=user_prompt,
    temperature=0.2,
    max_output_tokens=32768,
    text={
        "format": {
            "type": "json_schema",
            "name": "confirm_movement_milestone_dataset",
            "strict": True,
            "schema": SCHEMA,
        }
    },
)

if getattr(resp, "status", None) == "incomplete":
    raise RuntimeError(f"Incomplete response: {resp.incomplete_details}")

dataset_obj = json.loads(resp.output_text)

# ----------------------------
# 4) Validações mínimas (recomendado)
# ----------------------------

def full_utterance(example: dict) -> str:
    return "".join(seg["text"] for seg in example["data"]).strip()

bad = []
seen = set()

for idx, ex in enumerate(dataset_obj["Confirm Movement Milestone"]):
    utt = " ".join(full_utterance(ex).split())
    norm = utt.lower()

    # (a) tamanho mínimo por palavras
    wc = len(utt.split())
    if wc < 5 or wc > 25:
        bad.append((idx, "length", wc, utt))

    # (b) pelo menos 3 segmentos (schema já força) e checar 3 ou 5 quando tiver entidades
    ents = [seg.get("entity") for seg in ex["data"] if "entity" in seg]
    if ents.count("milestone") > 1 or ents.count("location_ref") > 1:
        bad.append((idx, "too_many_entities", ents, utt))

    # (c) checar se spans rotulados são curtos (defesa extra)
    for seg in ex["data"]:
        if "entity" in seg and len(seg["text"].split()) > 4:
            bad.append((idx, "entity_span_too_long", seg["entity"], seg["text"], utt))

    # (d) dedupe exato
    if norm in seen:
        bad.append((idx, "duplicate", utt))
    seen.add(norm)

if bad:
    print(f"WARNING: found {len(bad)} issues. Showing first 10:")
    for row in bad[:10]:
        print(row)

FIELD_OPERATIONS_RAW_DIR.mkdir(parents=True, exist_ok=True)
with FILE_OUT.open("w", encoding="utf-8") as f:
    json.dump(dataset_obj, f, ensure_ascii=False, indent=2)

print(f"Saved: {FILE_OUT}")
