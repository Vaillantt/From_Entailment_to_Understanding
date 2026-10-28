from __future__ import annotations

import json
import re
import string
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - depends on active environment
    pd = None

from src.common.text_utils import (
    add_row_scores_masked,
    build_alias_collapsed_grouping_view,
    find_all_matching_substrings,
    fuse_special_tokens,
    is_string_not_present,
    process_until_no_more_groups,
    remove_duplicates,
    remove_duplicates_and_substrings,
    sanitize_edges,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ROOT = REPO_ROOT / "evaluation"
DEFAULT_ALIAS_REGISTRY_PATH = REPO_ROOT / "datasets" / "metadata" / "semantic_alias_registry.json"
DEFAULT_SLOT_DELIMITER = " || "
EVALUATION_SUBDIRS = ("dataset_csv", "results_csv", "metrics")
_EXPORT_STEM_PREFIXES = ("train_", "test_", "dev_")

_METADATA_SLOT_DICT_PATHS = {
    "snips": REPO_ROOT / "datasets" / "snips" / "metadata" / "snips_intents_slot_dict.json",
    "multiwoz": REPO_ROOT / "datasets" / "multiwoz" / "metadata" / "multiwoz_intents_slot_dict.json",
    "atis": REPO_ROOT / "datasets" / "atis" / "metadata" / "atis_intents_slot_dict.json",
    "field_operations": REPO_ROOT / "datasets" / "field_operations" / "metadata" / "field_operations_intents_slot_dict.json",
}


def _require_pandas() -> None:
    if pd is None:
        raise ModuleNotFoundError(
            "pandas is required for batch_eval_utils. Install it in the active notebook/kernel environment."
        )


def _load_json_file(path: Path) -> Dict[str, Any]:
    last_error: Exception | None = None
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return json.loads(path.read_text(encoding=encoding))
        except Exception as exc:  # pragma: no cover - best effort fallback
            last_error = exc
    raise RuntimeError(f"Could not read JSON file: {path}") from last_error


def strip_accents(text: str) -> str:
    if not isinstance(text, str):
        return text
    return "".join(
        ch for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )


def norm_text(text: str) -> str:
    if text is None:
        return ""
    return " ".join(strip_accents(str(text)).lower().strip().split())


def reconstruct_sentence(data_list: Sequence[Mapping[str, Any]]) -> str:
    return "".join(span.get("text", "") for span in data_list)


def detect_dataset_name(dataset_path: str | Path) -> str:
    path_str = str(dataset_path).replace("\\", "/").lower()
    if "/snips/" in path_str:
        return "snips"
    if "/multiwoz/" in path_str:
        return "multiwoz"
    if "/atis/" in path_str:
        return "atis"
    if "/field_operations/" in path_str:
        return "field_operations"
    raise ValueError(f"Could not detect dataset name from path: {dataset_path}")


def load_semantic_alias_registry(
    registry_path: str | Path = DEFAULT_ALIAS_REGISTRY_PATH,
) -> Dict[str, Any]:
    return _load_json_file(Path(registry_path))


def load_metadata_intent_slot_dict(dataset_name: str) -> Dict[str, Any]:
    if dataset_name not in _METADATA_SLOT_DICT_PATHS:
        raise KeyError(f"No metadata prompt file configured for dataset '{dataset_name}'.")
    return _load_json_file(_METADATA_SLOT_DICT_PATHS[dataset_name])


def validate_registry_against_metadata(
    registry: Mapping[str, Any] | None = None,
) -> bool:
    registry = registry or load_semantic_alias_registry()
    errors: List[str] = []

    for dataset_name, dataset_cfg in registry.items():
        metadata = load_metadata_intent_slot_dict(dataset_name)
        intents = dataset_cfg.get("intents", {})
        for raw_intent, intent_cfg in intents.items():
            semantic_intent = intent_cfg.get("semantic_intent")
            if semantic_intent not in metadata:
                errors.append(
                    f"[{dataset_name}] semantic intent '{semantic_intent}' from raw intent "
                    f"'{raw_intent}' not found in metadata."
                )
                continue

            metadata_slots = set(metadata[semantic_intent].get("slots", {}).keys())
            mapped_slots = set(intent_cfg.get("slot_map", {}).values())
            missing_slots = sorted(mapped_slots - metadata_slots)
            if missing_slots:
                errors.append(
                    f"[{dataset_name}] semantic slots {missing_slots} from raw intent "
                    f"'{raw_intent}' are missing in metadata intent '{semantic_intent}'."
                )

    if errors:
        raise ValueError("Registry validation failed:\n- " + "\n- ".join(errors))
    return True


def build_semantic_intent_slot_dict(
    dataset_names: Sequence[str] | None = None,
    registry: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    registry = registry or load_semantic_alias_registry()
    dataset_names = list(dataset_names or registry.keys())
    semantic_dict: Dict[str, Any] = {}

    for dataset_name in dataset_names:
        metadata = load_metadata_intent_slot_dict(dataset_name)
        for _, intent_cfg in registry.get(dataset_name, {}).get("intents", {}).items():
            semantic_intent = intent_cfg["semantic_intent"]
            if semantic_intent not in metadata:
                raise KeyError(
                    f"Semantic intent '{semantic_intent}' missing from metadata for dataset '{dataset_name}'."
                )
            semantic_dict[semantic_intent] = metadata[semantic_intent]

    return semantic_dict


def build_semantic_intent_candidates_by_dataset(
    dataset_names: Sequence[str] | None = None,
    registry: Mapping[str, Any] | None = None,
) -> Dict[str, List[str]]:
    registry = registry or load_semantic_alias_registry()
    dataset_names = list(dataset_names or registry.keys())
    candidates_by_dataset: Dict[str, List[str]] = {}

    for dataset_name in dataset_names:
        semantic_dict = build_semantic_intent_slot_dict(
            [dataset_name],
            registry=registry,
        )
        candidates_by_dataset[str(dataset_name)] = list(semantic_dict.keys())

    return candidates_by_dataset


def _relative_source_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def normalize_export_stem(source_file: str | Path) -> str:
    stem = Path(source_file).stem
    stem_lower = stem.lower()
    for prefix in _EXPORT_STEM_PREFIXES:
        if stem_lower.startswith(prefix):
            return stem[len(prefix):]
    return stem


def ensure_evaluation_dataset_dirs(
    dataset_name: str,
    evaluation_root: str | Path = EVALUATION_ROOT,
) -> Dict[str, Path]:
    dataset_root = Path(evaluation_root) / str(dataset_name)
    directories = {
        subdir: dataset_root / subdir
        for subdir in EVALUATION_SUBDIRS
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return directories


def get_evaluation_output_dir(
    dataset_name: str,
    subdir: str,
    evaluation_root: str | Path = EVALUATION_ROOT,
) -> Path:
    if subdir not in EVALUATION_SUBDIRS:
        raise ValueError(
            f"Unsupported evaluation subdir '{subdir}'. Expected one of {EVALUATION_SUBDIRS}."
        )
    return ensure_evaluation_dataset_dirs(dataset_name, evaluation_root=evaluation_root)[subdir]


def _normalize_slot_value_dict(
    slot_values: Mapping[str, Sequence[str]] | None,
) -> Dict[str, List[str]]:
    normalized: Dict[str, List[str]] = {}
    if not slot_values:
        return normalized

    for slot_name, values in slot_values.items():
        seen = set()
        ordered_values = []
        for value in values:
            text = "" if value is None else str(value)
            if text not in seen:
                seen.add(text)
                ordered_values.append(text)
        if ordered_values:
            normalized[slot_name] = ordered_values
    return normalized


def _join_slot_values(values: Sequence[str], delimiter: str = DEFAULT_SLOT_DELIMITER) -> str:
    cleaned = ["" if value is None else str(value) for value in values if str(value or "").strip()]
    return delimiter.join(cleaned)


def _load_dataset_payload(dataset_path: str | Path) -> Dict[str, Any]:
    return _load_json_file(Path(dataset_path))


def build_dataset_view_for_file(
    dataset_path: str | Path,
    registry: Mapping[str, Any] | None = None,
    slot_delimiter: str = DEFAULT_SLOT_DELIMITER,
) -> pd.DataFrame:
    _require_pandas()
    registry = registry or load_semantic_alias_registry()
    path = Path(dataset_path)
    dataset_name = detect_dataset_name(path)
    payload = _load_dataset_payload(path)

    if not isinstance(payload, dict):
        raise ValueError(f"Dataset file must be a dict keyed by intent: {path}")

    base_columns = [
        "example_id",
        "dataset_name",
        "source_file",
        "dataset_intent",
        "semantic_intent",
        "sentence",
        "has_repeated_slot_value",
    ]
    rows: List[Dict[str, Any]] = []
    raw_slot_columns: List[str] = []

    for dataset_intent, examples in payload.items():
        if dataset_intent not in registry.get(dataset_name, {}).get("intents", {}):
            raise KeyError(
                f"Raw intent '{dataset_intent}' from '{path}' is missing in the alias registry for dataset '{dataset_name}'."
            )

        intent_cfg = registry[dataset_name]["intents"][dataset_intent]
        semantic_intent = intent_cfg["semantic_intent"]
        raw_slot_order = list(intent_cfg.get("slot_map", {}).keys())
        raw_slot_columns.extend(raw_slot_order)

        for example_id, example in enumerate(examples):
            data_list = example.get("data", [])
            sentence = reconstruct_sentence(data_list)
            raw_slot_values: Dict[str, List[str]] = defaultdict(list)
            semantic_slot_values: Dict[str, List[str]] = defaultdict(list)

            for span in data_list:
                raw_slot = span.get("entity")
                text = span.get("text", "")
                if raw_slot:
                    raw_slot_values[raw_slot].append(text)
                    mapped_slot = intent_cfg.get("slot_map", {}).get(raw_slot)
                    if mapped_slot:
                        semantic_slot_values[mapped_slot].append(text)

            raw_slot_values = _normalize_slot_value_dict(raw_slot_values)
            semantic_slot_values = _normalize_slot_value_dict(semantic_slot_values)

            row: Dict[str, Any] = {
                "example_id": example_id,
                "dataset_name": dataset_name,
                "source_file": _relative_source_path(path),
                "dataset_intent": dataset_intent,
                "semantic_intent": semantic_intent,
                "sentence": sentence,
                "has_repeated_slot_value": any(len(values) > 1 for values in raw_slot_values.values()),
                "gold_slots_dataset_json": json.dumps(raw_slot_values, ensure_ascii=False),
                "gold_slots_semantic_json": json.dumps(semantic_slot_values, ensure_ascii=False),
            }

            for raw_slot in raw_slot_order:
                row[raw_slot] = _join_slot_values(raw_slot_values.get(raw_slot, []), slot_delimiter)

            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=base_columns + ["gold_slots_dataset_json", "gold_slots_semantic_json"])

    dataset_view = pd.DataFrame(rows)
    ordered_raw_slots = []
    seen_slots = set()
    for slot_name in raw_slot_columns:
        if slot_name not in seen_slots:
            ordered_raw_slots.append(slot_name)
            seen_slots.add(slot_name)

    final_columns = (
        base_columns
        + ordered_raw_slots
        + ["gold_slots_dataset_json", "gold_slots_semantic_json"]
    )
    return dataset_view.reindex(columns=final_columns)


def build_dataset_view(
    dataset_paths: Sequence[str | Path],
    registry: Mapping[str, Any] | None = None,
    slot_delimiter: str = DEFAULT_SLOT_DELIMITER,
) -> pd.DataFrame:
    _require_pandas()
    registry = registry or load_semantic_alias_registry()
    frames = [
        build_dataset_view_for_file(path, registry=registry, slot_delimiter=slot_delimiter)
        for path in dataset_paths
    ]
    if not frames:
        return pd.DataFrame()

    dataset_view = pd.concat(frames, ignore_index=True, sort=False)
    base_columns = [
        "example_id",
        "dataset_name",
        "source_file",
        "dataset_intent",
        "semantic_intent",
        "sentence",
        "has_repeated_slot_value",
    ]
    tail_columns = ["gold_slots_dataset_json", "gold_slots_semantic_json"]
    slot_columns = [
        column for column in dataset_view.columns
        if column not in base_columns + tail_columns
    ]
    final_columns = base_columns + slot_columns + tail_columns
    return dataset_view.reindex(columns=final_columns)


def _semantic_slot_inventory_for_view(
    dataset_view: pd.DataFrame,
    registry: Mapping[str, Any],
) -> List[str]:
    slot_names: List[str] = []
    seen = set()

    for _, row in dataset_view[["dataset_name", "dataset_intent"]].drop_duplicates().iterrows():
        intent_cfg = registry[row["dataset_name"]]["intents"][row["dataset_intent"]]
        for semantic_slot in intent_cfg.get("slot_map", {}).values():
            if semantic_slot not in seen:
                seen.add(semantic_slot)
                slot_names.append(semantic_slot)

    return slot_names


def build_semantic_view(
    dataset_view: pd.DataFrame,
    registry: Mapping[str, Any] | None = None,
    slot_delimiter: str = DEFAULT_SLOT_DELIMITER,
) -> pd.DataFrame:
    _require_pandas()
    registry = registry or load_semantic_alias_registry()
    if dataset_view.empty:
        return pd.DataFrame()

    semantic_slot_columns = _semantic_slot_inventory_for_view(dataset_view, registry)
    rows: List[Dict[str, Any]] = []

    for _, row in dataset_view.iterrows():
        semantic_slot_values = json.loads(row["gold_slots_semantic_json"] or "{}")
        semantic_slot_values = _normalize_slot_value_dict(semantic_slot_values)
        semantic_row = {
            "example_id": row["example_id"],
            "dataset_name": row["dataset_name"],
            "source_file": row["source_file"],
            "dataset_intent": row["dataset_intent"],
            "semantic_intent": row["semantic_intent"],
            "sentence": row["sentence"],
            "has_repeated_slot_value": row["has_repeated_slot_value"],
            "gold_slots_dataset_json": row["gold_slots_dataset_json"],
            "gold_slots_semantic_json": row["gold_slots_semantic_json"],
            "pred_intent": "",
            "pred_intent_score": 0.0,
            "slot_eval_intent": row["semantic_intent"],
            "pred_slots_semantic_json": "{}",
            "pred_slots_score_json": "{}",
        }

        for semantic_slot in semantic_slot_columns:
            semantic_row[semantic_slot] = _join_slot_values(
                semantic_slot_values.get(semantic_slot, []),
                slot_delimiter,
            )
            semantic_row[f"pred__{semantic_slot}"] = ""

        rows.append(semantic_row)

    semantic_view = pd.DataFrame(rows)
    base_columns = [
        "example_id",
        "dataset_name",
        "source_file",
        "dataset_intent",
        "semantic_intent",
        "sentence",
        "has_repeated_slot_value",
        "gold_slots_dataset_json",
        "gold_slots_semantic_json",
        "pred_intent",
        "pred_intent_score",
        "slot_eval_intent",
        "pred_slots_semantic_json",
        "pred_slots_score_json",
    ]
    prediction_columns = [f"pred__{slot_name}" for slot_name in semantic_slot_columns]
    final_columns = base_columns + semantic_slot_columns + prediction_columns
    return semantic_view.reindex(columns=final_columns)


def build_dataset_and_semantic_views(
    dataset_paths: Sequence[str | Path],
    registry: Mapping[str, Any] | None = None,
    slot_delimiter: str = DEFAULT_SLOT_DELIMITER,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    registry = registry or load_semantic_alias_registry()
    dataset_view = build_dataset_view(dataset_paths, registry=registry, slot_delimiter=slot_delimiter)
    semantic_view = build_semantic_view(dataset_view, registry=registry, slot_delimiter=slot_delimiter)
    return dataset_view, semantic_view


def _export_dataset_partition_csvs(
    dataset_view: pd.DataFrame,
    partition_name: str,
    file_suffix: str,
    evaluation_root: str | Path = EVALUATION_ROOT,
) -> List[Path]:
    _require_pandas()
    if partition_name not in {"dataset_csv", "results_csv"}:
        raise ValueError(
            "partition_name must be 'dataset_csv' or 'results_csv'."
        )
    if dataset_view.empty:
        return []

    written: List[Path] = []
    for dataset_name, dataset_frame in dataset_view.groupby("dataset_name", sort=True):
        output_dir = get_evaluation_output_dir(
            str(dataset_name),
            partition_name,
            evaluation_root=evaluation_root,
        )
        dataset_frame = dataset_frame.reset_index(drop=True)
        aggregate_path = output_dir / f"{dataset_name}_{file_suffix}.csv"
        dataset_frame.to_csv(aggregate_path, index=False, encoding="utf-8-sig")
        written.append(aggregate_path)

        for source_file, frame in dataset_frame.groupby("source_file", sort=False):
            normalized_stem = normalize_export_stem(source_file)
            path = output_dir / f"{normalized_stem}_{file_suffix}.csv"
            frame.to_csv(path, index=False, encoding="utf-8-sig")
            written.append(path)

    return written


def export_evaluation_csvs(
    dataset_view: pd.DataFrame | None = None,
    results_view: pd.DataFrame | None = None,
    evaluation_root: str | Path = EVALUATION_ROOT,
) -> Dict[str, List[Path]]:
    _require_pandas()
    written: Dict[str, List[Path]] = {"dataset_csv": [], "results_csv": []}

    if dataset_view is not None:
        written["dataset_csv"] = _export_dataset_partition_csvs(
            dataset_view,
            partition_name="dataset_csv",
            file_suffix="dataset",
            evaluation_root=evaluation_root,
        )

    if results_view is not None:
        written["results_csv"] = _export_dataset_partition_csvs(
            results_view,
            partition_name="results_csv",
            file_suffix="predictions",
            evaluation_root=evaluation_root,
        )

    return written


def _normalize_classifier_outputs(outputs: Any, expected_length: int) -> List[Dict[str, Any]]:
    if isinstance(outputs, dict):
        outputs = [outputs]
    outputs = list(outputs or [])
    if len(outputs) != expected_length:
        raise ValueError(
            f"Expected {expected_length} classifier outputs, got {len(outputs)}."
        )
    return outputs


def batch_intent_recognition(
    nlu_processor,
    sentences: Sequence[str],
    semantic_intents: Sequence[str],
    batch_size: int = 16,
    unknown_label: str = "UNK_INTENT",
) -> List[Tuple[str, float]]:
    if not sentences:
        return []

    outputs = nlu_processor.models.classifier(
        list(sentences),
        list(semantic_intents),
        hypothesis_template="This command is intended to {}",
        multi_label=False,
        batch_size=batch_size,
    )
    outputs = _normalize_classifier_outputs(outputs, len(sentences))

    predictions: List[Tuple[str, float]] = []
    threshold = getattr(
        nlu_processor,
        "tau_intent",
        getattr(nlu_processor, "INTENT_THRESHOLD", 0.0),
    )
    for output in outputs:
        labels = output.get("labels", [])
        scores = output.get("scores", [])
        if not labels or not scores:
            predictions.append((unknown_label, 0.0))
            continue

        top_label = str(labels[0])
        top_score = float(scores[0])
        if top_score < threshold:
            predictions.append((unknown_label, round(top_score, 3)))
        else:
            predictions.append((top_label, round(top_score, 3)))

    return predictions


def _extract_entities_from_doc(doc) -> List[str]:
    return [ent.text for ent in doc.ents]


def _remove_slot_words(text: str, slot_labels: Sequence[str]) -> str:
    current = text
    for slot_label in slot_labels:
        for word in slot_label.split():
            current = re.sub(rf"\b{re.escape(word)}\b", "", current, flags=re.IGNORECASE)
    return current


def _extract_slot_tokens_from_doc(doc, intent: str, intent_slot_dict: Mapping[str, Any]) -> List[str]:
    entities = [ent.text for ent in doc.ents]
    slot_labels = list(intent_slot_dict.get(intent, {}).get("slots", {}).keys())
    pos_desired = ("NOUN", "PROPN", "ADV", "PRON", "VERB")

    nouns_and_tokens: List[str] = []
    i = 0
    n = len(doc)

    while i < n:
        token = doc[i]

        if token.text in entities:
            i += 1
            continue

        if token.pos_ not in pos_desired:
            i += 1
            continue

        if not is_string_not_present(token.text, intent_slot_dict):
            i += 1
            continue

        if token.pos_ == "VERB" and token.dep_ == "ROOT":
            i += 1
            continue

        if token.pos_ == "VERB":
            subtree_tokens = list(token.subtree)
            if len(subtree_tokens) > 1:
                phrase = doc[subtree_tokens[0].i:subtree_tokens[-1].i + 1].text
                phrase = _remove_slot_words(phrase, slot_labels).strip()
                if phrase:
                    nouns_and_tokens.append(phrase)
            i += 1
            continue

        if token.pos_ in ("ADV", "PRON"):
            nouns_and_tokens.append(token.text)
            i += 1
            continue

        if token.dep_ != "compound":
            nouns_and_tokens.append(token.text)
            i += 1
            continue

        start = i
        while i < n and doc[i].pos_ in ("NOUN", "PROPN"):
            if doc[i].dep_ != "compound":
                i += 1
                break
            i += 1

        compound_phrase = doc[start:i].text
        compound_phrase = _remove_slot_words(compound_phrase, slot_labels).strip()
        if compound_phrase:
            nouns_and_tokens.append(compound_phrase)

    tokens_list = []
    for token_str in nouns_and_tokens:
        if not any(token_str.lower() in ent.lower() for ent in entities):
            tokens_list.append(token_str)

    fused_candidates = []
    for fused_token in fuse_special_tokens(doc):
        if "-" not in fused_token and "'" not in fused_token and "’" not in fused_token:
            continue
        if any(fused_token.lower() in ent.lower() for ent in entities):
            continue

        cleaned = _remove_slot_words(fused_token, slot_labels).strip()
        if cleaned and is_string_not_present(cleaned, intent_slot_dict):
            fused_candidates.append(cleaned)

    tokens_list.extend(fused_candidates)
    return remove_duplicates_and_substrings(tokens_list)


def _filter_doctitles_for_intent(
    nlp,
    doc_titles: Sequence[str],
    intent: str,
    intent_slot_dict: Mapping[str, Any],
) -> List[str]:
    slot_labels = list(intent_slot_dict.get(intent, {}).get("slots", {}).keys())
    pos_discard = {"ADP", "ADV", "AUX", "INTJ", "CONJ", "CCONJ", "SCONJ", "VERB"}

    filtered = list(doc_titles)
    for idx in reversed(range(len(filtered))):
        term = filtered[idx]
        doc = nlp(term)
        if len(doc) == 1 and doc[0].pos_ in pos_discard:
            filtered.pop(idx)

    cleaned_titles: List[str] = []
    for title in filtered:
        current = _remove_slot_words(title, slot_labels).strip()
        if current:
            cleaned_titles.append(current)

    return cleaned_titles


def _batch_retrieval_titles(nlu_processor, sentences: Sequence[str]) -> List[List[str]]:
    _require_pandas()
    if not sentences:
        return []

    retriever = nlu_processor.models.retrieval_engine.retriever
    query_df = pd.DataFrame(
        {
            "qid": [str(idx) for idx in range(len(sentences))],
            "query": [sentence.translate(str.maketrans("", "", string.punctuation)) for sentence in sentences],
        }
    )
    results = retriever.transform(query_df)
    titles_by_qid: Dict[str, List[str]] = {str(idx): [] for idx in range(len(sentences))}

    if "qid" in results.columns and "title" in results.columns:
        for qid, frame in results.groupby("qid", sort=False):
            titles_by_qid[str(qid)] = frame["title"].fillna("").astype(str).tolist()

    return [titles_by_qid[str(idx)] for idx in range(len(sentences))]


def _collapse_qa_candidates(items: Sequence[Tuple[str, str, float]]) -> List[str]:
    final_candidates: List[Tuple[str, str, float]] = []
    for candidate in sorted(items, key=lambda item: item[2], reverse=True):
        answer = (candidate[1] or "").lower()
        if not answer:
            continue
        redundant = False
        for _, chosen_answer, _ in final_candidates:
            chosen = (chosen_answer or "").lower()
            if answer in chosen or chosen in answer:
                redundant = True
                break
        if not redundant:
            final_candidates.append(candidate)

    seen = set()
    unique_answers = []
    for _, answer, _ in final_candidates:
        key = (answer or "").lower()
        if key and key not in seen:
            seen.add(key)
            unique_answers.append(answer)
    return unique_answers


def _batch_qa_for_intent(
    nlu_processor,
    intent: str,
    sentences: Sequence[str],
    intent_slot_dict: Mapping[str, Any],
    batch_size: int = 16,
) -> List[Tuple[List[str], Dict[str, float], Dict[str, List[str]]]]:
    slots = intent_slot_dict.get(intent, {}).get("slots", {})
    if not slots:
        return [([], {}, {}) for _ in sentences]

    per_sentence_items: List[List[Tuple[str, str, float]]] = [[] for _ in sentences]
    per_sentence_scores: List[Dict[str, float]] = [dict() for _ in sentences]
    per_sentence_spans: List[Dict[str, List[str]]] = [dict() for _ in sentences]

    for slot_label, slot_info in slots.items():
        qa_inputs = [{"question": slot_info["question"], "context": sentence} for sentence in sentences]
        outputs = nlu_processor.models.qa_model(qa_inputs, batch_size=batch_size)
        outputs = _normalize_classifier_outputs(outputs, len(sentences))

        for idx, output in enumerate(outputs):
            answer = sanitize_edges(output.get("answer", "") or "")
            score = float(output.get("score", 0.0) or 0.0)
            per_sentence_scores[idx][slot_label] = score
            per_sentence_spans[idx][slot_label] = [answer] if answer else []
            if answer:
                per_sentence_items[idx].append((slot_label, answer, score))

    bundled: List[Tuple[List[str], Dict[str, float], Dict[str, List[str]]]] = []
    for idx in range(len(sentences)):
        unique_answers = _collapse_qa_candidates(per_sentence_items[idx])
        bundled.append((unique_answers, per_sentence_scores[idx], per_sentence_spans[idx]))
    return bundled


def _project_slot_score_columns(df_slot_score: pd.DataFrame, selected_tokens: Sequence[str]) -> pd.DataFrame:
    _require_pandas()
    kept_tokens = [token for token in selected_tokens if token in df_slot_score.columns]
    if not kept_tokens:
        return pd.DataFrame(index=df_slot_score.index)
    return df_slot_score.loc[:, kept_tokens].copy()


def _build_slot_score_matrices_batch(
    nlu_processor,
    intent: str,
    sentences: Sequence[str],
    final_spans_per_sentence: Sequence[Sequence[str]],
    intent_slot_dict: Mapping[str, Any],
    batch_size: int = 16,
) -> List[pd.DataFrame]:
    _require_pandas()
    slots = intent_slot_dict.get(intent, {}).get("slots", {})
    slot_names = list(slots.keys())
    if not slot_names:
        return [pd.DataFrame() for _ in sentences]

    union_tokens: List[str] = []
    seen_tokens = set()
    for token_list in final_spans_per_sentence:
        for token in token_list:
            if token not in seen_tokens:
                seen_tokens.add(token)
                union_tokens.append(token)

    if not union_tokens:
        return [pd.DataFrame(index=slot_names) for _ in sentences]

    per_sentence_frames = [
        pd.DataFrame(0.0, index=slot_names, columns=list(token_list), dtype=float)
        for token_list in final_spans_per_sentence
    ]

    for slot_name, slot_info in slots.items():
        outputs = nlu_processor.models.classifier(
            list(sentences),
            union_tokens,
            hypothesis_template=slot_info["hypothesis"],
            multi_label=True,
            batch_size=batch_size,
        )
        outputs = _normalize_classifier_outputs(outputs, len(sentences))

        for idx, output in enumerate(outputs):
            score_map = {
                label: float(score)
                for label, score in zip(output.get("labels", []), output.get("scores", []))
            }
            for token in final_spans_per_sentence[idx]:
                per_sentence_frames[idx].at[slot_name, token] = score_map.get(token, 0.0)

    return per_sentence_frames


def _slot_assignment_to_jsonable(slot_assignment: Mapping[str, Tuple[str, float]]) -> Tuple[Dict[str, List[str]], Dict[str, float]]:
    slot_values: Dict[str, List[str]] = {}
    slot_scores: Dict[str, float] = {}
    for slot_name, (token, score) in slot_assignment.items():
        slot_scores[slot_name] = round(float(score), 3)
        if token and token != "NULL":
            slot_values[slot_name] = [token]
    return slot_values, slot_scores


def _run_single_sentence_postprocessing(
    nlu_processor,
    sentence: str,
    df_slot_score: pd.DataFrame,
    qa_slot_scores: Mapping[str, float],
    qa_slot_spans: Mapping[str, List[str]],
) -> Dict[str, Tuple[str, float]]:
    if df_slot_score is None or df_slot_score.empty:
        return {}

    grouping_df_slot_score, _ = build_alias_collapsed_grouping_view(df_slot_score, sentence)
    slot_threshold = getattr(
        nlu_processor,
        "tau_slots",
        getattr(nlu_processor, "SLOT_THRESHOLD", 0.5),
    )
    new_final_slot_tokens = process_until_no_more_groups(
        grouping_df_slot_score,
        threshold=slot_threshold,
    )
    if not new_final_slot_tokens:
        return {}

    final_df_slot_score = _project_slot_score_columns(grouping_df_slot_score, new_final_slot_tokens)
    if final_df_slot_score.empty:
        return {}

    final_df_slot_score = add_row_scores_masked(final_df_slot_score, qa_slot_scores, qa_slot_spans)
    if final_df_slot_score.empty:
        return {}

    return nlu_processor.slot_associations(final_df_slot_score)


def run_semantic_batch_inference(
    nlu_processor,
    semantic_view: pd.DataFrame,
    intent_slot_dict: Mapping[str, Any],
    *,
    use_gold_intent_for_slots: bool = True,
    intent_batch_size: int = 16,
    spacy_batch_size: int = 64,
    spacy_n_process: int = 1,
    qa_batch_size: int = 16,
    slot_sentence_batch_size: int = 8,
    nli_batch_size: int = 16,
    unknown_intent_label: str = "UNK_INTENT",
    intent_candidates_by_dataset: Mapping[str, Sequence[str]] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    _require_pandas()
    if semantic_view.empty:
        return semantic_view.copy(), {
            "intent_seconds": 0.0,
            "slot_seconds": 0.0,
            "n_rows": 0,
            "tau_intent": getattr(
                nlu_processor,
                "tau_intent",
                getattr(nlu_processor, "INTENT_THRESHOLD", 0.0),
            ),
            "tau_slots": getattr(
                nlu_processor,
                "tau_slots",
                getattr(nlu_processor, "SLOT_THRESHOLD", 0.5),
            ),
        }

    result_view = semantic_view.copy()
    dataset_names_in_view = [str(name) for name in pd.unique(result_view["dataset_name"].astype(str))]
    if intent_candidates_by_dataset is None:
        if len(dataset_names_in_view) > 1:
            raise ValueError(
                "run_semantic_batch_inference(...) requires intent_candidates_by_dataset "
                "when semantic_view contains multiple datasets. This prevents mixing "
                "candidate intent labels across datasets during batch inference."
            )
        semantic_intents = list(intent_slot_dict.keys())
        intent_candidates_by_dataset = {
            dataset_names_in_view[0]: semantic_intents,
        }
    else:
        intent_candidates_by_dataset = {
            str(dataset_name): list(candidate_intents)
            for dataset_name, candidate_intents in intent_candidates_by_dataset.items()
        }
        missing_datasets = [
            dataset_name
            for dataset_name in dataset_names_in_view
            if dataset_name not in intent_candidates_by_dataset
        ]
        if missing_datasets:
            raise KeyError(
                "Missing intent candidate lists for dataset(s): "
                + ", ".join(sorted(missing_datasets))
            )
        empty_datasets = [
            dataset_name
            for dataset_name in dataset_names_in_view
            if not intent_candidates_by_dataset[dataset_name]
        ]
        if empty_datasets:
            raise ValueError(
                "Intent candidate lists cannot be empty for dataset(s): "
                + ", ".join(sorted(empty_datasets))
            )

    start_intent = time.perf_counter()
    result_view["pred_intent"] = ""
    result_view["pred_intent_score"] = 0.0
    for dataset_name, group_frame in result_view.groupby("dataset_name", sort=False):
        candidate_intents = intent_candidates_by_dataset[str(dataset_name)]
        intent_predictions = batch_intent_recognition(
            nlu_processor,
            group_frame["sentence"].tolist(),
            candidate_intents,
            batch_size=intent_batch_size,
            unknown_label=unknown_intent_label,
        )
        result_view.loc[group_frame.index, "pred_intent"] = [
            label for label, _ in intent_predictions
        ]
        result_view.loc[group_frame.index, "pred_intent_score"] = [
            score for _, score in intent_predictions
        ]
    intent_seconds = time.perf_counter() - start_intent

    if use_gold_intent_for_slots:
        result_view["slot_eval_intent"] = result_view["semantic_intent"]
    else:
        result_view["slot_eval_intent"] = result_view["pred_intent"]

    pred_columns = [column for column in result_view.columns if column.startswith("pred__")]
    for column in pred_columns:
        result_view[column] = result_view[column].fillna("")
    result_view["pred_slots_semantic_json"] = "{}"
    result_view["pred_slots_score_json"] = "{}"

    start_slots = time.perf_counter()
    group_sizes: Dict[str, int] = {}
    grouped = result_view.groupby("slot_eval_intent", sort=False)
    for slot_intent, group_frame in grouped:
        group_sizes[str(slot_intent)] = len(group_frame)
        if slot_intent == unknown_intent_label or slot_intent not in intent_slot_dict:
            continue

        sentences = group_frame["sentence"].tolist()
        docs = list(
            nlu_processor.models.nlp.pipe(
                sentences,
                batch_size=spacy_batch_size,
                n_process=spacy_n_process,
            )
        )
        retrieval_titles = _batch_retrieval_titles(nlu_processor, sentences)
        qa_results = _batch_qa_for_intent(
            nlu_processor,
            slot_intent,
            sentences,
            intent_slot_dict,
            batch_size=qa_batch_size,
        )

        final_spans_per_sentence: List[List[str]] = []
        per_sentence_payloads: List[Tuple[Dict[str, float], Dict[str, List[str]]]] = []
        for doc, sentence, titles, qa_bundle in zip(docs, sentences, retrieval_titles, qa_results):
            pos_tokens = _extract_slot_tokens_from_doc(doc, slot_intent, intent_slot_dict)
            entities = _extract_entities_from_doc(doc)
            matching_titles = find_all_matching_substrings(titles, sentence)
            matching_titles = _filter_doctitles_for_intent(
                nlu_processor.models.nlp,
                matching_titles,
                slot_intent,
                intent_slot_dict,
            )
            relevant_spans = remove_duplicates_and_substrings(pos_tokens + matching_titles)
            qa_answers, qa_slot_scores, qa_slot_spans = qa_bundle
            final_spans = remove_duplicates(relevant_spans + entities + qa_answers)
            final_spans_per_sentence.append(final_spans)
            per_sentence_payloads.append((qa_slot_scores, qa_slot_spans))

        group_indices = list(group_frame.index)
        for start in range(0, len(group_indices), slot_sentence_batch_size):
            end = start + slot_sentence_batch_size
            batch_indices = group_indices[start:end]
            batch_sentences = result_view.loc[batch_indices, "sentence"].tolist()
            batch_spans = final_spans_per_sentence[start:end]
            batch_payloads = per_sentence_payloads[start:end]
            batch_scores = _build_slot_score_matrices_batch(
                nlu_processor,
                slot_intent,
                batch_sentences,
                batch_spans,
                intent_slot_dict,
                batch_size=nli_batch_size,
            )

            for row_index, sentence, df_slot_score, qa_payload in zip(
                batch_indices,
                batch_sentences,
                batch_scores,
                batch_payloads,
            ):
                qa_slot_scores, qa_slot_spans = qa_payload
                slot_assignment = _run_single_sentence_postprocessing(
                    nlu_processor,
                    sentence,
                    df_slot_score,
                    qa_slot_scores,
                    qa_slot_spans,
                )
                slot_values, slot_scores = _slot_assignment_to_jsonable(slot_assignment)
                result_view.at[row_index, "pred_slots_semantic_json"] = json.dumps(slot_values, ensure_ascii=False)
                result_view.at[row_index, "pred_slots_score_json"] = json.dumps(slot_scores, ensure_ascii=False)

                for slot_name in intent_slot_dict[slot_intent]["slots"].keys():
                    column_name = f"pred__{slot_name}"
                    if column_name not in result_view.columns:
                        result_view[column_name] = ""
                    if slot_name in slot_values:
                        result_view.at[row_index, column_name] = slot_values[slot_name][0]
                    else:
                        result_view.at[row_index, column_name] = ""

    slot_seconds = time.perf_counter() - start_slots
    stats = {
        "intent_seconds": intent_seconds,
        "slot_seconds": slot_seconds,
        "total_seconds": intent_seconds + slot_seconds,
        "n_rows": len(result_view),
        "rows_per_second": (len(result_view) / (intent_seconds + slot_seconds))
        if (intent_seconds + slot_seconds) > 0
        else 0.0,
        "group_sizes": group_sizes,
        "tau_intent": getattr(
            nlu_processor,
            "tau_intent",
            getattr(nlu_processor, "INTENT_THRESHOLD", 0.0),
        ),
        "tau_slots": getattr(
            nlu_processor,
            "tau_slots",
            getattr(nlu_processor, "SLOT_THRESHOLD", 0.5),
        ),
        "intent_candidate_counts_by_dataset": {
            dataset_name: len(intent_candidates_by_dataset[dataset_name])
            for dataset_name in dataset_names_in_view
        },
    }
    return result_view, stats


def _normalized_slot_json(slot_json: str | Mapping[str, Any] | None) -> Dict[str, List[str]]:
    if not slot_json:
        return {}
    payload = json.loads(slot_json) if isinstance(slot_json, str) else dict(slot_json)
    normalized: Dict[str, List[str]] = {}
    for slot_name, values in payload.items():
        normalized_values = sorted(
            {norm_text(value) for value in values if norm_text(value)}
        )
        if normalized_values:
            normalized[slot_name] = normalized_values
    return normalized


def compute_semantic_exact_match(semantic_view: pd.DataFrame) -> Dict[str, Any]:
    if semantic_view.empty:
        return {
            "rows": 0,
            "slot_exact_match": 0.0,
            "joint_intent_slot_exact_match": 0.0,
            "slot_exact_match_count": 0,
            "joint_exact_match_count": 0,
        }

    slot_matches = 0
    joint_matches = 0
    for _, row in semantic_view.iterrows():
        gold = _normalized_slot_json(row.get("gold_slots_semantic_json"))
        pred = _normalized_slot_json(row.get("pred_slots_semantic_json"))
        slot_ok = gold == pred
        intent_ok = row.get("semantic_intent") == row.get("pred_intent")
        slot_matches += int(slot_ok)
        joint_matches += int(slot_ok and intent_ok)

    total = len(semantic_view)
    return {
        "rows": total,
        "slot_exact_match": slot_matches / total,
        "joint_intent_slot_exact_match": joint_matches / total,
        "slot_exact_match_count": slot_matches,
        "joint_exact_match_count": joint_matches,
    }


def merge_predictions_into_dataset_view(
    dataset_view: pd.DataFrame,
    semantic_view: pd.DataFrame,
) -> pd.DataFrame:
    _require_pandas()
    merge_keys = ["example_id", "source_file", "dataset_name", "dataset_intent", "semantic_intent"]
    prediction_columns = [
        column for column in semantic_view.columns
        if column.startswith("pred__")
        or column in {
            "pred_intent",
            "pred_intent_score",
            "slot_eval_intent",
            "pred_slots_semantic_json",
            "pred_slots_score_json",
        }
    ]
    return dataset_view.merge(
        semantic_view[merge_keys + prediction_columns],
        on=merge_keys,
        how="left",
    )
