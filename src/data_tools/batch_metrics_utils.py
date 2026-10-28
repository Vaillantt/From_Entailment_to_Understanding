from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - depends on active environment
    pd = None

from src.data_tools.batch_eval_utils import (
    DEFAULT_ALIAS_REGISTRY_PATH,
    EVALUATION_ROOT,
    REPO_ROOT,
    load_semantic_alias_registry,
)


DEFAULT_METRICS_OUTPUT_DIR = EVALUATION_ROOT / "metrics"
REQUIRED_RESULTS_VIEW_COLUMNS = {
    "example_id",
    "dataset_name",
    "source_file",
    "dataset_intent",
    "semantic_intent",
    "sentence",
    "pred_intent",
    "pred_slots_semantic_json",
}
UNIQUE_ROW_KEYS = [
    "example_id",
    "dataset_name",
    "source_file",
    "dataset_intent",
    "semantic_intent",
]


def _require_pandas() -> None:
    if pd is None:
        raise ModuleNotFoundError(
            "pandas is required for batch_metrics_utils. Install it in the active notebook/kernel environment."
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


def _safe_div(numerator: float, denominator: float) -> float:
    return (numerator / denominator) if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0


def _mean_metric(values: Sequence[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0


def tokenize_with_spans(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    tokens: List[str] = []
    spans: List[Tuple[int, int]] = []
    for match in re.finditer(r"\S+", text):
        tokens.append(match.group(0))
        spans.append((match.start(), match.end()))
    return tokens, spans


def slot2tag(slot_name: str) -> str:
    return slot_name.replace(" ", "_")


def normalize_token(text: str) -> str:
    return strip_accents(text).lower()


def bio_to_chunks(tags: Sequence[str]) -> List[Tuple[str, int, int]]:
    chunks: List[Tuple[str, int, int]] = []
    index = 0
    while index < len(tags):
        tag = tags[index]
        if tag.startswith("B-"):
            chunk_type = tag[2:]
            end = index + 1
            while end < len(tags) and tags[end] == f"I-{chunk_type}":
                end += 1
            chunks.append((chunk_type, index, end))
            index = end
        else:
            index += 1
    return chunks


def assign_span_on_tokens(
    tags: List[str],
    sent_tokens: Sequence[str],
    pred_value: str,
    slot_name: str,
) -> None:
    if not pred_value:
        return
    pred_tokens, _ = tokenize_with_spans(pred_value)
    pred_tokens = [normalize_token(token) for token in pred_tokens if token]
    if not pred_tokens:
        return
    normalized_sentence = [normalize_token(token) for token in sent_tokens]
    window = len(pred_tokens)
    for start in range(0, len(normalized_sentence) - window + 1):
        if normalized_sentence[start:start + window] == pred_tokens and all(
            tags[pos] == "O" for pos in range(start, start + window)
        ):
            tag_name = slot2tag(slot_name)
            tags[start] = f"B-{tag_name}"
            for pos in range(start + 1, start + window):
                tags[pos] = f"I-{tag_name}"
            break


def _normalize_slot_json(slot_json: str | Mapping[str, Any] | None) -> Dict[str, List[str]]:
    if not slot_json:
        return {}
    payload = json.loads(slot_json) if isinstance(slot_json, str) else dict(slot_json)
    normalized: Dict[str, List[str]] = {}
    for slot_name, values in payload.items():
        if isinstance(values, (str, int, float)):
            values = [values]
        ordered_values: List[str] = []
        seen = set()
        for value in values or []:
            normalized_value = norm_text(str(value))
            if normalized_value and normalized_value not in seen:
                seen.add(normalized_value)
                ordered_values.append(normalized_value)
        if ordered_values:
            normalized[slot_name] = ordered_values
    return normalized


def _resolve_source_path(source_file: str | Path) -> Path:
    source_path = Path(source_file)
    if source_path.is_absolute():
        return source_path
    return (REPO_ROOT / source_path).resolve()


def _normalize_filter_values(values: Sequence[str] | None) -> List[str]:
    return [str(value) for value in values or [] if str(value).strip()]


def _validate_results_view_columns(frame: "pd.DataFrame") -> None:
    missing = sorted(REQUIRED_RESULTS_VIEW_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(
            "Results CSV is missing required columns: "
            + ", ".join(missing)
        )


def load_prediction_results_csvs(
    csv_paths: Sequence[str | Path],
    dataset_names: Sequence[str] | None = None,
    source_files: Sequence[str] | None = None,
    semantic_intents: Sequence[str] | None = None,
) -> "pd.DataFrame":
    _require_pandas()
    csv_paths = [Path(path) for path in csv_paths]
    if not csv_paths:
        raise ValueError("At least one results_csv path is required.")

    frames: List["pd.DataFrame"] = []
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path, encoding="utf-8-sig", keep_default_na=False, na_filter=False)
        _validate_results_view_columns(frame)
        frame["__input_csv"] = str(Path(csv_path))
        frames.append(frame)

    results_view = pd.concat(frames, ignore_index=True)
    results_view = results_view.drop_duplicates(subset=UNIQUE_ROW_KEYS, keep="first").reset_index(drop=True)

    dataset_name_filter = _normalize_filter_values(dataset_names)
    source_file_filter = _normalize_filter_values(source_files)
    semantic_intent_filter = _normalize_filter_values(semantic_intents)

    if dataset_name_filter:
        results_view = results_view[results_view["dataset_name"].isin(dataset_name_filter)]
    if source_file_filter:
        results_view = results_view[results_view["source_file"].isin(source_file_filter)]
    if semantic_intent_filter:
        results_view = results_view[results_view["semantic_intent"].isin(semantic_intent_filter)]

    return results_view.reset_index(drop=True)


def load_semantic_prediction_views(
    csv_paths: Sequence[str | Path],
    dataset_names: Sequence[str] | None = None,
    source_files: Sequence[str] | None = None,
    semantic_intents: Sequence[str] | None = None,
) -> "pd.DataFrame":
    return load_prediction_results_csvs(
        csv_paths,
        dataset_names=dataset_names,
        source_files=source_files,
        semantic_intents=semantic_intents,
    )


def _get_raw_to_semantic_slot_map(
    dataset_name: str,
    dataset_intent: str,
    registry: Mapping[str, Any],
) -> Dict[str, str]:
    try:
        intent_cfg = registry[dataset_name]["intents"][dataset_intent]
    except KeyError as exc:
        raise KeyError(
            f"Could not find alias mapping for dataset='{dataset_name}' and raw intent='{dataset_intent}'."
        ) from exc
    return dict(intent_cfg.get("slot_map", {}))


def _build_gold_semantic_slot_values(
    data_list: Sequence[Mapping[str, Any]],
    raw_to_semantic_slot_map: Mapping[str, str],
) -> Dict[str, List[str]]:
    gold_slot_values: Dict[str, List[str]] = defaultdict(list)
    for span in data_list:
        raw_slot = span.get("entity")
        semantic_slot = raw_to_semantic_slot_map.get(raw_slot)
        normalized_value = norm_text(span.get("text", ""))
        if semantic_slot and normalized_value and normalized_value not in gold_slot_values[semantic_slot]:
            gold_slot_values[semantic_slot].append(normalized_value)
    return dict(gold_slot_values)


def build_gold_bio(
    data_list: Sequence[Mapping[str, Any]],
    tokens: Sequence[str],
    spans: Sequence[Tuple[int, int]],
    raw_to_semantic_slot_map: Mapping[str, str],
) -> List[str]:
    tags = ["O"] * len(tokens)
    cursor = 0
    for segment in data_list:
        text = segment.get("text", "") or ""
        seg_start, seg_end = cursor, cursor + len(text)
        raw_slot = segment.get("entity")
        semantic_slot = raw_to_semantic_slot_map.get(raw_slot)
        if semantic_slot and text.strip():
            tag_name = slot2tag(semantic_slot)
            first = True
            for index, (start, end) in enumerate(spans):
                if start >= seg_start and end <= seg_end and end > start:
                    tags[index] = ("B-" if first else "I-") + tag_name
                    first = False
        cursor = seg_end
    return tags


def build_pred_bio(
    sentence: str,
    pred_slots_semantic: Mapping[str, Sequence[str]],
) -> Tuple[List[str], List[str], List[Tuple[int, int]]]:
    tokens, spans = tokenize_with_spans(sentence)
    tags = ["O"] * len(tokens)
    for slot_name, values in pred_slots_semantic.items():
        for value in values:
            assign_span_on_tokens(tags, tokens, value, slot_name)
    return tags, tokens, spans


def _row_to_raw_example(
    row: Mapping[str, Any],
    dataset_cache: Dict[Path, Dict[str, Any]],
) -> Tuple[Path, Mapping[str, Any], Sequence[Mapping[str, Any]]]:
    resolved_source = _resolve_source_path(row["source_file"])
    if resolved_source not in dataset_cache:
        dataset_cache[resolved_source] = _load_json_file(resolved_source)

    payload = dataset_cache[resolved_source]
    dataset_intent = row["dataset_intent"]
    if dataset_intent not in payload:
        raise KeyError(
            f"Raw intent '{dataset_intent}' not found in source file '{resolved_source}'."
        )

    examples = payload[dataset_intent]
    example_id = int(row["example_id"])
    if not isinstance(examples, list) or not (0 <= example_id < len(examples)):
        raise IndexError(
            f"Could not locate example_id={example_id} for raw intent '{dataset_intent}' in '{resolved_source}'."
        )

    example = examples[example_id]
    data_list = example.get("data", [])
    reconstructed = reconstruct_sentence(data_list)
    row_sentence = str(row["sentence"])
    if reconstructed != row_sentence:
        raise ValueError(
            "Sentence mismatch while rebuilding gold annotations:\n"
            f"- source_file: {resolved_source}\n"
            f"- dataset_intent: {dataset_intent}\n"
            f"- example_id: {example_id}\n"
            f"- csv sentence: {row_sentence!r}\n"
            f"- raw sentence: {reconstructed!r}"
        )

    return resolved_source, example, data_list


def _validate_registry_coverage(
    semantic_view: "pd.DataFrame",
    registry: Mapping[str, Any],
) -> None:
    for dataset_name in sorted(semantic_view["dataset_name"].drop_duplicates().tolist()):
        if dataset_name not in registry:
            raise KeyError(f"Dataset '{dataset_name}' is missing from the semantic alias registry.")


def build_evaluation_examples(
    semantic_view: "pd.DataFrame",
    registry: Mapping[str, Any] | None = None,
) -> "pd.DataFrame":
    _require_pandas()
    registry = registry or load_semantic_alias_registry(DEFAULT_ALIAS_REGISTRY_PATH)
    if semantic_view.empty:
        return pd.DataFrame()

    _validate_registry_coverage(semantic_view, registry)
    dataset_cache: Dict[Path, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []

    for _, row in semantic_view.iterrows():
        resolved_source, _example, data_list = _row_to_raw_example(row, dataset_cache)
        raw_to_semantic_slot_map = _get_raw_to_semantic_slot_map(
            str(row["dataset_name"]),
            str(row["dataset_intent"]),
            registry,
        )
        sentence = str(row["sentence"])
        tokens, token_spans = tokenize_with_spans(sentence)
        gold_bio = build_gold_bio(data_list, tokens, token_spans, raw_to_semantic_slot_map)
        gold_chunks = bio_to_chunks(gold_bio)
        gold_slot_values = _build_gold_semantic_slot_values(data_list, raw_to_semantic_slot_map)

        pred_slot_values = _normalize_slot_json(row.get("pred_slots_semantic_json"))
        pred_bio, _pred_tokens, _pred_spans = build_pred_bio(sentence, pred_slot_values)
        pred_chunks = bio_to_chunks(pred_bio)

        gold_chunk_set = set(gold_chunks)
        pred_chunk_set = set(pred_chunks)
        chunk_tp = len(gold_chunk_set & pred_chunk_set)
        chunk_fp = len(pred_chunk_set - gold_chunk_set)
        chunk_fn = len(gold_chunk_set - pred_chunk_set)

        rows.append(
            {
                "example_id": int(row["example_id"]),
                "dataset_name": row["dataset_name"],
                "source_file": row["source_file"],
                "resolved_source_file": str(resolved_source),
                "dataset_intent": row["dataset_intent"],
                "semantic_intent": row["semantic_intent"],
                "pred_intent": row.get("pred_intent", ""),
                "sentence": sentence,
                "intent_correct": str(row["semantic_intent"]) == str(row.get("pred_intent", "")),
                "tokens_json": json.dumps(tokens, ensure_ascii=False),
                "gold_slots_semantic_json_rebuilt": json.dumps(gold_slot_values, ensure_ascii=False),
                "pred_slots_semantic_json_normalized": json.dumps(pred_slot_values, ensure_ascii=False),
                "gold_bio_json": json.dumps(gold_bio, ensure_ascii=False),
                "pred_bio_json": json.dumps(pred_bio, ensure_ascii=False),
                "gold_chunks_json": json.dumps(gold_chunks, ensure_ascii=False),
                "pred_chunks_json": json.dumps(pred_chunks, ensure_ascii=False),
                "slot_chunk_exact": gold_chunk_set == pred_chunk_set,
                "slot_chunk_tp": chunk_tp,
                "slot_chunk_fp": chunk_fp,
                "slot_chunk_fn": chunk_fn,
            }
        )

    return pd.DataFrame(rows)


def compute_intent_accuracy_reports(
    evaluation_examples: "pd.DataFrame",
) -> Tuple["pd.DataFrame", "pd.DataFrame", "pd.DataFrame"]:
    _require_pandas()
    if evaluation_examples.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    gold_intents = evaluation_examples["semantic_intent"].astype(str)
    pred_intents = evaluation_examples["pred_intent"].astype(str)
    labels = sorted(set(gold_intents.tolist()) | set(pred_intents.tolist()), key=lambda item: item.lower())
    gold_labels = sorted(set(gold_intents.tolist()), key=lambda item: item.lower())

    total_rows = len(evaluation_examples)
    intent_correct = int(evaluation_examples["intent_correct"].sum())

    per_intent_rows: List[Dict[str, Any]] = []
    for label in labels:
        gold_mask = gold_intents == label
        pred_mask = pred_intents == label
        support = int(gold_mask.sum())
        predicted = int(pred_mask.sum())
        tp = int((gold_mask & pred_mask).sum())
        fp = int((~gold_mask & pred_mask).sum())
        fn = int((gold_mask & ~pred_mask).sum())
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        per_intent_rows.append(
            {
                "semantic_intent": label,
                "support": support,
                "predicted": predicted,
                "correct": tp,
                "errors": support - tp,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": _f1(precision, recall),
                "accuracy": _safe_div(tp, support),
            }
        )

    per_intent = pd.DataFrame(per_intent_rows).sort_values(
        by=["support", "semantic_intent"],
        ascending=[False, True],
    ).reset_index(drop=True)
    macro_rows = per_intent[per_intent["semantic_intent"].isin(gold_labels)]
    macro_precision = _mean_metric(macro_rows["precision"].astype(float).tolist())
    macro_recall = _mean_metric(macro_rows["recall"].astype(float).tolist())
    macro_f1 = _mean_metric(macro_rows["f1"].astype(float).tolist())
    micro_fp = total_rows - intent_correct
    micro_fn = total_rows - intent_correct
    micro_precision = _safe_div(intent_correct, intent_correct + micro_fp)
    micro_recall = _safe_div(intent_correct, intent_correct + micro_fn)
    micro_f1 = _f1(micro_precision, micro_recall)
    summary = pd.DataFrame(
        [
            {
                "rows": total_rows,
                "intent_count": len(gold_labels),
                "intent_correct": intent_correct,
                "intent_errors": total_rows - intent_correct,
                "intent_accuracy": _safe_div(intent_correct, total_rows),
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f1": macro_f1,
                "micro_precision": micro_precision,
                "micro_recall": micro_recall,
                "micro_f1": micro_f1,
            }
        ]
    )

    confusion = pd.crosstab(
        gold_intents.rename("gold_intent"),
        pred_intents.rename("pred_intent"),
        dropna=False,
    )
    confusion = confusion.reindex(index=labels, columns=labels, fill_value=0)
    confusion.index.name = "gold_intent"
    confusion.columns.name = "pred_intent"

    return summary, per_intent, confusion


def _chunks_from_json(chunks_json: str | Sequence[Sequence[Any]]) -> List[Tuple[str, int, int]]:
    payload = json.loads(chunks_json) if isinstance(chunks_json, str) else list(chunks_json)
    return [(str(item[0]), int(item[1]), int(item[2])) for item in payload]


def compute_slot_conll_reports(
    evaluation_examples: "pd.DataFrame",
) -> Tuple["pd.DataFrame", "pd.DataFrame", "pd.DataFrame"]:
    _require_pandas()
    if evaluation_examples.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    global_tp = 0
    global_fp = 0
    global_fn = 0
    per_slot_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})
    per_intent_counts: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"TP": 0, "FP": 0, "FN": 0, "rows": 0}
    )

    for _, row in evaluation_examples.iterrows():
        semantic_intent = str(row["semantic_intent"])
        gold_chunks = set(_chunks_from_json(row["gold_chunks_json"]))
        pred_chunks = set(_chunks_from_json(row["pred_chunks_json"]))
        tp_chunks = gold_chunks & pred_chunks
        fp_chunks = pred_chunks - gold_chunks
        fn_chunks = gold_chunks - pred_chunks

        global_tp += len(tp_chunks)
        global_fp += len(fp_chunks)
        global_fn += len(fn_chunks)
        per_intent_counts[semantic_intent]["TP"] += len(tp_chunks)
        per_intent_counts[semantic_intent]["FP"] += len(fp_chunks)
        per_intent_counts[semantic_intent]["FN"] += len(fn_chunks)
        per_intent_counts[semantic_intent]["rows"] += 1

        for slot_name, _start, _end in tp_chunks:
            per_slot_counts[slot_name]["TP"] += 1
        for slot_name, _start, _end in fp_chunks:
            per_slot_counts[slot_name]["FP"] += 1
        for slot_name, _start, _end in fn_chunks:
            per_slot_counts[slot_name]["FN"] += 1

    global_precision = _safe_div(global_tp, global_tp + global_fp)
    global_recall = _safe_div(global_tp, global_tp + global_fn)
    global_f1 = _f1(global_precision, global_recall)

    per_intent_rows: List[Dict[str, Any]] = []
    for semantic_intent in sorted(per_intent_counts.keys(), key=lambda item: item.lower()):
        counts = per_intent_counts[semantic_intent]
        tp = counts["TP"]
        fp = counts["FP"]
        fn = counts["FN"]
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        per_intent_rows.append(
            {
                "semantic_intent": semantic_intent,
                "rows": counts["rows"],
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "support_gold": tp + fn,
                "support_pred": tp + fp,
                "precision": precision,
                "recall": recall,
                "f1": _f1(precision, recall),
            }
        )

    per_intent = pd.DataFrame(per_intent_rows).sort_values(
        by=["support_gold", "semantic_intent"],
        ascending=[False, True],
    ).reset_index(drop=True)
    macro_precision = _mean_metric(per_intent["precision"].astype(float).tolist())
    macro_recall = _mean_metric(per_intent["recall"].astype(float).tolist())
    macro_f1 = _mean_metric(per_intent["f1"].astype(float).tolist())

    global_summary = pd.DataFrame(
        [
            {
                "rows": len(evaluation_examples),
                "intent_count": len(per_intent),
                "tp": global_tp,
                "fp": global_fp,
                "fn": global_fn,
                "precision": macro_precision,
                "recall": macro_recall,
                "f1": macro_f1,
                "micro_tp": global_tp,
                "micro_fp": global_fp,
                "micro_fn": global_fn,
                "micro_precision": global_precision,
                "micro_recall": global_recall,
                "micro_f1": global_f1,
            }
        ]
    )

    per_slot_rows: List[Dict[str, Any]] = []
    for slot_name in sorted(per_slot_counts.keys(), key=lambda item: item.lower()):
        tp = per_slot_counts[slot_name]["TP"]
        fp = per_slot_counts[slot_name]["FP"]
        fn = per_slot_counts[slot_name]["FN"]
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        per_slot_rows.append(
            {
                "semantic_slot": slot_name,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "support_gold": tp + fn,
                "support_pred": tp + fp,
                "precision": precision,
                "recall": recall,
                "f1": _f1(precision, recall),
            }
        )

    if per_slot_rows:
        per_slot = pd.DataFrame(per_slot_rows).sort_values(
            by=["support_gold", "semantic_slot"],
            ascending=[False, True],
        ).reset_index(drop=True)
    else:
        per_slot = pd.DataFrame(
            columns=[
                "semantic_slot",
                "tp",
                "fp",
                "fn",
                "support_gold",
                "support_pred",
                "precision",
                "recall",
                "f1",
            ]
        )

    return global_summary, per_slot, per_intent


def build_overall_metrics_summary(
    intent_summary: "pd.DataFrame",
    slot_conll_summary: "pd.DataFrame",
) -> "pd.DataFrame":
    _require_pandas()
    if intent_summary.empty or slot_conll_summary.empty:
        return pd.DataFrame()

    intent_row = intent_summary.iloc[0]
    slot_row = slot_conll_summary.iloc[0]
    return pd.DataFrame(
        [
            {
                "rows": int(intent_row["rows"]),
                "intent_accuracy": float(intent_row["intent_accuracy"]),
                "intent_macro_precision": float(intent_row["macro_precision"]),
                "intent_macro_recall": float(intent_row["macro_recall"]),
                "intent_macro_f1": float(intent_row["macro_f1"]),
                "intent_micro_precision": float(intent_row["micro_precision"]),
                "intent_micro_recall": float(intent_row["micro_recall"]),
                "intent_micro_f1": float(intent_row["micro_f1"]),
                "slot_conll_precision": float(slot_row["precision"]),
                "slot_conll_recall": float(slot_row["recall"]),
                "slot_conll_f1": float(slot_row["f1"]),
                "slot_conll_micro_precision": float(slot_row["micro_precision"]),
                "slot_conll_micro_recall": float(slot_row["micro_recall"]),
                "slot_conll_micro_f1": float(slot_row["micro_f1"]),
            }
        ]
    )


def compute_batch_metrics_reports(
    semantic_view: "pd.DataFrame",
    registry: Mapping[str, Any] | None = None,
) -> Dict[str, "pd.DataFrame"]:
    registry = registry or load_semantic_alias_registry(DEFAULT_ALIAS_REGISTRY_PATH)
    evaluation_examples = build_evaluation_examples(semantic_view, registry=registry)
    intent_summary, intent_by_label, intent_confusion = compute_intent_accuracy_reports(evaluation_examples)
    slot_conll_summary, slot_conll_by_label, slot_conll_by_intent = compute_slot_conll_reports(evaluation_examples)
    overall_summary = build_overall_metrics_summary(intent_summary, slot_conll_summary)

    return {
        "semantic_view": semantic_view,
        "evaluation_examples": evaluation_examples,
        "overall_summary": overall_summary,
        "intent_summary": intent_summary,
        "intent_by_label": intent_by_label,
        "intent_confusion": intent_confusion,
        "slot_conll_summary": slot_conll_summary,
        "slot_conll_by_label": slot_conll_by_label,
        "slot_conll_by_intent": slot_conll_by_intent,
    }


def export_metrics_reports(
    reports: Mapping[str, "pd.DataFrame"],
    output_dir: str | Path,
    include_examples: bool = False,
) -> Dict[str, Path]:
    _require_pandas()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    export_map = {
        "overall_summary": "overall_summary.csv",
        "intent_summary": "intent_summary.csv",
        "intent_by_label": "intent_by_label.csv",
        "intent_confusion": "intent_confusion.csv",
        "slot_conll_summary": "slot_conll_summary.csv",
        "slot_conll_by_label": "slot_conll_by_label.csv",
        "slot_conll_by_intent": "slot_conll_by_intent.csv",
    }
    if include_examples:
        export_map["evaluation_examples"] = "evaluation_examples_debug.csv"

    written: Dict[str, Path] = {}
    for report_name, filename in export_map.items():
        frame = reports.get(report_name)
        if frame is None or frame.empty:
            continue
        path = output_root / filename
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        written[report_name] = path

    return written
