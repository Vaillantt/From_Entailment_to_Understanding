from __future__ import annotations

import csv
import html
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ROOT = REPO_ROOT / "evaluation"
OUTPUT_PATH = EVALUATION_ROOT / "reports" / "batch_eval_unified_dashboard.html"

SUBSAMPLING_METHOD = "stratified_by_intent"
SUBSAMPLING_CONFIG = "r1000_seed42"

DATASETS = [
    ("field_operations", "Field Operations", "Primary domain analysis"),
    ("snips", "SNIPS", "Complementary benchmark"),
    ("atis", "ATIS", "Complementary benchmark"),
    ("multiwoz", "MultiWOZ", "Complementary benchmark"),
]

NORMAL_METRIC_FILES = [
    "overall_summary.csv",
    "intent_summary.csv",
    "intent_by_label.csv",
    "intent_confusion.csv",
    "slot_conll_summary.csv",
    "slot_conll_by_label.csv",
    "slot_conll_by_intent.csv",
]

SUBSAMPLING_FILES = [
    "summary.csv",
    "rounds.csv",
    "coverage.csv",
    "intent_by_label_rounds.csv",
    "slot_by_intent_rounds.csv",
    "slot_by_label_rounds.csv",
]

CURVE_METRICS = [
    ("intent_macro_f1", "Intent Macro-F1", "#1565c0"),
    ("slot_macro_f1_by_intent", "Slot Macro-F1 by Intent", "#00897b"),
    ("slot_micro_f1", "Slot Micro-F1", "#ef6c00"),
]

COUNT_COLUMNS = {
    "n",
    "n_per_intent",
    "primary_n",
    "round",
    "rounds",
    "rows",
    "rows_per_round",
    "round_rows",
    "rounds_requested",
    "rounds_executed",
    "seed",
    "support",
    "support_gold",
    "support_pred",
    "mean_support",
    "normal_support",
    "subsampling_support",
    "predicted",
    "correct",
    "errors",
    "tp",
    "fp",
    "fn",
    "intent_count",
    "sampled_per_round",
    "total_sampled_across_rounds",
    "csv_rows",
}


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def read_csv(path: Path, limit: int | None = None) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: List[Dict[str, str]] = []
        for index, row in enumerate(reader):
            if limit is not None and index >= limit:
                break
            rows.append(dict(row))
    return rows


def read_first_row(path: Path) -> Dict[str, str]:
    rows = read_csv(path, limit=1)
    return rows[0] if rows else {}


def count_csv_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return max(sum(1 for _line in handle) - 1, 0)


def file_info(path: Path) -> Dict[str, Any]:
    exists = path.exists()
    stat = path.stat() if exists else None
    return {
        "path": path,
        "exists": exists,
        "bytes": stat.st_size if stat else None,
        "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S") if stat else "",
        "rows": count_csv_rows(path) if exists else None,
    }


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def to_int(value: Any) -> int | None:
    number = to_float(value)
    return int(number) if number is not None else None


def to_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in value.lower()).strip("-")


def format_int(value: int | float | None) -> str:
    if value is None:
        return "-"
    return f"{int(round(float(value))):,}".replace(",", ".")


def format_bytes(value: int | None) -> str:
    if value is None:
        return "-"
    units = ["B", "KB", "MB", "GB"]
    amount = float(value)
    unit = units[0]
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            break
        amount /= 1024
    return f"{amount:.1f} {unit}" if unit != "B" else f"{int(amount)} B"


def format_metric(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.2f}%"


def format_decimal(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def format_cell(column: str, value: Any) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, str) and any(ch.isalpha() for ch in value) and not value.replace(".", "", 1).isdigit():
        return value
    numeric = to_float(value)
    if numeric is None:
        return str(value)
    key = column.lower()
    if key in COUNT_COLUMNS or key.endswith("_support") or key.endswith("_rows"):
        return format_int(numeric)
    if any(token in key for token in ("accuracy", "precision", "recall", "f1", "fdr", "rate", "std", "p2_5", "p97_5")):
        return format_metric(numeric)
    if key in {"mean", "min", "max", "median"}:
        return format_metric(numeric)
    return format_decimal(numeric)


def metric_interval(row: Mapping[str, Any] | None) -> str:
    if not row:
        return "-"
    mean = to_float(row.get("mean"))
    p2_5 = to_float(row.get("p2_5"))
    p97_5 = to_float(row.get("p97_5"))
    if mean is None:
        return "-"
    if p2_5 is None or p97_5 is None:
        return format_metric(mean)
    return f"{format_metric(mean)} [{format_metric(p2_5)}, {format_metric(p97_5)}]"


def table_html(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str] | None = None,
    *,
    compact: bool = False,
    preformatted: bool = False,
    limit: int | None = None,
) -> str:
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        return '<p class="muted">No rows available.</p>'
    columns = list(columns or rows[0].keys())
    parts = ['<div class="table-wrap">', '<table class="compact">' if compact else "<table>"]
    parts.append("<thead><tr>")
    for column in columns:
        parts.append(f"<th>{esc(column)}</th>")
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        for column in columns:
            value = row.get(column, "")
            text = str(value) if preformatted else format_cell(column, value)
            parts.append(f"<td>{esc(text)}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def sectioned_table_html(
    sections: Sequence[tuple[str, Sequence[Mapping[str, Any]]]],
    columns: Sequence[str],
    *,
    compact: bool = False,
) -> str:
    if not any(rows for _title, rows in sections):
        return '<p class="muted">No rows available.</p>'
    parts = ['<div class="table-wrap">', '<table class="compact">' if compact else "<table>"]
    parts.append("<thead><tr>")
    for column in columns:
        parts.append(f"<th>{esc(column)}</th>")
    parts.append("</tr></thead><tbody>")
    for title, rows in sections:
        parts.append(f'<tr class="table-section"><th colspan="{len(columns)}">{esc(title)}</th></tr>')
        if rows:
            for row in rows:
                parts.append("<tr>")
                for column in columns:
                    parts.append(f"<td>{esc(format_cell(column, row.get(column, '')))}</td>")
                parts.append("</tr>")
        else:
            parts.append(f'<tr><td colspan="{len(columns)}" class="muted">No rows available.</td></tr>')
    parts.append("</tbody></table></div>")
    return "".join(parts)


def metric_bar(value: float | None) -> str:
    if value is None:
        return '<div class="bar empty"><span>n/a</span></div>'
    width = max(0.0, min(100.0, value * 100.0))
    return (
        '<div class="bar">'
        f'<div class="bar-fill" style="width: {width:.2f}%"></div>'
        f'<span>{esc(format_metric(value))}</span>'
        "</div>"
    )


def scan_predictions(path: Path) -> Dict[str, Any]:
    result = {"rows": 0, "unknown_predictions": 0}
    if not path.exists():
        return result
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            result["rows"] += 1
            if row.get("pred_intent") == "UNK_INTENT":
                result["unknown_predictions"] += 1
    return result


def pick_float(row: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        value = to_float(row.get(key))
        if value is not None:
            return value
    return None


def load_normal_dataset(dataset: str, label: str, role: str) -> Dict[str, Any]:
    root = EVALUATION_ROOT / dataset
    metrics_dir = root / "metrics"
    results_dir = root / "results_csv"
    predictions_path = results_dir / f"{dataset}_predictions.csv"
    paths = {name: metrics_dir / name for name in NORMAL_METRIC_FILES}

    overall = read_first_row(paths["overall_summary.csv"])
    intent_summary = read_first_row(paths["intent_summary.csv"])
    slot_summary = read_first_row(paths["slot_conll_summary.csv"])
    intent_by_label = read_csv(paths["intent_by_label.csv"])
    slot_by_label = read_csv(paths["slot_conll_by_label.csv"])
    slot_by_intent = read_csv(paths["slot_conll_by_intent.csv"])
    scan = scan_predictions(predictions_path)

    schema = "macro/micro schema" if (
        "intent_macro_f1" in overall
        or "macro_f1" in intent_summary
        or "micro_f1" in slot_summary
        or paths["slot_conll_by_intent.csv"].exists()
    ) else "legacy saved schema"

    row_count = scan["rows"] or to_int(overall.get("rows")) or to_int(intent_summary.get("rows"))
    intent_accuracy = pick_float(overall, ["intent_accuracy"]) or pick_float(intent_summary, ["intent_accuracy"])
    intent_macro_f1 = pick_float(overall, ["intent_macro_f1"]) or pick_float(intent_summary, ["macro_f1"])
    saved_slot_f1 = pick_float(overall, ["slot_conll_f1"]) or pick_float(slot_summary, ["f1"])
    slot_micro_f1 = pick_float(overall, ["slot_conll_micro_f1"]) or pick_float(slot_summary, ["micro_f1"])

    return {
        "dataset": dataset,
        "label": label,
        "role": role,
        "schema": schema,
        "root": root,
        "predictions_path": predictions_path,
        "paths": paths,
        "artifacts": {
            "aggregate_predictions": file_info(predictions_path),
            **{name: file_info(path) for name, path in paths.items()},
        },
        "overall": overall,
        "intent_summary": intent_summary,
        "slot_summary": slot_summary,
        "intent_by_label": intent_by_label,
        "slot_by_label": slot_by_label,
        "slot_by_intent": slot_by_intent,
        "row_count": row_count,
        "unknown_predictions": scan["unknown_predictions"],
        "intent_accuracy": intent_accuracy,
        "intent_macro_f1": intent_macro_f1,
        "saved_slot_f1": saved_slot_f1,
        "slot_micro_f1": slot_micro_f1,
    }


def load_subsampling_dataset(dataset: str, label: str, role: str) -> Dict[str, Any]:
    root = EVALUATION_ROOT / dataset / "metrics_subsampling" / SUBSAMPLING_METHOD / SUBSAMPLING_CONFIG
    paths = {name: root / name for name in SUBSAMPLING_FILES}
    summary = read_csv(paths["summary.csv"])
    coverage = read_csv(paths["coverage.csv"])
    rounds = read_csv(paths["rounds.csv"], limit=24)
    intent_by_label_rounds = read_csv(paths["intent_by_label_rounds.csv"])
    slot_by_intent_rounds = read_csv(paths["slot_by_intent_rounds.csv"])
    slot_by_label_rounds = read_csv(paths["slot_by_label_rounds.csv"])

    metric_rows: Dict[tuple[int, str], Mapping[str, str]] = {}
    full_anchor_by_n: Dict[int, bool] = {}
    for row in summary:
        n = to_int(row.get("n_per_intent"))
        metric = row.get("metric", "")
        if n is None or not metric:
            continue
        metric_rows[(n, metric)] = row
        full_anchor_by_n[n] = to_bool(row.get("full_support_anchor"))

    n_values = sorted({n for n, _metric in metric_rows})
    stochastic_n_values = [n for n in n_values if not full_anchor_by_n.get(n, False)]
    primary_n = max(stochastic_n_values) if stochastic_n_values else (max(n_values) if n_values else None)
    full_anchor_n = max((n for n in n_values if full_anchor_by_n.get(n, False)), default=None)

    return {
        "dataset": dataset,
        "label": label,
        "role": role,
        "root": root,
        "paths": paths,
        "artifacts": {name: file_info(path) for name, path in paths.items()},
        "summary": summary,
        "coverage": coverage,
        "rounds": rounds,
        "intent_by_label_rounds": intent_by_label_rounds,
        "slot_by_intent_rounds": slot_by_intent_rounds,
        "slot_by_label_rounds": slot_by_label_rounds,
        "metric_rows": metric_rows,
        "n_values": n_values,
        "primary_n": primary_n,
        "full_anchor_n": full_anchor_n,
        "full_anchor_by_n": full_anchor_by_n,
    }


def load_dataset_story(dataset: str, label: str, role: str) -> Dict[str, Any]:
    return {
        "dataset": dataset,
        "label": label,
        "role": role,
        "normal": load_normal_dataset(dataset, label, role),
        "subsampling": load_subsampling_dataset(dataset, label, role),
    }


def subsampling_metric_row(subsampling: Mapping[str, Any], metric: str, n: int | None = None) -> Mapping[str, str] | None:
    n = subsampling.get("primary_n") if n is None else n
    if n is None:
        return None
    return subsampling["metric_rows"].get((int(n), metric))


def subsampling_metric_mean(subsampling: Mapping[str, Any], metric: str, n: int | None = None) -> float | None:
    row = subsampling_metric_row(subsampling, metric, n=n)
    return to_float(row.get("mean")) if row else None


def build_metric_card(label: str, value: float | None, sublabel: str = "") -> str:
    return f'''
    <div class="metric-card">
        <span>{esc(label)}</span>
        <b>{format_metric(value)}</b>
        <small>{esc(sublabel)}</small>
        {metric_bar(value)}
    </div>
    '''


def normal_cards(normal: Mapping[str, Any]) -> str:
    return f'''
    <div class="card-block">
        <div class="block-title">
            <h3>Normal evaluation</h3>
            <span class="schema {slug(normal['schema'])}">{esc(normal['schema'])}</span>
        </div>
        <div class="mini-grid">
            {build_metric_card("Intent Accuracy", normal["intent_accuracy"], f"rows={format_int(normal['row_count'])}")}
            {build_metric_card("Intent Macro-F1", normal["intent_macro_f1"], "available only in macro/micro schema")}
            {build_metric_card("Saved Slot F1", normal["saved_slot_f1"], "schema-specific saved value")}
            {build_metric_card("Slot Micro-F1", normal["slot_micro_f1"], "available only when saved")}
        </div>
    </div>
    '''


def subsampling_cards(subsampling: Mapping[str, Any]) -> str:
    primary_n = subsampling["primary_n"]
    intent_row = subsampling_metric_row(subsampling, "intent_macro_f1")
    slot_row = subsampling_metric_row(subsampling, "slot_macro_f1_by_intent")
    micro_row = subsampling_metric_row(subsampling, "slot_micro_f1")
    accuracy_row = subsampling_metric_row(subsampling, "intent_accuracy")
    rows_per_round = to_int(intent_row.get("rows_per_round")) if intent_row else None
    rounds = to_int(intent_row.get("rounds")) if intent_row else None
    return f'''
    <div class="card-block">
        <div class="block-title">
            <h3>Stratified subsampling</h3>
            <span class="schema">{esc(SUBSAMPLING_CONFIG)} / primary n={format_int(primary_n)}</span>
        </div>
        <div class="mini-grid">
            {build_metric_card("Intent Macro-F1", subsampling_metric_mean(subsampling, "intent_macro_f1"), metric_interval(intent_row))}
            {build_metric_card("Intent Accuracy", subsampling_metric_mean(subsampling, "intent_accuracy"), metric_interval(accuracy_row))}
            {build_metric_card("Slot Macro-F1 by Intent", subsampling_metric_mean(subsampling, "slot_macro_f1_by_intent"), metric_interval(slot_row))}
            {build_metric_card("Slot Micro-F1", subsampling_metric_mean(subsampling, "slot_micro_f1"), metric_interval(micro_row))}
        </div>
        <p class="muted small">Rows/round: {format_int(rows_per_round)}. Rounds: {format_int(rounds)}. Intervals are empirical 2.5%-97.5% intervals across rounds.</p>
    </div>
    '''


def normal_vs_subsampling_table(story: Mapping[str, Any]) -> str:
    normal = story["normal"]
    subsampling = story["subsampling"]
    rows = [
        {
            "metric": "Intent Accuracy",
            "saved_normal_metrics": format_metric(normal["intent_accuracy"]),
            "subsampling_primary_n": metric_interval(subsampling_metric_row(subsampling, "intent_accuracy")),
            "interpretation": "Accuracy is useful but can be misleading in imbalanced datasets.",
        },
        {
            "metric": "Intent Macro-F1",
            "saved_normal_metrics": format_metric(normal["intent_macro_f1"]) if normal["intent_macro_f1"] is not None else "not saved in legacy schema",
            "subsampling_primary_n": metric_interval(subsampling_metric_row(subsampling, "intent_macro_f1")),
            "interpretation": "Main intent metric for balanced class-level interpretation.",
        },
        {
            "metric": "Saved Normal Slot F1 vs Slot Macro-F1 by Intent",
            "saved_normal_metrics": format_metric(normal["saved_slot_f1"]),
            "subsampling_primary_n": metric_interval(subsampling_metric_row(subsampling, "slot_macro_f1_by_intent")),
            "interpretation": "Normal value is schema-specific; subsampling value is macro by intent.",
        },
        {
            "metric": "Slot Micro-F1",
            "saved_normal_metrics": format_metric(normal["slot_micro_f1"]) if normal["slot_micro_f1"] is not None else "not saved in legacy schema",
            "subsampling_primary_n": metric_interval(subsampling_metric_row(subsampling, "slot_micro_f1")),
            "interpretation": "Micro score is dominated by frequent slot chunks.",
        },
    ]
    return table_html(rows, compact=True, preformatted=True)


def summary_by_n_table(subsampling: Mapping[str, Any]) -> str:
    rows = []
    for n in subsampling["n_values"]:
        intent = subsampling_metric_row(subsampling, "intent_macro_f1", n=n)
        accuracy = subsampling_metric_row(subsampling, "intent_accuracy", n=n)
        slot_macro = subsampling_metric_row(subsampling, "slot_macro_f1_by_intent", n=n)
        slot_micro = subsampling_metric_row(subsampling, "slot_micro_f1", n=n)
        base = intent or slot_macro or {}
        rows.append(
            {
                "n_per_intent": format_int(n),
                "rows_per_round": format_int(to_int(base.get("rows_per_round")) if base else None),
                "rounds": format_int(to_int(base.get("rounds")) if base else None),
                "full_support_anchor": str(base.get("full_support_anchor", "")),
                "intent_accuracy": metric_interval(accuracy),
                "intent_macro_f1": metric_interval(intent),
                "slot_macro_f1_by_intent": metric_interval(slot_macro),
                "slot_micro_f1": metric_interval(slot_micro),
            }
        )
    return table_html(rows, compact=True, preformatted=True)


def aggregate_round_labels(
    rows: Sequence[Mapping[str, str]],
    *,
    n: int | None,
    label_column: str,
    support_column: str,
    limit: int = 8,
    descending: bool = False,
) -> List[Dict[str, Any]]:
    if n is None:
        return []
    groups: Dict[str, List[Mapping[str, str]]] = {}
    for row in rows:
        if to_int(row.get("n_per_intent")) != n:
            continue
        label = row.get(label_column, "")
        if label:
            groups.setdefault(label, []).append(row)

    result = []
    for label, group in groups.items():
        def avg(column: str) -> float | None:
            values = [to_float(item.get(column)) for item in group]
            values = [value for value in values if value is not None]
            return sum(values) / len(values) if values else None

        support = avg(support_column)
        if support is None or support <= 0:
            continue
        precision = avg("precision")
        result.append(
            {
                label_column: label,
                "mean_support": support,
                "mean_precision": precision,
                "mean_recall": avg("recall"),
                "mean_f1": avg("f1"),
                "mean_accuracy": avg("accuracy"),
                "mean_fdr": (1.0 - precision) if precision is not None else None,
                "rounds": len(group),
            }
        )
    if descending:
        result.sort(
            key=lambda item: (
                item.get("mean_f1") is None,
                -(item.get("mean_f1") if item.get("mean_f1") is not None else -1.0),
                -float(item.get("mean_support") or 0),
                item[label_column],
            )
        )
    else:
        result.sort(
            key=lambda item: (
                item.get("mean_f1") is None,
                item.get("mean_f1") if item.get("mean_f1") is not None else 2.0,
                -float(item.get("mean_support") or 0),
                item[label_column],
            )
        )
    return result[:limit]


def hardest_normal_intents(normal: Mapping[str, Any], limit: int = 8) -> List[Dict[str, Any]]:
    rows = []
    for row in normal["intent_by_label"]:
        support = to_float(row.get("support"))
        if support is None or support <= 0:
            continue
        precision = to_float(row.get("precision"))
        score = to_float(row.get("f1"))
        if score is None:
            score = to_float(row.get("accuracy"))
        rows.append(
            {
                "semantic_intent": row.get("semantic_intent", ""),
                "support": support,
                "precision": precision,
                "recall": to_float(row.get("recall")),
                "f1": to_float(row.get("f1")),
                "accuracy": to_float(row.get("accuracy")),
                "fdr": (1.0 - precision) if precision is not None else None,
            }
        )
    rows.sort(key=lambda item: (item.get("f1") is None and item.get("accuracy") is None, item.get("f1") if item.get("f1") is not None else item.get("accuracy") or 2.0))
    return rows[:limit]


def normal_slots_by_f1(normal: Mapping[str, Any], limit: int = 8, *, descending: bool = False) -> List[Dict[str, Any]]:
    rows = []
    for row in normal["slot_by_label"]:
        support = to_float(row.get("support_gold"))
        if support is None or support <= 0:
            continue
        rows.append(
            {
                "semantic_slot": row.get("semantic_slot", ""),
                "support_gold": support,
                "precision": to_float(row.get("precision")),
                "recall": to_float(row.get("recall")),
                "f1": to_float(row.get("f1")),
            }
        )
    if descending:
        rows.sort(
            key=lambda item: (
                item.get("f1") is None,
                -(item.get("f1") if item.get("f1") is not None else -1.0),
                -float(item.get("support_gold") or 0),
                item.get("semantic_slot", ""),
            )
        )
    else:
        rows.sort(
            key=lambda item: (
                item.get("f1") is None,
                item.get("f1") if item.get("f1") is not None else 2.0,
                -float(item.get("support_gold") or 0),
                item.get("semantic_slot", ""),
            )
        )
    return rows[:limit]


def hardest_normal_slots(normal: Mapping[str, Any], limit: int = 8) -> List[Dict[str, Any]]:
    return normal_slots_by_f1(normal, limit=limit, descending=False)


def best_normal_slots(normal: Mapping[str, Any], limit: int = 8) -> List[Dict[str, Any]]:
    return normal_slots_by_f1(normal, limit=limit, descending=True)


def diagnostics_tables(story: Mapping[str, Any]) -> str:
    normal = story["normal"]
    subsampling = story["subsampling"]
    primary_n = subsampling["primary_n"]
    sampling_intents = aggregate_round_labels(
        subsampling["intent_by_label_rounds"],
        n=primary_n,
        label_column="semantic_intent",
        support_column="support",
    )
    sampling_hardest_slots = aggregate_round_labels(
        subsampling["slot_by_label_rounds"],
        n=primary_n,
        label_column="semantic_slot",
        support_column="support_gold",
    )
    sampling_best_slots = aggregate_round_labels(
        subsampling["slot_by_label_rounds"],
        n=primary_n,
        label_column="semantic_slot",
        support_column="support_gold",
        descending=True,
    )
    return f'''
    <div class="two-cols">
        <div>
            <h3>Hardest intents - saved normal metrics</h3>
            {table_html(hardest_normal_intents(normal), columns=["semantic_intent", "support", "precision", "recall", "f1", "accuracy", "fdr"], compact=True)}
        </div>
        <div>
            <h3>Hardest intents - subsampling @ primary n</h3>
            {table_html(sampling_intents, columns=["semantic_intent", "mean_support", "mean_precision", "mean_recall", "mean_f1", "mean_accuracy", "mean_fdr", "rounds"], compact=True)}
        </div>
    </div>
    <div class="two-cols">
        <div>
            <h3>Slot F1 extremes - saved normal metrics</h3>
            {sectioned_table_html(
                [
                    ("Highest F1 score", best_normal_slots(normal)),
                    ("Lowest F1 score", hardest_normal_slots(normal)),
                ],
                columns=["semantic_slot", "support_gold", "precision", "recall", "f1"],
                compact=True,
            )}
        </div>
        <div>
            <h3>Slot F1 extremes - subsampling @ primary n</h3>
            {sectioned_table_html(
                [
                    ("Highest F1 score", sampling_best_slots),
                    ("Lowest F1 score", sampling_hardest_slots),
                ],
                columns=["semantic_slot", "mean_support", "mean_precision", "mean_recall", "mean_f1", "rounds"],
                compact=True,
            )}
        </div>
    </div>
    '''


def curve_svg(subsampling: Mapping[str, Any]) -> str:
    if not subsampling["n_values"]:
        return '<p class="muted">No subsampling summary available.</p>'
    width, height = 920, 310
    left, right, top, bottom = 58, 28, 24, 62
    plot_w = width - left - right
    plot_h = height - top - bottom
    n_values = subsampling["n_values"]
    min_n, max_n = min(n_values), max(n_values)

    def x_for_n(n: int) -> float:
        if min_n == max_n:
            return left + plot_w / 2
        return left + ((n - min_n) / (max_n - min_n)) * plot_w

    def y_for_value(value: float | None) -> float:
        value = 0.0 if value is None else max(0.0, min(1.0, value))
        return top + (1.0 - value) * plot_h

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Subsampling stability curve">']
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = y_for_value(tick)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="10" y="{y + 4:.1f}" class="axis">{tick:.2f}</text>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" class="axis-line"/>')
    parts.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" class="axis-line"/>')
    for n in n_values:
        x = x_for_n(n)
        parts.append(f'<line x1="{x:.1f}" y1="{height-bottom}" x2="{x:.1f}" y2="{height-bottom+5}" class="axis-line"/>')
        parts.append(f'<text x="{x:.1f}" y="{height-bottom+20}" text-anchor="middle" class="axis">{n}</text>')

    for metric, label, color in CURVE_METRICS:
        points = []
        for n in n_values:
            row = subsampling_metric_row(subsampling, metric, n=n)
            mean = to_float(row.get("mean")) if row else None
            p2_5 = to_float(row.get("p2_5")) if row else None
            p97_5 = to_float(row.get("p97_5")) if row else None
            if mean is None:
                continue
            x = x_for_n(n)
            y = y_for_value(mean)
            points.append(f"{x:.1f},{y:.1f}")
            if p2_5 is not None and p97_5 is not None:
                low_y = y_for_value(p2_5)
                high_y = y_for_value(p97_5)
                parts.append(f'<line x1="{x:.1f}" y1="{high_y:.1f}" x2="{x:.1f}" y2="{low_y:.1f}" stroke="{color}" stroke-width="1.5" opacity="0.45"/>')
                parts.append(f'<line x1="{x-5:.1f}" y1="{high_y:.1f}" x2="{x+5:.1f}" y2="{high_y:.1f}" stroke="{color}" stroke-width="1.5" opacity="0.45"/>')
                parts.append(f'<line x1="{x-5:.1f}" y1="{low_y:.1f}" x2="{x+5:.1f}" y2="{low_y:.1f}" stroke="{color}" stroke-width="1.5" opacity="0.45"/>')
        if points:
            parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="3"/>')
            for point in points:
                x_text, y_text = point.split(",")
                parts.append(f'<circle cx="{x_text}" cy="{y_text}" r="4" fill="{color}"/>')

    legend_x = left
    legend_y = height - 20
    for _metric, label, color in CURVE_METRICS:
        parts.append(f'<rect x="{legend_x}" y="{legend_y - 10}" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 18}" y="{legend_y}" class="legend">{esc(label)}</text>')
        legend_x += 250
    parts.append("</svg>")
    return "".join(parts)


def field_stability_note(story: Mapping[str, Any]) -> str:
    if story["dataset"] != "field_operations":
        return ""
    subsampling = story["subsampling"]
    primary_n = subsampling["primary_n"]
    full_n = subsampling["full_anchor_n"]
    primary_intent = subsampling_metric_mean(subsampling, "intent_macro_f1", n=primary_n)
    full_intent = subsampling_metric_mean(subsampling, "intent_macro_f1", n=full_n)
    primary_slot = subsampling_metric_mean(subsampling, "slot_macro_f1_by_intent", n=primary_n)
    full_slot = subsampling_metric_mean(subsampling, "slot_macro_f1_by_intent", n=full_n)
    return f'''
    <p class="note"><b>Primary stability reading:</b> Field Operations uses n={format_int(primary_n)} as the largest stochastic sample and n={format_int(full_n)} as the full-dataset anchor. Intent Macro-F1 is {format_metric(primary_intent)} at n={format_int(primary_n)} and {format_metric(full_intent)} at n={format_int(full_n)}; Slot Macro-F1 by Intent is {format_metric(primary_slot)} and {format_metric(full_slot)}, respectively. The mean is stable while the empirical interval narrows as n increases.</p>
    '''


def artifact_tables(story: Mapping[str, Any]) -> str:
    rows = []
    for family, data in [("normal", story["normal"]), ("subsampling", story["subsampling"])]:
        for name, info in data["artifacts"].items():
            rows.append(
                {
                    "family": family,
                    "artifact": name,
                    "status": "found" if info["exists"] else "missing",
                    "file": relpath(info["path"]),
                    "size": format_bytes(info["bytes"]),
                    "csv_rows": format_int(info["rows"]),
                    "modified": info["modified"],
                }
            )
    return table_html(rows, compact=True, preformatted=True)


def dataset_section(story: Mapping[str, Any]) -> str:
    normal = story["normal"]
    subsampling = story["subsampling"]
    schema_warning = ""
    if normal["schema"] == "legacy saved schema":
        schema_warning = '<p class="warning"><b>Legacy schema:</b> saved normal metrics for this dataset do not include all macro/micro fields. The subsampling analysis uses the newer macro-by-intent reporting, so direct slot-F1 comparisons should be interpreted carefully.</p>'
    return f'''
    <section id="{esc(story['dataset'])}" class="dataset-section">
        <div class="section-title">
            <div>
                <span class="eyebrow">{esc(story['role'])}</span>
                <h2>{esc(story['label'])}</h2>
            </div>
            <span class="schema">{esc(SUBSAMPLING_METHOD)} / {esc(SUBSAMPLING_CONFIG)}</span>
        </div>
        {schema_warning}
        {field_stability_note(story)}
        {normal_cards(normal)}
        {subsampling_cards(subsampling)}
        <div class="panel-inner">
            <h3>Normal vs subsampling</h3>
            {normal_vs_subsampling_table(story)}
        </div>
        <div class="panel-inner">
            <h3>Subsampling stability curve</h3>
            <p class="muted small">Error bars show empirical 2.5%-97.5% interval across saved subsampling rounds.</p>
            {curve_svg(subsampling)}
        </div>
        <div class="panel-inner">
            <h3>Subsampling summary by n</h3>
            {summary_by_n_table(subsampling)}
        </div>
        <div class="panel-inner">
            <h3>Diagnostics</h3>
            {diagnostics_tables(story)}
        </div>
        <details>
            <summary>Artifact Integrity</summary>
            {artifact_tables(story)}
        </details>
    </section>
    '''


def final_overview_table(stories: Sequence[Mapping[str, Any]]) -> str:
    rows = []
    for story in stories:
        subsampling = story["subsampling"]
        primary = subsampling_metric_row(subsampling, "intent_macro_f1")
        rows.append(
            {
                "dataset": story["label"],
                "primary_n": format_int(subsampling["primary_n"]),
                "rows_per_round": format_int(to_int(primary.get("rows_per_round")) if primary else None),
                "rounds": format_int(to_int(primary.get("rounds")) if primary else None),
                "intent_macro_f1": metric_interval(subsampling_metric_row(subsampling, "intent_macro_f1")),
                "intent_accuracy": metric_interval(subsampling_metric_row(subsampling, "intent_accuracy")),
                "slot_macro_f1_by_intent": metric_interval(subsampling_metric_row(subsampling, "slot_macro_f1_by_intent")),
                "slot_micro_f1": metric_interval(subsampling_metric_row(subsampling, "slot_micro_f1")),
            }
        )
    return table_html(rows, compact=True, preformatted=True)


def normal_overview_table(stories: Sequence[Mapping[str, Any]]) -> str:
    rows = []
    for story in stories:
        normal = story["normal"]
        rows.append(
            {
                "dataset": story["label"],
                "schema": normal["schema"],
                "rows": format_int(normal["row_count"]),
                "intent_accuracy": format_metric(normal["intent_accuracy"]),
                "intent_macro_f1": format_metric(normal["intent_macro_f1"]) if normal["intent_macro_f1"] is not None else "not saved",
                "saved_slot_f1": format_metric(normal["saved_slot_f1"]),
                "slot_micro_f1": format_metric(normal["slot_micro_f1"]) if normal["slot_micro_f1"] is not None else "not saved",
                "unknown_predictions": format_int(normal["unknown_predictions"]),
            }
        )
    return table_html(rows, compact=True, preformatted=True)


def final_artifact_overview(stories: Sequence[Mapping[str, Any]]) -> str:
    rows = []
    for story in stories:
        normal_found = sum(1 for info in story["normal"]["artifacts"].values() if info["exists"])
        subsampling_found = sum(1 for info in story["subsampling"]["artifacts"].values() if info["exists"])
        rows.append(
            {
                "dataset": story["label"],
                "normal_artifacts": f"{normal_found}/{len(story['normal']['artifacts'])}",
                "subsampling_artifacts": f"{subsampling_found}/{len(story['subsampling']['artifacts'])}",
                "normal_schema": story["normal"]["schema"],
                "subsampling_dir": relpath(story["subsampling"]["root"]),
            }
        )
    return table_html(rows, compact=True, preformatted=True)


def page_css() -> str:
    return """
    :root {
        --bg: #f4f6f8;
        --panel: #ffffff;
        --ink: #172033;
        --muted: #667085;
        --line: #d9e0e8;
        --blue: #1565c0;
        --teal: #00897b;
        --orange: #ef6c00;
        --green: #16794c;
        --green-bg: #e7f6ee;
        --warn-bg: #fff5df;
        --warn: #8a5200;
    }
    * { box-sizing: border-box; }
    body {
        margin: 0;
        background: var(--bg);
        color: var(--ink);
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        line-height: 1.45;
    }
    header {
        background: var(--panel);
        border-bottom: 1px solid var(--line);
        padding: 30px 36px 18px;
    }
    h1, h2, h3 { margin: 0 0 8px; line-height: 1.15; }
    h1 { font-size: 32px; }
    h2 { font-size: 25px; }
    h3 { font-size: 17px; margin-top: 12px; }
    code {
        background: #eef2f6;
        border: 1px solid #dfe5ec;
        border-radius: 5px;
        padding: 1px 5px;
    }
    .muted { color: var(--muted); }
    .small { font-size: 13px; }
    nav {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 18px;
    }
    nav a {
        color: var(--ink);
        text-decoration: none;
        border: 1px solid var(--line);
        background: #f9fafb;
        padding: 7px 11px;
        border-radius: 6px;
        font-size: 14px;
    }
    main { padding: 24px 36px 44px; }
    .dataset-section, .overview-section {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        padding: 22px;
        margin: 0 0 24px;
    }
    .section-title, .block-title {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        flex-wrap: wrap;
    }
    .eyebrow {
        display: inline-block;
        color: var(--teal);
        font-size: 12px;
        font-weight: 800;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .note, .warning {
        border: 1px solid var(--line);
        border-radius: 6px;
        padding: 12px 14px;
        margin: 12px 0;
    }
    .note {
        border-left: 4px solid var(--teal);
        background: #eef7f5;
        color: #123b35;
    }
    .warning {
        border-left: 4px solid var(--orange);
        background: var(--warn-bg);
        color: #4c3a18;
    }
    .card-block, .panel-inner {
        border-top: 1px solid var(--line);
        margin-top: 16px;
        padding-top: 16px;
    }
    .mini-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin: 14px 0;
    }
    .metric-card {
        border: 1px solid var(--line);
        border-radius: 7px;
        padding: 12px;
        background: #fbfcfd;
    }
    .metric-card b {
        display: block;
        font-size: 24px;
        margin: 4px 0 2px;
    }
    .metric-card span, .metric-card small {
        display: block;
        color: var(--muted);
        font-size: 12px;
    }
    .schema {
        display: inline-flex;
        align-items: center;
        white-space: nowrap;
        border-radius: 999px;
        padding: 4px 9px;
        font-size: 12px;
        font-weight: 700;
        background: #edf2f7;
        color: #425466;
    }
    .schema.macro-micro-schema { background: var(--green-bg); color: var(--green); }
    .schema.legacy-saved-schema { background: var(--warn-bg); color: var(--warn); }
    .bar {
        position: relative;
        height: 20px;
        background: #edf1f5;
        border-radius: 5px;
        overflow: hidden;
        min-width: 120px;
        margin-top: 8px;
    }
    .bar-fill { height: 100%; background: linear-gradient(90deg, var(--teal), var(--blue)); }
    .bar span {
        position: absolute;
        inset: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        font-weight: 700;
        color: #132235;
    }
    .two-cols {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 16px;
        align-items: start;
    }
    details {
        border-top: 1px solid var(--line);
        padding-top: 12px;
        margin-top: 14px;
    }
    summary {
        cursor: pointer;
        font-weight: 750;
        margin-bottom: 12px;
    }
    .table-wrap { width: 100%; overflow-x: auto; }
    table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }
    table.compact { font-size: 12px; }
    th {
        text-align: left;
        padding: 8px 9px;
        border-bottom: 1px solid var(--line);
        color: #334155;
        background: #f8fafc;
        position: sticky;
        top: 0;
    }
    .table-section th {
        position: static;
        text-align: center;
        font-weight: 850;
        color: #1f2933;
        background: #eef3f7;
        border-top: 2px solid #cbd5df;
        border-bottom: 2px solid #cbd5df;
        letter-spacing: 0;
    }
    td {
        padding: 7px 9px;
        border-bottom: 1px solid #eef1f5;
        vertical-align: top;
    }
    tr:hover td { background: #fbfdff; }
    svg {
        width: 100%;
        height: auto;
        background: #fbfcfd;
        border: 1px solid var(--line);
        border-radius: 8px;
        margin: 8px 0 14px;
    }
    .grid { stroke: #dfe5ec; stroke-width: 1; }
    .axis-line { stroke: #9aa6b2; stroke-width: 1.2; }
    .axis, .legend { fill: #52606d; font-size: 12px; }
    footer {
        color: var(--muted);
        font-size: 12px;
        padding: 8px 36px 28px;
    }
    @media (max-width: 1100px) {
        .mini-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .two-cols { grid-template-columns: 1fr; }
    }
    @media (max-width: 700px) {
        header, main, footer { padding-left: 18px; padding-right: 18px; }
        .mini-grid { grid-template-columns: 1fr; }
    }
    """


def build_html(stories: Sequence[Mapping[str, Any]]) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dataset_sections = "\n".join(dataset_section(story) for story in stories)
    return f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Unified Batch Eval Dashboard</title>
    <style>{page_css()}</style>
</head>
<body>
    <header>
        <h1>Unified Batch Eval Dashboard</h1>
        <p class="muted">Vertical dataset-first report combining saved normal evaluation metrics and stratified subsampling metrics. Generated at {esc(generated_at)}.</p>
        <nav>
            {''.join(f'<a href="#{esc(story["dataset"])}">{esc(story["label"])}</a>' for story in stories)}
            <a href="#overall">Overall Panorama</a>
            <a href="#artifacts">Artifact Integrity</a>
        </nav>
    </header>
    <main>
        <p class="warning"><b>Methodological note:</b> subsampling uses saved predictions and estimates sensitivity to the evaluation sample, not training randomness. The reported intervals are empirical 2.5%-97.5% intervals across rounds.</p>
        {dataset_sections}
        <section id="overall" class="overview-section">
            <h2>Overall Panorama</h2>
            <p class="muted">Main table for paper-style reporting. Subsampling values use each dataset's primary n.</p>
            {final_overview_table(stories)}
            <h3>Saved normal metrics overview</h3>
            {normal_overview_table(stories)}
            <h3>Interpretation notes</h3>
            <p>The subsampling analysis is strongest for Field Operations because it is the target domain and is balanced by design. Public datasets are complementary stability checks. Stable intervals do not imply uniformly high performance: a metric can be stable because errors are systematic.</p>
            <p>For intent recognition, Intent Macro-F1 under stratified subsampling is the main class-balanced metric. For slot filling, Slot Macro-F1 by Intent is the main class-balanced metric, while Slot Micro-F1 remains useful as a frequency-weighted diagnostic.</p>
        </section>
        <section id="artifacts" class="overview-section">
            <h2>Artifact Integrity</h2>
            {final_artifact_overview(stories)}
        </section>
    </main>
    <footer>
        Generated by src/data_tools/build_unified_eval_dashboard.py. This page reads saved CSV artifacts only.
    </footer>
</body>
</html>
"""


def main() -> None:
    stories = [load_dataset_story(dataset, label, role) for dataset, label, role in DATASETS]
    html_text = build_html(stories)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(html_text, encoding="utf-8")
    print(f"Wrote {relpath(OUTPUT_PATH)}")


if __name__ == "__main__":
    main()
