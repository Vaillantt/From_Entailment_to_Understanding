from __future__ import annotations

import csv
import html
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ROOT = REPO_ROOT / "evaluation"
OUTPUT_PATH = EVALUATION_ROOT / "reports" / "batch_eval_dashboard.html"

DATASETS = [
    ("snips", "SNIPS"),
    ("multiwoz", "MultiWOZ"),
    ("atis", "ATIS"),
    ("field_operations", "Field Operations"),
]

AGGREGATE_PREDICTION_FILES = {
    dataset: f"{dataset}_predictions.csv" for dataset, _label in DATASETS
}

METRIC_FILES = [
    "overall_summary.csv",
    "intent_summary.csv",
    "intent_by_label.csv",
    "intent_confusion.csv",
    "slot_conll_summary.csv",
    "slot_conll_by_label.csv",
    "slot_conll_by_intent.csv",
]

STRATIFIED_DIR = EVALUATION_ROOT / "atis" / "metrics_stratified" / "n50_k30_seed42"
STRATIFIED_FILES = [
    "summary.csv",
    "rounds.csv",
    "coverage.csv",
    "intent_by_label_rounds.csv",
    "slot_by_intent_rounds.csv",
]


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


def csv_columns(path: Path) -> List[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader, [])


def file_info(path: Path) -> Dict[str, Any]:
    exists = path.exists()
    stat = path.stat() if exists else None
    return {
        "path": path,
        "exists": exists,
        "bytes": stat.st_size if stat else None,
        "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S") if stat else "",
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
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def pick_float(row: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        value = to_float(row.get(key))
        if value is not None:
            return value
    return None


def pick_int(row: Mapping[str, Any], keys: Sequence[str]) -> int | None:
    for key in keys:
        value = to_int(row.get(key))
        if value is not None:
            return value
    return None


def format_int(value: int | None) -> str:
    if value is None:
        return "-"
    return f"{value:,}".replace(",", ".")


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


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def css_class_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in value.lower()).strip("-")


def scan_prediction_file(path: Path) -> Dict[str, Any]:
    result = {
        "row_count": 0,
        "semantic_counts": Counter(),
        "pred_counts": Counter(),
        "source_counts": Counter(),
        "unknown_predictions": 0,
        "slot_eval_mismatches": 0,
        "score_values": [],
    }
    if not path.exists():
        return result

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            result["row_count"] += 1
            semantic_intent = row.get("semantic_intent", "")
            pred_intent = row.get("pred_intent", "")
            source_file = row.get("source_file", "")
            slot_eval_intent = row.get("slot_eval_intent", "")
            if semantic_intent:
                result["semantic_counts"][semantic_intent] += 1
            if pred_intent:
                result["pred_counts"][pred_intent] += 1
            if source_file:
                result["source_counts"][source_file] += 1
            if pred_intent == "UNK_INTENT":
                result["unknown_predictions"] += 1
            if slot_eval_intent and semantic_intent and slot_eval_intent != semantic_intent:
                result["slot_eval_mismatches"] += 1
            score = to_float(row.get("pred_intent_score"))
            if score is not None:
                result["score_values"].append(score)

    scores = result["score_values"]
    result["score_min"] = min(scores) if scores else None
    result["score_max"] = max(scores) if scores else None
    result["score_mean"] = (sum(scores) / len(scores)) if scores else None
    del result["score_values"]
    return result


def load_dataset(dataset: str, label: str) -> Dict[str, Any]:
    root = EVALUATION_ROOT / dataset
    results_dir = root / "results_csv"
    metrics_dir = root / "metrics"
    predictions_path = results_dir / AGGREGATE_PREDICTION_FILES[dataset]

    metric_paths = {name: metrics_dir / name for name in METRIC_FILES}
    overall = read_first_row(metric_paths["overall_summary.csv"])
    intent_summary = read_first_row(metric_paths["intent_summary.csv"])
    slot_summary = read_first_row(metric_paths["slot_conll_summary.csv"])
    schema = "macro/micro schema" if (
        "intent_macro_f1" in overall
        or "macro_f1" in intent_summary
        or "micro_f1" in slot_summary
        or metric_paths["slot_conll_by_intent.csv"].exists()
    ) else "legacy saved schema"

    scan = scan_prediction_file(predictions_path)
    row_count = scan["row_count"] or pick_int(overall, ["rows"]) or pick_int(intent_summary, ["rows"])
    intent_accuracy = pick_float(overall, ["intent_accuracy"]) or pick_float(intent_summary, ["intent_accuracy"])
    intent_macro_f1 = pick_float(overall, ["intent_macro_f1"]) or pick_float(intent_summary, ["macro_f1"])
    saved_slot_f1 = pick_float(overall, ["slot_conll_f1"]) or pick_float(slot_summary, ["f1"])
    slot_micro_f1 = pick_float(overall, ["slot_conll_micro_f1"]) or pick_float(slot_summary, ["micro_f1"])

    result_files = sorted(results_dir.glob("*.csv")) if results_dir.exists() else []
    metric_files = sorted(metrics_dir.glob("*.csv")) if metrics_dir.exists() else []

    return {
        "dataset": dataset,
        "label": label,
        "root": root,
        "results_dir": results_dir,
        "metrics_dir": metrics_dir,
        "predictions_path": predictions_path,
        "metric_paths": metric_paths,
        "schema": schema,
        "scan": scan,
        "row_count": row_count,
        "intent_accuracy": intent_accuracy,
        "intent_macro_f1": intent_macro_f1,
        "saved_slot_f1": saved_slot_f1,
        "slot_micro_f1": slot_micro_f1,
        "overall": overall,
        "intent_summary": intent_summary,
        "slot_summary": slot_summary,
        "intent_by_label": read_csv(metric_paths["intent_by_label.csv"]),
        "intent_confusion": read_csv(metric_paths["intent_confusion.csv"]),
        "slot_by_label": read_csv(metric_paths["slot_conll_by_label.csv"]),
        "slot_by_intent": read_csv(metric_paths["slot_conll_by_intent.csv"]),
        "result_files": result_files,
        "metric_files": metric_files,
        "artifacts": {
            "aggregate_predictions": file_info(predictions_path),
            **{name: file_info(path) for name, path in metric_paths.items()},
        },
    }


def load_stratified() -> Dict[str, Any]:
    paths = {name: STRATIFIED_DIR / name for name in STRATIFIED_FILES}
    return {
        "dir": STRATIFIED_DIR,
        "paths": paths,
        "summary": read_csv(paths["summary.csv"]),
        "rounds": read_csv(paths["rounds.csv"]),
        "coverage": read_csv(paths["coverage.csv"]),
        "intent_by_label_rounds": read_csv(paths["intent_by_label_rounds.csv"], limit=80),
        "slot_by_intent_rounds": read_csv(paths["slot_by_intent_rounds.csv"], limit=80),
        "artifacts": {name: file_info(path) for name, path in paths.items()},
    }


def table_html(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str] | None = None,
    numeric_as_percent: Iterable[str] = (),
    compact: bool = False,
    preformatted: bool = False,
) -> str:
    if not rows:
        return '<p class="muted">No rows available.</p>'
    columns = list(columns or rows[0].keys())
    percent_cols = set(numeric_as_percent)
    parts = ['<div class="table-wrap">', '<table class="compact">' if compact else "<table>"]
    parts.append("<thead><tr>")
    for column in columns:
        parts.append(f"<th>{esc(column)}</th>")
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        for column in columns:
            value = row.get(column, "")
            if preformatted:
                text = str(value)
            else:
                numeric = to_float(value)
                column_key = column.lower()
                count_columns = {
                    "round",
                    "rows",
                    "n",
                    "k",
                    "seed",
                    "support",
                    "support_gold",
                    "support_pred",
                    "predicted",
                    "correct",
                    "errors",
                    "intent_count",
                    "intent_correct",
                    "intent_errors",
                    "tp",
                    "fp",
                    "fn",
                    "tn",
                    "micro_tp",
                    "micro_fp",
                    "micro_fn",
                    "effective_per_round",
                    "total_sampled_across_rounds",
                }
                if column in percent_cols and numeric is not None:
                    text = format_metric(numeric)
                elif numeric is not None and column_key in count_columns:
                    text = format_int(int(numeric))
                elif numeric is not None and any(word in column_key for word in ("accuracy", "precision", "recall", "f1")):
                    text = format_metric(numeric)
                elif numeric is not None:
                    text = format_decimal(numeric)
                else:
                    text = str(value)
            parts.append(f"<td>{esc(text)}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def metric_bar(value: float | None, label: str = "") -> str:
    if value is None:
        return '<div class="bar empty"><span>n/a</span></div>'
    width = max(0.0, min(100.0, value * 100.0))
    return (
        '<div class="bar">'
        f'<div class="bar-fill" style="width: {width:.2f}%"></div>'
        f'<span>{esc(label or format_metric(value))}</span>'
        "</div>"
    )


def count_table(counter: Counter, total: int, title: str, limit: int = 12) -> str:
    rows = []
    for label, count in counter.most_common(limit):
        rows.append(
            {
                title: label,
                "count": format_int(count),
                "share": format_metric(count / total) if total else "-",
                "bar": metric_bar(count / total if total else None, ""),
            }
        )
    if not rows:
        return '<p class="muted">No distribution available.</p>'
    parts = ['<div class="table-wrap"><table class="compact distribution"><thead><tr>']
    for column in [title, "count", "share", "bar"]:
        parts.append(f"<th>{esc(column)}</th>")
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        parts.append(f"<td>{esc(row[title])}</td>")
        parts.append(f"<td>{esc(row['count'])}</td>")
        parts.append(f"<td>{esc(row['share'])}</td>")
        parts.append(f"<td>{row['bar']}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def artifact_table(files: Sequence[Path]) -> str:
    rows = []
    for path in files:
        info = file_info(path)
        rows.append(
            {
                "file": relpath(path),
                "size": format_bytes(info["bytes"]),
                "modified": info["modified"],
            }
        )
    return table_html(rows, compact=True)


def expected_artifacts_table(dataset_data: Mapping[str, Any]) -> str:
    rows = []
    for name, info in dataset_data["artifacts"].items():
        rows.append(
            {
                "artifact": name,
                "status": "found" if info["exists"] else "missing",
                "file": relpath(info["path"]),
                "size": format_bytes(info["bytes"]),
                "modified": info["modified"],
            }
        )
    return table_html(rows, compact=True)


def line_chart_svg(rounds: Sequence[Mapping[str, str]]) -> str:
    if not rounds:
        return '<p class="muted">No stratified rounds available.</p>'
    series = {
        "intent_macro_f1": "#1565c0",
        "slot_macro_f1_by_intent": "#00897b",
        "slot_micro_f1": "#ef6c00",
    }
    width, height = 900, 260
    left, right, top, bottom = 54, 24, 22, 38
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_round = max(to_int(row.get("round")) or 0 for row in rounds) or 1

    def point(row: Mapping[str, str], key: str) -> str:
        r = to_int(row.get("round")) or 1
        value = to_float(row.get(key)) or 0.0
        x = left + ((r - 1) / max(max_round - 1, 1)) * plot_w
        y = top + (1.0 - max(0.0, min(1.0, value))) * plot_h
        return f"{x:.1f},{y:.1f}"

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="ATIS stratified round metrics">']
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = top + (1.0 - tick) * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="8" y="{y + 4:.1f}" class="axis">{tick:.2f}</text>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" class="axis-line"/>')
    parts.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" class="axis-line"/>')
    for key, color in series.items():
        points = " ".join(point(row, key) for row in rounds)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
    legend_x = left
    for key, color in series.items():
        parts.append(f'<rect x="{legend_x}" y="{height-20}" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 18}" y="{height-10}" class="legend">{esc(key)}</text>')
        legend_x += 220
    parts.append("</svg>")
    return "".join(parts)


def summary_cards(datasets: Sequence[Mapping[str, Any]]) -> str:
    best_intent = max((d for d in datasets if d["intent_accuracy"] is not None), key=lambda d: d["intent_accuracy"], default=None)
    worst_intent = min((d for d in datasets if d["intent_accuracy"] is not None), key=lambda d: d["intent_accuracy"], default=None)
    best_slot = max((d for d in datasets if d["saved_slot_f1"] is not None), key=lambda d: d["saved_slot_f1"], default=None)
    worst_slot = min((d for d in datasets if d["saved_slot_f1"] is not None), key=lambda d: d["saved_slot_f1"], default=None)

    parts = ['<section class="cards">']
    for data in datasets:
        badges = []
        if best_intent and data["dataset"] == best_intent["dataset"]:
            badges.append('<span class="badge good">Best intent acc</span>')
        if worst_intent and data["dataset"] == worst_intent["dataset"]:
            badges.append('<span class="badge warn">Lowest intent acc</span>')
        if best_slot and data["dataset"] == best_slot["dataset"]:
            badges.append('<span class="badge good">Best saved slot F1</span>')
        if worst_slot and data["dataset"] == worst_slot["dataset"]:
            badges.append('<span class="badge warn">Lowest saved slot F1</span>')
        schema_class = "schema-new" if data["schema"] == "macro/micro schema" else "schema-legacy"
        parts.append(
            f'''
            <article class="card {css_class_slug(data['dataset'])}">
                <div class="card-head">
                    <h3>{esc(data['label'])}</h3>
                    <span class="schema {schema_class}">{esc(data['schema'])}</span>
                </div>
                <div class="big">{format_metric(data['intent_accuracy'])}</div>
                <div class="label">Intent accuracy</div>
                {metric_bar(data['intent_accuracy'])}
                <dl>
                    <dt>Rows</dt><dd>{format_int(data['row_count'])}</dd>
                    <dt>Saved Slot F1</dt><dd>{format_metric(data['saved_slot_f1'])}</dd>
                    <dt>Intent Macro-F1</dt><dd>{format_metric(data['intent_macro_f1'])}</dd>
                    <dt>Unknown intent predictions</dt><dd>{format_int(data['scan']['unknown_predictions'])}</dd>
                </dl>
                <div class="badges">{''.join(badges)}</div>
            </article>
            '''
        )
    parts.append("</section>")
    return "".join(parts)


def comparison_table(datasets: Sequence[Mapping[str, Any]]) -> str:
    rows = []
    for data in datasets:
        rows.append(
            {
                "dataset": data["label"],
                "rows": format_int(data["row_count"]),
                "schema": data["schema"],
                "intent_accuracy": format_metric(data["intent_accuracy"]),
                "intent_macro_f1": format_metric(data["intent_macro_f1"]),
                "saved_slot_f1": format_metric(data["saved_slot_f1"]),
                "slot_micro_f1": format_metric(data["slot_micro_f1"]),
                "unknown_preds": format_int(data["scan"]["unknown_predictions"]),
                "slot_eval_mismatches": format_int(data["scan"]["slot_eval_mismatches"]),
            }
        )
    return table_html(rows, compact=True, preformatted=True)


def metric_kv_table(title: str, row: Mapping[str, str]) -> str:
    count_metrics = {
        "rows",
        "intent_count",
        "intent_correct",
        "intent_errors",
        "tp",
        "fp",
        "fn",
        "micro_tp",
        "micro_fp",
        "micro_fn",
        "support",
        "support_gold",
        "support_pred",
        "predicted",
        "correct",
        "errors",
    }
    metric_words = ("accuracy", "precision", "recall", "f1")

    def format_summary_value(metric: str, value: str) -> str:
        number = to_float(value)
        if number is None:
            return value
        metric_key = metric.lower()
        if metric_key in count_metrics:
            return format_int(int(number))
        if any(word in metric_key for word in metric_words):
            return format_metric(number)
        return format_decimal(number)

    rows = [{"metric": key, "value": format_summary_value(key, value)} for key, value in row.items()]
    return f"<h4>{esc(title)}</h4>" + table_html(rows, compact=True, preformatted=True)


def dataset_section(data: Mapping[str, Any]) -> str:
    total = data["row_count"] or 0
    slot_by_intent_html = (
        table_html(data["slot_by_intent"], compact=True)
        if data["slot_by_intent"]
        else '<p class="note">slot_conll_by_intent.csv is not present in the saved metrics for this dataset.</p>'
    )
    return f'''
    <section id="{esc(data['dataset'])}" class="dataset-section">
        <div class="section-title">
            <h2>{esc(data['label'])}</h2>
            <span class="schema {'schema-new' if data['schema'] == 'macro/micro schema' else 'schema-legacy'}">{esc(data['schema'])}</span>
        </div>
        <div class="mini-grid">
            <div class="metric-card"><b>{format_int(data['row_count'])}</b><span>prediction rows</span></div>
            <div class="metric-card"><b>{format_metric(data['intent_accuracy'])}</b><span>intent accuracy</span></div>
            <div class="metric-card"><b>{format_metric(data['intent_macro_f1'])}</b><span>intent macro-F1</span></div>
            <div class="metric-card"><b>{format_metric(data['saved_slot_f1'])}</b><span>saved slot F1</span></div>
        </div>

        <details open>
            <summary>Saved summaries</summary>
            <div class="three-cols">
                <div>{metric_kv_table('overall_summary.csv', data['overall'])}</div>
                <div>{metric_kv_table('intent_summary.csv', data['intent_summary'])}</div>
                <div>{metric_kv_table('slot_conll_summary.csv', data['slot_summary'])}</div>
            </div>
        </details>

        <details open>
            <summary>Intent and prediction distributions</summary>
            <div class="two-cols">
                <div><h4>Gold semantic_intent distribution</h4>{count_table(data['scan']['semantic_counts'], total, 'semantic_intent')}</div>
                <div><h4>Predicted intent distribution</h4>{count_table(data['scan']['pred_counts'], total, 'pred_intent')}</div>
            </div>
        </details>

        <details>
            <summary>Intent by label</summary>
            {table_html(data['intent_by_label'], compact=True)}
        </details>

        <details>
            <summary>Slot CoNLL by intent</summary>
            {slot_by_intent_html}
        </details>

        <details>
            <summary>Slot CoNLL by label</summary>
            {table_html(data['slot_by_label'], compact=True)}
        </details>

        <details>
            <summary>Intent confusion matrix</summary>
            {table_html(data['intent_confusion'], compact=True)}
        </details>

        <details>
            <summary>Artifact integrity</summary>
            <h4>Expected core artifacts</h4>
            {expected_artifacts_table(data)}
            <h4>Result CSV files</h4>
            {artifact_table(data['result_files'])}
            <h4>Metric CSV files</h4>
            {artifact_table(data['metric_files'])}
        </details>
    </section>
    '''


def stratified_metric(summary_rows: Sequence[Mapping[str, str]], metric: str) -> Mapping[str, str]:
    for row in summary_rows:
        if row.get("metric") == metric:
            return row
    return {}


def format_ci_metric(row: Mapping[str, str]) -> str:
    mean_value = to_float(row.get("mean"))
    ci_low = to_float(row.get("ci95_low"))
    ci_high = to_float(row.get("ci95_high"))
    if mean_value is None:
        return "-"
    if ci_low is None or ci_high is None:
        return format_metric(mean_value)
    return f"{format_metric(mean_value)} ({format_metric(ci_low)} .. {format_metric(ci_high)})"


def stratified_spotlight(stratified: Mapping[str, Any], datasets: Sequence[Mapping[str, Any]]) -> str:
    if not stratified["summary"]:
        return ""

    atis = next((data for data in datasets if data["dataset"] == "atis"), None)
    summary = stratified["summary"]
    coverage = stratified["coverage"]
    rounds = stratified["rounds"]
    intent_accuracy = stratified_metric(summary, "intent_accuracy")
    intent_macro_f1 = stratified_metric(summary, "intent_macro_f1")
    slot_macro_f1 = stratified_metric(summary, "slot_macro_f1_by_intent")
    slot_micro_f1 = stratified_metric(summary, "slot_micro_f1")

    effective_per_round = sum(to_int(row.get("effective_per_round")) or 0 for row in coverage)
    intent_count = len(coverage)
    n_value = coverage[0].get("N") if coverage else "50"
    k_value = coverage[0].get("K") if coverage else str(len(rounds))
    seed_value = coverage[0].get("seed") if coverage else "42"

    comparison_rows = [
        {
            "view": "ATIS saved main metrics",
            "rows/sample": format_int(atis["row_count"]) if atis else "-",
            "intent_accuracy": format_metric(atis["intent_accuracy"]) if atis else "-",
            "intent_macro_f1": format_metric(atis["intent_macro_f1"]) if atis else "-",
            "slot_macro_or_saved_f1": format_metric(atis["saved_slot_f1"]) if atis else "-",
            "slot_micro_f1": format_metric(atis["slot_micro_f1"]) if atis else "-",
        },
        {
            "view": "ATIS stratified Macro@N",
            "rows/sample": f"{format_int(effective_per_round)} per round x {k_value}",
            "intent_accuracy": format_ci_metric(intent_accuracy),
            "intent_macro_f1": format_ci_metric(intent_macro_f1),
            "slot_macro_or_saved_f1": format_ci_metric(slot_macro_f1),
            "slot_micro_f1": format_ci_metric(slot_micro_f1),
        },
    ]

    return f'''
    <section class="strat-spotlight" aria-label="ATIS Stratified Macro@N spotlight">
        <div class="spotlight-copy">
            <span class="eyebrow">Featured analysis</span>
            <h2>ATIS Stratified Macro@N</h2>
            <p>Balanced post-prediction analysis with N={esc(n_value)} examples per intent, K={esc(k_value)} rounds, seed={esc(seed_value)}. Each round samples {format_int(effective_per_round)} rows across {format_int(intent_count)} intents, making the ATIS result much easier to interpret despite the original class imbalance.</p>
            <a class="button-link" href="#atis-stratified">Jump to full stratified section</a>
        </div>
        <div class="spotlight-metrics">
            <div class="spotlight-card primary">
                <span>Intent Macro-F1</span>
                <b>{format_metric(to_float(intent_macro_f1.get('mean')))}</b>
                <small>95% CI {format_metric(to_float(intent_macro_f1.get('ci95_low')))} .. {format_metric(to_float(intent_macro_f1.get('ci95_high')))}</small>
            </div>
            <div class="spotlight-card">
                <span>Intent Accuracy</span>
                <b>{format_metric(to_float(intent_accuracy.get('mean')))}</b>
                <small>balanced rounds mean</small>
            </div>
            <div class="spotlight-card">
                <span>Slot Macro-F1 By Intent</span>
                <b>{format_metric(to_float(slot_macro_f1.get('mean')))}</b>
                <small>slot extension of the old analysis</small>
            </div>
            <div class="spotlight-card">
                <span>Slot Micro-F1</span>
                <b>{format_metric(to_float(slot_micro_f1.get('mean')))}</b>
                <small>global slot counts on sampled rounds</small>
            </div>
        </div>
        <div class="spotlight-table">
            <h3>ATIS main vs stratified</h3>
            {table_html(comparison_rows, compact=True, preformatted=True)}
        </div>
    </section>
    '''


def stratified_section(stratified: Mapping[str, Any]) -> str:
    if not any(info["exists"] for info in stratified["artifacts"].values()):
        return '<section id="atis-stratified"><h2>ATIS Stratified Macro@N</h2><p class="note">No stratified artifacts found.</p></section>'

    summary_rows = stratified["summary"]
    coverage_rows = stratified["coverage"]
    rounds_rows = stratified["rounds"]
    summary_display_rows = [
        {
            "metric": row.get("metric", ""),
            "mean": format_metric(to_float(row.get("mean"))),
            "ci95_low": format_metric(to_float(row.get("ci95_low"))),
            "ci95_high": format_metric(to_float(row.get("ci95_high"))),
            "N": row.get("N", ""),
            "K": row.get("K", ""),
            "seed": row.get("seed", ""),
        }
        for row in summary_rows
    ]
    artifact_rows = []
    for name, info in stratified["artifacts"].items():
        artifact_rows.append(
            {
                "artifact": name,
                "status": "found" if info["exists"] else "missing",
                "file": relpath(info["path"]),
                "size": format_bytes(info["bytes"]),
                "modified": info["modified"],
            }
        )
    return f'''
    <section id="atis-stratified" class="dataset-section stratified">
        <div class="section-title">
            <h2>ATIS Stratified Macro@N</h2>
            <span class="schema schema-new">n50_k30_seed42</span>
        </div>
        <p class="muted">Post-prediction balanced sampling analysis. This section reads saved CSVs from {esc(relpath(stratified['dir']))}.</p>
        {line_chart_svg(rounds_rows)}
        <details open>
            <summary>Stratified summary</summary>
            {table_html(summary_display_rows, compact=True, preformatted=True)}
        </details>
        <details open>
            <summary>Coverage by intent</summary>
            {table_html(coverage_rows, compact=True)}
        </details>
        <details>
            <summary>Rounds</summary>
            {table_html(rounds_rows, compact=True)}
        </details>
        <details>
            <summary>Stratified artifact integrity</summary>
            {table_html(artifact_rows, compact=True)}
        </details>
    </section>
    '''


def page_css() -> str:
    return """
    :root {
        --bg: #f6f7f9;
        --panel: #ffffff;
        --ink: #1f2933;
        --muted: #687383;
        --line: #d8dde6;
        --blue: #1565c0;
        --teal: #00897b;
        --orange: #ef6c00;
        --green-bg: #e7f5ee;
        --green: #176b45;
        --warn-bg: #fff4df;
        --warn: #8a4d00;
        --legacy-bg: #f3eefc;
        --legacy: #5b3b8c;
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
        padding: 28px 36px 18px;
        background: #ffffff;
        border-bottom: 1px solid var(--line);
    }
    h1, h2, h3, h4 { margin: 0 0 10px; letter-spacing: 0; }
    h1 { font-size: 30px; }
    h2 { font-size: 23px; }
    h3 { font-size: 18px; }
    h4 { font-size: 15px; color: var(--muted); }
    .muted { color: var(--muted); }
    .note, .warning {
        border: 1px solid var(--line);
        border-left: 4px solid var(--orange);
        background: #fffaf0;
        padding: 12px 14px;
        border-radius: 6px;
        color: #4c3a18;
    }
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
    .cards {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 14px;
        margin: 18px 0 22px;
    }
    .card, .dataset-section, .panel {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    }
    .card { padding: 16px; min-width: 0; }
    .card-head, .section-title {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        flex-wrap: wrap;
    }
    .big { font-size: 34px; font-weight: 750; margin-top: 8px; }
    .label { color: var(--muted); font-size: 13px; margin-bottom: 10px; }
    dl {
        display: grid;
        grid-template-columns: 1fr auto;
        gap: 6px 12px;
        margin: 14px 0 0;
        font-size: 13px;
    }
    dt { color: var(--muted); }
    dd { margin: 0; font-weight: 650; }
    .schema, .badge {
        display: inline-flex;
        align-items: center;
        white-space: nowrap;
        border-radius: 999px;
        padding: 4px 9px;
        font-size: 12px;
        font-weight: 700;
    }
    .schema-new, .badge.good { background: var(--green-bg); color: var(--green); }
    .schema-legacy { background: var(--legacy-bg); color: var(--legacy); }
    .badge.warn { background: var(--warn-bg); color: var(--warn); }
    .badges { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px; min-height: 24px; }
    .bar {
        position: relative;
        height: 22px;
        background: #edf1f5;
        border-radius: 5px;
        overflow: hidden;
        min-width: 120px;
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
    .bar.empty span { color: var(--muted); }
    .strat-spotlight {
        display: grid;
        grid-template-columns: minmax(280px, 0.9fr) minmax(420px, 1.25fr);
        gap: 18px;
        margin: 22px 0;
        padding: 22px;
        background: linear-gradient(135deg, #ffffff 0%, #eef7f5 100%);
        border: 2px solid #9ed8cd;
        border-radius: 8px;
        box-shadow: 0 10px 28px rgba(0, 105, 92, 0.08);
    }
    .strat-spotlight .eyebrow {
        display: inline-block;
        color: var(--teal);
        font-size: 12px;
        font-weight: 800;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .strat-spotlight h2 {
        font-size: 28px;
        margin-bottom: 8px;
    }
    .button-link {
        display: inline-flex;
        margin-top: 10px;
        padding: 8px 12px;
        background: var(--teal);
        color: #ffffff;
        text-decoration: none;
        border-radius: 6px;
        font-size: 13px;
        font-weight: 750;
    }
    .spotlight-metrics {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
    }
    .spotlight-card {
        background: rgba(255, 255, 255, 0.86);
        border: 1px solid #b9ddd6;
        border-radius: 8px;
        padding: 14px;
    }
    .spotlight-card.primary {
        background: #e4f5f1;
        border-color: var(--teal);
    }
    .spotlight-card span {
        display: block;
        color: var(--muted);
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .spotlight-card b {
        display: block;
        font-size: 30px;
        line-height: 1.1;
    }
    .spotlight-card small {
        color: var(--muted);
        font-size: 12px;
    }
    .spotlight-table {
        grid-column: 1 / -1;
        background: rgba(255, 255, 255, 0.7);
        border: 1px solid #cfe7e2;
        border-radius: 8px;
        padding: 12px;
    }
    .dataset-section, .panel {
        padding: 20px;
        margin-top: 20px;
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
    .metric-card b { display: block; font-size: 22px; }
    .metric-card span { color: var(--muted); font-size: 12px; }
    .two-cols, .three-cols {
        display: grid;
        gap: 16px;
        align-items: start;
    }
    .two-cols { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .three-cols { grid-template-columns: repeat(3, minmax(0, 1fr)); }
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
    td {
        padding: 7px 9px;
        border-bottom: 1px solid #eef1f5;
        vertical-align: top;
    }
    tr:hover td { background: #fbfdff; }
    .distribution td:last-child { min-width: 160px; }
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
        .cards, .mini-grid, .three-cols { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .two-cols { grid-template-columns: 1fr; }
        .strat-spotlight { grid-template-columns: 1fr; }
    }
    @media (max-width: 700px) {
        header, main, footer { padding-left: 18px; padding-right: 18px; }
        .cards, .mini-grid, .three-cols { grid-template-columns: 1fr; }
        .spotlight-metrics { grid-template-columns: 1fr; }
    }
    """


def build_html(datasets: Sequence[Mapping[str, Any]], stratified: Mapping[str, Any]) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dataset_sections = "\n".join(dataset_section(data) for data in datasets)
    return f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Batch Eval Dashboard</title>
    <style>{page_css()}</style>
</head>
<body>
    <header>
        <h1>Batch Eval Dashboard</h1>
        <p class="muted">Static view generated from saved CSV artifacts in <code>evaluation/</code>. Generated at {esc(generated_at)}.</p>
        <nav>
            <a href="#overview">Overview</a>
            <a href="#atis-stratified">ATIS Stratified</a>
            <a href="#integrity">Artifact Integrity</a>
            {''.join(f'<a href="#{esc(data["dataset"])}">{esc(data["label"])}</a>' for data in datasets)}
        </nav>
    </header>
    <main>
        <section id="overview">
            <h2>Executive Overview</h2>
            <p class="warning"><b>Interpretation note:</b> this dashboard uses saved metrics exactly as found on disk. The current reports are expected to use the macro/micro schema, including Intent Macro-F1 and Slot Macro-F1 by Intent. If a dataset is ever marked as legacy, compare its slot-F1 with care.</p>
            {summary_cards(datasets)}
            {stratified_spotlight(stratified, datasets)}
            <div class="panel">
                <h3>Dataset comparison</h3>
                {comparison_table(datasets)}
            </div>
        </section>

        {stratified_section(stratified)}

        <section id="integrity" class="panel">
            <h2>Artifact Integrity</h2>
            <p class="muted">Core expected files and aggregate result row counts.</p>
            {table_html([
                {
                    "dataset": data["label"],
                    "aggregate_predictions": "found" if data["predictions_path"].exists() else "missing",
                    "rows": format_int(data["row_count"]),
                    "metrics_schema": data["schema"],
                    "metric_files_found": f"{len(data['metric_files'])}/{len(METRIC_FILES)}",
                    "result_csv_files": format_int(len(data["result_files"])),
                }
                for data in datasets
            ], compact=True, preformatted=True)}
        </section>

        {dataset_sections}
    </main>
    <footer>
        Generated by src/data_tools/build_eval_dashboard.py. This page does not run inference or recompute metrics.
    </footer>
</body>
</html>
"""


def main() -> None:
    datasets = [load_dataset(dataset, label) for dataset, label in DATASETS]
    stratified = load_stratified()
    html_text = build_html(datasets, stratified)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(html_text, encoding="utf-8")
    print(f"Wrote {relpath(OUTPUT_PATH)}")


if __name__ == "__main__":
    main()
