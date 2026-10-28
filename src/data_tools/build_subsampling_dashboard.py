from __future__ import annotations

import csv
import html
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ROOT = REPO_ROOT / "evaluation"
OUTPUT_PATH = EVALUATION_ROOT / "reports" / "batch_eval_subsampling_dashboard.html"

SUBSAMPLING_METHOD = "stratified_by_intent"
SUBSAMPLING_CONFIG = "r1000_seed42"

DATASETS = [
    ("field_operations", "Field Operations", "Primary domain analysis"),
    ("snips", "SNIPS", "Complementary benchmark"),
    ("atis", "ATIS", "Complementary benchmark"),
    ("multiwoz", "MultiWOZ", "Complementary benchmark"),
]

SUBSAMPLING_FILES = [
    "summary.csv",
    "rounds.csv",
    "coverage.csv",
    "intent_by_label_rounds.csv",
    "slot_by_intent_rounds.csv",
    "slot_by_label_rounds.csv",
]

KEY_METRICS = [
    "intent_accuracy",
    "intent_macro_f1",
    "slot_macro_f1_by_intent",
    "slot_micro_f1",
]

CURVE_METRICS = [
    ("intent_macro_f1", "Intent Macro-F1", "#1565c0"),
    ("slot_macro_f1_by_intent", "Slot Macro-F1 by Intent", "#00897b"),
    ("slot_micro_f1", "Slot Micro-F1", "#ef6c00"),
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


def css_class_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in value.lower()).strip("-")


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
            text = str(value)
            if not preformatted:
                numeric = to_float(value)
                lower = column.lower()
                count_columns = {
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
                    "sampled_per_round",
                    "total_sampled_across_rounds",
                    "size_bytes",
                    "csv_rows",
                    "intent_count",
                    "tp",
                    "fp",
                    "fn",
                    "predicted",
                    "correct",
                    "errors",
                }
                if numeric is not None and (lower in count_columns or lower.endswith("_support")):
                    text = format_int(int(numeric))
                elif numeric is not None and any(word in lower for word in ("accuracy", "precision", "recall", "f1", "mean", "std", "min", "max", "median", "p2_5", "p97_5")):
                    text = format_metric(numeric)
                elif numeric is not None:
                    text = format_decimal(numeric)
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


def load_subsampling_dataset(dataset: str, label: str, role: str) -> Dict[str, Any]:
    root = EVALUATION_ROOT / dataset / "metrics_subsampling" / SUBSAMPLING_METHOD / SUBSAMPLING_CONFIG
    paths = {name: root / name for name in SUBSAMPLING_FILES}
    summary = read_csv(paths["summary.csv"])
    rounds = read_csv(paths["rounds.csv"])
    coverage = read_csv(paths["coverage.csv"])
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
        "rounds": rounds,
        "coverage": coverage,
        "intent_by_label_rounds": intent_by_label_rounds,
        "slot_by_intent_rounds": slot_by_intent_rounds,
        "slot_by_label_rounds": slot_by_label_rounds,
        "metric_rows": metric_rows,
        "n_values": n_values,
        "primary_n": primary_n,
        "full_anchor_n": full_anchor_n,
        "full_anchor_by_n": full_anchor_by_n,
    }


def metric_row(data: Mapping[str, Any], metric: str, n: int | None = None) -> Mapping[str, str] | None:
    n = data.get("primary_n") if n is None else n
    if n is None:
        return None
    return data["metric_rows"].get((int(n), metric))


def metric_mean(data: Mapping[str, Any], metric: str, n: int | None = None) -> float | None:
    row = metric_row(data, metric, n=n)
    return to_float(row.get("mean")) if row else None


def artifact_table(data: Mapping[str, Any]) -> str:
    rows = []
    for name, info in data["artifacts"].items():
        rows.append(
            {
                "artifact": name,
                "status": "found" if info["exists"] else "missing",
                "file": relpath(info["path"]),
                "size": format_bytes(info["bytes"]),
                "csv_rows": format_int(info["rows"]),
                "modified": info["modified"],
            }
        )
    return table_html(rows, compact=True, preformatted=True)


def coverage_table(data: Mapping[str, Any]) -> str:
    columns = [
        "semantic_intent",
        "support",
        "n_per_intent",
        "rounds_executed",
        "sampled_per_round",
        "total_sampled_across_rounds",
        "full_support_anchor",
    ]
    rows = sorted(data["coverage"], key=lambda row: (to_int(row.get("n_per_intent")) or 0, row.get("semantic_intent", "")))
    return table_html(rows, columns=columns, compact=True)


def summarize_round_table(data: Mapping[str, Any]) -> str:
    rows = []
    for n in data["n_values"]:
        intent = metric_row(data, "intent_macro_f1", n)
        intent_acc = metric_row(data, "intent_accuracy", n)
        slot_macro = metric_row(data, "slot_macro_f1_by_intent", n)
        slot_micro = metric_row(data, "slot_micro_f1", n)
        base = intent or slot_macro or slot_micro or {}
        rows.append(
            {
                "n_per_intent": n,
                "rows_per_round": base.get("rows_per_round", ""),
                "rounds": base.get("rounds", ""),
                "full_support_anchor": base.get("full_support_anchor", ""),
                "intent_accuracy": metric_interval(intent_acc),
                "intent_macro_f1": metric_interval(intent),
                "slot_macro_f1_by_intent": metric_interval(slot_macro),
                "slot_micro_f1": metric_interval(slot_micro),
            }
        )
    return table_html(rows, compact=True, preformatted=True)


def aggregate_by_label(
    rows: Sequence[Mapping[str, str]],
    *,
    n: int | None,
    label_column: str,
    support_column: str,
    limit: int | None = None,
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

    aggregated: List[Dict[str, Any]] = []
    for label, group in groups.items():
        def avg(column: str) -> float | None:
            values = [to_float(row.get(column)) for row in group]
            values = [value for value in values if value is not None]
            return sum(values) / len(values) if values else None

        aggregated.append(
            {
                label_column: label,
                "mean_support": avg(support_column),
                "mean_precision": avg("precision"),
                "mean_recall": avg("recall"),
                "mean_f1": avg("f1"),
                "mean_accuracy": avg("accuracy"),
                "rounds": len(group),
            }
        )

    aggregated.sort(
        key=lambda row: (
            1 if (row["mean_support"] or 0) <= 0 else 0,
            row["mean_f1"] if row["mean_f1"] is not None else 2.0,
            -(row["mean_support"] or 0),
            row[label_column],
        )
    )
    return aggregated[:limit] if limit is not None else aggregated


def intent_by_label_table(data: Mapping[str, Any]) -> str:
    rows = aggregate_by_label(
        data["intent_by_label_rounds"],
        n=data["primary_n"],
        label_column="semantic_intent",
        support_column="support",
    )
    return table_html(
        rows,
        columns=["semantic_intent", "mean_support", "mean_precision", "mean_recall", "mean_f1", "mean_accuracy", "rounds"],
        compact=True,
    )


def slot_by_intent_table(data: Mapping[str, Any]) -> str:
    rows = aggregate_by_label(
        data["slot_by_intent_rounds"],
        n=data["primary_n"],
        label_column="semantic_intent",
        support_column="support_gold",
    )
    return table_html(
        rows,
        columns=["semantic_intent", "mean_support", "mean_precision", "mean_recall", "mean_f1", "rounds"],
        compact=True,
    )


def slot_by_label_table(data: Mapping[str, Any]) -> str:
    rows = aggregate_by_label(
        data["slot_by_label_rounds"],
        n=data["primary_n"],
        label_column="semantic_slot",
        support_column="support_gold",
        limit=25,
    )
    return table_html(
        rows,
        columns=["semantic_slot", "mean_support", "mean_precision", "mean_recall", "mean_f1", "rounds"],
        compact=True,
    )


def curve_svg(data: Mapping[str, Any]) -> str:
    if not data["n_values"]:
        return '<p class="muted">No summary rows available.</p>'

    width, height = 920, 300
    left, right, top, bottom = 58, 28, 24, 56
    plot_w = width - left - right
    plot_h = height - top - bottom
    n_values = data["n_values"]
    min_n, max_n = min(n_values), max(n_values)

    def x_for_n(n: int) -> float:
        if min_n == max_n:
            return left + plot_w / 2
        return left + ((n - min_n) / (max_n - min_n)) * plot_w

    def y_for_value(value: float | None) -> float:
        value = 0.0 if value is None else max(0.0, min(1.0, value))
        return top + (1.0 - value) * plot_h

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(data["label"])} subsampling metric curves">']
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
            value = metric_mean(data, metric, n=n)
            if value is not None:
                points.append(f"{x_for_n(n):.1f},{y_for_value(value):.1f}")
        if points:
            parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="3"/>')
            for n in n_values:
                value = metric_mean(data, metric, n=n)
                if value is not None:
                    parts.append(f'<circle cx="{x_for_n(n):.1f}" cy="{y_for_value(value):.1f}" r="4" fill="{color}"/>')

    legend_x = left
    legend_y = height - 18
    for _metric, label, color in CURVE_METRICS:
        parts.append(f'<rect x="{legend_x}" y="{legend_y - 10}" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 18}" y="{legend_y}" class="legend">{esc(label)}</text>')
        legend_x += 250
    parts.append("</svg>")
    return "".join(parts)


def overview_cards(datasets: Sequence[Mapping[str, Any]]) -> str:
    best_intent = max(datasets, key=lambda data: metric_mean(data, "intent_macro_f1") or -1, default=None)
    best_slot = max(datasets, key=lambda data: metric_mean(data, "slot_macro_f1_by_intent") or -1, default=None)
    parts = ['<section class="cards">']
    for data in datasets:
        n = data["primary_n"]
        intent = metric_mean(data, "intent_macro_f1")
        slot = metric_mean(data, "slot_macro_f1_by_intent")
        badges = []
        if data["role"].startswith("Primary"):
            badges.append('<span class="badge primary">Primary</span>')
        if best_intent and best_intent["dataset"] == data["dataset"]:
            badges.append('<span class="badge good">Best intent macro-F1</span>')
        if best_slot and best_slot["dataset"] == data["dataset"]:
            badges.append('<span class="badge good">Best slot macro-F1</span>')
        parts.append(
            f'''
            <article class="card {css_class_slug(data['dataset'])}">
                <div class="card-head">
                    <h3>{esc(data['label'])}</h3>
                    <span class="schema">{esc(data['role'])}</span>
                </div>
                <div class="big">{format_metric(intent)}</div>
                <div class="label">Intent Macro-F1 @ n={esc(n)}</div>
                {metric_bar(intent)}
                <dl>
                    <dt>Primary n</dt><dd>{format_int(n)}</dd>
                    <dt>n values</dt><dd>{esc(', '.join(str(item) for item in data['n_values']))}</dd>
                    <dt>Slot Macro-F1</dt><dd>{format_metric(slot)}</dd>
                    <dt>Full anchor n</dt><dd>{format_int(data['full_anchor_n'])}</dd>
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
        n = data["primary_n"]
        base = metric_row(data, "intent_macro_f1")
        rows.append(
            {
                "dataset": data["label"],
                "role": data["role"],
                "primary_n": format_int(n),
                "rows_per_round": format_int(to_int(base.get("rows_per_round")) if base else None),
                "rounds": format_int(to_int(base.get("rounds")) if base else None),
                "intent_macro_f1": metric_interval(metric_row(data, "intent_macro_f1")),
                "intent_accuracy": metric_interval(metric_row(data, "intent_accuracy")),
                "slot_macro_f1_by_intent": metric_interval(metric_row(data, "slot_macro_f1_by_intent")),
                "slot_micro_f1": metric_interval(metric_row(data, "slot_micro_f1")),
            }
        )
    return table_html(rows, compact=True, preformatted=True)


def field_operations_spotlight(datasets: Sequence[Mapping[str, Any]]) -> str:
    field = next((data for data in datasets if data["dataset"] == "field_operations"), None)
    if not field:
        return ""
    primary_n = field["primary_n"]
    full_n = field["full_anchor_n"]
    full_intent = metric_row(field, "intent_macro_f1", full_n) if full_n is not None else None
    full_slot = metric_row(field, "slot_macro_f1_by_intent", full_n) if full_n is not None else None
    return f'''
    <section class="spotlight" id="field-spotlight">
        <div>
            <span class="eyebrow">Primary stability analysis</span>
            <h2>Field Operations Subsampling</h2>
            <p>Amostragem estratificada por gold intent, sem rerodar inferencia. O n={primary_n} e o maior ponto estocastico; n={full_n} e ancora do dataset completo.</p>
            <a class="button-link" href="#field_operations">Open Field Operations section</a>
        </div>
        <div class="spotlight-metrics">
            <div class="spotlight-card primary">
                <span>Intent Macro-F1 @ n={primary_n}</span>
                <b>{format_metric(metric_mean(field, "intent_macro_f1"))}</b>
                <small>{esc(metric_interval(metric_row(field, "intent_macro_f1")))}</small>
            </div>
            <div class="spotlight-card primary">
                <span>Slot Macro-F1 by Intent @ n={primary_n}</span>
                <b>{format_metric(metric_mean(field, "slot_macro_f1_by_intent"))}</b>
                <small>{esc(metric_interval(metric_row(field, "slot_macro_f1_by_intent")))}</small>
            </div>
            <div class="spotlight-card">
                <span>Full anchor Intent Macro-F1 @ n={full_n}</span>
                <b>{format_metric(to_float(full_intent.get("mean")) if full_intent else None)}</b>
                <small>{esc(metric_interval(full_intent))}</small>
            </div>
            <div class="spotlight-card">
                <span>Full anchor Slot Macro-F1 @ n={full_n}</span>
                <b>{format_metric(to_float(full_slot.get("mean")) if full_slot else None)}</b>
                <small>{esc(metric_interval(full_slot))}</small>
            </div>
        </div>
    </section>
    '''


def artifact_integrity_table(datasets: Sequence[Mapping[str, Any]]) -> str:
    rows = []
    for data in datasets:
        found = sum(1 for info in data["artifacts"].values() if info["exists"])
        summary = data["artifacts"].get("summary.csv", {})
        rounds = data["artifacts"].get("rounds.csv", {})
        rows.append(
            {
                "dataset": data["label"],
                "status": f"{found}/{len(SUBSAMPLING_FILES)} found",
                "summary_rows": format_int(summary.get("rows")),
                "round_rows": format_int(rounds.get("rows")),
                "output_dir": relpath(data["root"]),
                "last_modified": max((info["modified"] for info in data["artifacts"].values()), default=""),
            }
        )
    return table_html(rows, compact=True, preformatted=True)


def dataset_section(data: Mapping[str, Any]) -> str:
    primary_n = data["primary_n"]
    rounds_preview_columns = [
        "n_per_intent",
        "round",
        "rows",
        "intent_macro_f1",
        "slot_macro_f1_by_intent",
        "slot_micro_f1",
    ]
    return f'''
    <section id="{esc(data['dataset'])}" class="dataset-section">
        <div class="section-title">
            <div>
                <h2>{esc(data['label'])}</h2>
                <p class="muted">{esc(data['role'])}. Primary n for per-label diagnostics: <b>{format_int(primary_n)}</b>.</p>
            </div>
            <span class="schema">{esc(SUBSAMPLING_METHOD)} / {esc(SUBSAMPLING_CONFIG)}</span>
        </div>

        <div class="mini-grid">
            <div class="metric-card"><span>Intent Macro-F1</span><b>{format_metric(metric_mean(data, "intent_macro_f1"))}</b></div>
            <div class="metric-card"><span>Intent Accuracy</span><b>{format_metric(metric_mean(data, "intent_accuracy"))}</b></div>
            <div class="metric-card"><span>Slot Macro-F1 by Intent</span><b>{format_metric(metric_mean(data, "slot_macro_f1_by_intent"))}</b></div>
            <div class="metric-card"><span>Slot Micro-F1</span><b>{format_metric(metric_mean(data, "slot_micro_f1"))}</b></div>
        </div>

        <h3>Metric curves by n</h3>
        {curve_svg(data)}

        <h3>Summary by sample size</h3>
        {summarize_round_table(data)}

        <div class="two-cols">
            <div>
                <h3>Coverage by intent and n</h3>
                {coverage_table(data)}
            </div>
            <div>
                <h3>Artifacts</h3>
                {artifact_table(data)}
            </div>
        </div>

        <details open>
            <summary>Intent metrics by label @ primary n</summary>
            {intent_by_label_table(data)}
        </details>
        <details>
            <summary>Slot metrics by intent @ primary n</summary>
            {slot_by_intent_table(data)}
        </details>
        <details>
            <summary>Hardest slot labels @ primary n</summary>
            {slot_by_label_table(data)}
        </details>
        <details>
            <summary>Rounds preview</summary>
            {table_html(data['rounds'], columns=rounds_preview_columns, compact=True, limit=12)}
        </details>
    </section>
    '''


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
        --primary-bg: #eaf4ff;
        --primary: #155a9c;
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
        padding: 30px 36px 18px;
        background: var(--panel);
        border-bottom: 1px solid var(--line);
    }
    h1, h2, h3 { margin: 0 0 8px; line-height: 1.15; }
    h1 { font-size: 32px; }
    h2 { font-size: 24px; }
    h3 { font-size: 17px; margin-top: 12px; }
    code {
        background: #eef2f6;
        border: 1px solid #dfe5ec;
        border-radius: 5px;
        padding: 1px 5px;
    }
    .muted { color: var(--muted); }
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
    .note, .warning {
        border: 1px solid var(--line);
        border-left: 4px solid var(--orange);
        background: #fffaf0;
        padding: 12px 14px;
        border-radius: 6px;
        color: #4c3a18;
    }
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
    dd { margin: 0; font-weight: 650; text-align: right; }
    .schema, .badge {
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
    .badge.good { background: var(--green-bg); color: var(--green); }
    .badge.primary { background: var(--primary-bg); color: var(--primary); }
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
    .spotlight {
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
    .eyebrow {
        display: inline-block;
        color: var(--teal);
        font-size: 12px;
        font-weight: 800;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .spotlight h2 { font-size: 28px; }
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
    .spotlight-card small { color: var(--muted); font-size: 12px; }
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
        .cards, .mini-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .two-cols, .spotlight { grid-template-columns: 1fr; }
    }
    @media (max-width: 700px) {
        header, main, footer { padding-left: 18px; padding-right: 18px; }
        .cards, .mini-grid, .spotlight-metrics { grid-template-columns: 1fr; }
    }
    """


def build_html(datasets: Sequence[Mapping[str, Any]]) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sections = "\n".join(dataset_section(data) for data in datasets)
    return f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Batch Eval Subsampling Dashboard</title>
    <style>{page_css()}</style>
</head>
<body>
    <header>
        <h1>Batch Eval Subsampling Dashboard</h1>
        <p class="muted">Static view generated from saved <code>metrics_subsampling</code> CSV artifacts. Generated at {esc(generated_at)}.</p>
        <nav>
            <a href="#overview">Overview</a>
            <a href="#field-spotlight">Field Operations Spotlight</a>
            <a href="#integrity">Artifact Integrity</a>
            {''.join(f'<a href="#{esc(data["dataset"])}">{esc(data["label"])}</a>' for data in datasets)}
        </nav>
    </header>
    <main>
        <section id="overview">
            <h2>Executive Overview</h2>
            <p class="warning"><b>Interpretation note:</b> this dashboard uses saved subsampling metrics exactly as found on disk. No inference is run and no threshold is tuned here. Intervals are empirical 2.5%-97.5% ranges across rounds, not new model training uncertainty.</p>
            {overview_cards(datasets)}
            {field_operations_spotlight(datasets)}
            <div class="panel">
                <h3>Dataset comparison @ primary n</h3>
                {comparison_table(datasets)}
            </div>
        </section>

        <section id="integrity" class="panel">
            <h2>Artifact Integrity</h2>
            <p class="muted">Expected subsampling files under <code>evaluation/&lt;dataset&gt;/metrics_subsampling/{esc(SUBSAMPLING_METHOD)}/{esc(SUBSAMPLING_CONFIG)}/</code>.</p>
            {artifact_integrity_table(datasets)}
        </section>

        {sections}
    </main>
    <footer>
        Generated by src/data_tools/build_subsampling_dashboard.py. This page only reads saved subsampling CSVs.
    </footer>
</body>
</html>
"""


def main() -> None:
    datasets = [load_subsampling_dataset(dataset, label, role) for dataset, label, role in DATASETS]
    html_text = build_html(datasets)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(html_text, encoding="utf-8")
    print(f"Wrote {relpath(OUTPUT_PATH)}")


if __name__ == "__main__":
    main()
