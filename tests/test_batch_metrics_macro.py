import json
import unittest

import pandas as pd

from src.data_tools.batch_metrics_utils import (
    compute_batch_metrics_reports,
    compute_intent_accuracy_reports,
    compute_slot_conll_reports,
)


class BatchMetricsMacroTests(unittest.TestCase):
    def test_slot_summary_uses_macro_f1_by_intent_and_keeps_micro_columns(self):
        rows = []
        perfect_chunk = json.dumps([["slot_a", 0, 1]])
        missed_chunk = json.dumps([["slot_b", 0, 1]])
        for index in range(9):
            rows.append(
                {
                    "semantic_intent": "IntentA",
                    "gold_chunks_json": perfect_chunk,
                    "pred_chunks_json": perfect_chunk,
                }
            )
        rows.append(
            {
                "semantic_intent": "IntentB",
                "gold_chunks_json": missed_chunk,
                "pred_chunks_json": json.dumps([]),
            }
        )
        evaluation_examples = pd.DataFrame(rows)

        summary, by_label, by_intent = compute_slot_conll_reports(evaluation_examples)
        summary_row = summary.iloc[0]
        intent_f1 = dict(zip(by_intent["semantic_intent"], by_intent["f1"]))

        self.assertAlmostEqual(intent_f1["IntentA"], 1.0)
        self.assertAlmostEqual(intent_f1["IntentB"], 0.0)
        self.assertAlmostEqual(summary_row["f1"], 0.5)
        self.assertAlmostEqual(summary_row["micro_f1"], 18 / 19)
        self.assertIn("micro_tp", summary.columns)
        self.assertIn("micro_fp", summary.columns)
        self.assertIn("micro_fn", summary.columns)
        self.assertEqual(int(summary_row["tp"]), 9)
        self.assertEqual(int(summary_row["fn"]), 1)
        self.assertEqual(set(by_label["semantic_slot"].tolist()), {"slot_a", "slot_b"})

    def test_intent_macro_f1_ignores_unk_as_gold_class_but_counts_it_as_error(self):
        evaluation_examples = pd.DataFrame(
            [
                {
                    "semantic_intent": "IntentA",
                    "pred_intent": "IntentA",
                    "intent_correct": True,
                },
                {
                    "semantic_intent": "IntentA",
                    "pred_intent": "UNK_INTENT",
                    "intent_correct": False,
                },
                {
                    "semantic_intent": "IntentB",
                    "pred_intent": "IntentB",
                    "intent_correct": True,
                },
            ]
        )

        summary, by_label, confusion = compute_intent_accuracy_reports(evaluation_examples)
        summary_row = summary.iloc[0]
        by_intent = by_label.set_index("semantic_intent")

        self.assertAlmostEqual(by_intent.loc["IntentA", "f1"], 2 / 3)
        self.assertAlmostEqual(by_intent.loc["IntentB", "f1"], 1.0)
        self.assertEqual(int(by_intent.loc["UNK_INTENT", "support"]), 0)
        self.assertEqual(int(by_intent.loc["UNK_INTENT", "fp"]), 1)
        self.assertAlmostEqual(summary_row["macro_f1"], (2 / 3 + 1.0) / 2)
        self.assertAlmostEqual(summary_row["micro_f1"], 2 / 3)
        self.assertEqual(int(summary_row["intent_count"]), 2)
        self.assertIn("UNK_INTENT", confusion.columns)
        self.assertEqual(int(confusion.loc["IntentA", "UNK_INTENT"]), 1)

    def test_batch_reports_include_slot_by_intent(self):
        semantic_view = pd.DataFrame()

        reports = compute_batch_metrics_reports(semantic_view, registry={"unused": {}})

        self.assertIn("slot_conll_by_intent", reports)
        self.assertTrue(reports["slot_conll_by_intent"].empty)


if __name__ == "__main__":
    unittest.main()
