import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from src.common.text_utils import process_until_no_more_groups
from src.core import pipeline as core_pipeline
from src.core.nlu_processor import NLUProcessor
from src.data_tools import batch_eval_utils


class ThresholdSplitTests(unittest.TestCase):
    def test_build_semantic_intent_candidates_by_dataset_groups_candidates_per_dataset(self):
        with patch.object(
            batch_eval_utils,
            "build_semantic_intent_slot_dict",
            side_effect=[
                {"Intent A": {}},
                {"Intent B": {}},
            ],
        ):
            candidates = batch_eval_utils.build_semantic_intent_candidates_by_dataset(
                ["snips", "atis"],
                registry={
                    "snips": {"intents": {"raw_a": {"semantic_intent": "Intent A"}}},
                    "atis": {"intents": {"raw_b": {"semantic_intent": "Intent B"}}},
                },
            )

        self.assertEqual(candidates, {"snips": ["Intent A"], "atis": ["Intent B"]})

    def test_threshold_aliases_stay_in_sync(self):
        processor = NLUProcessor(SimpleNamespace(), tau_intent=0.3, tau_slots=0.6)

        self.assertAlmostEqual(processor.tau_intent, 0.3)
        self.assertAlmostEqual(processor.INTENT_THRESHOLD, 0.3)
        self.assertAlmostEqual(processor.tau_slots, 0.6)
        self.assertAlmostEqual(processor.SLOT_THRESHOLD, 0.6)

        processor.INTENT_THRESHOLD = 0.75
        processor.tau_slots = 0.85

        self.assertAlmostEqual(processor.tau_intent, 0.75)
        self.assertAlmostEqual(processor.INTENT_THRESHOLD, 0.75)
        self.assertAlmostEqual(processor.tau_slots, 0.85)
        self.assertAlmostEqual(processor.SLOT_THRESHOLD, 0.85)

    def test_intent_recognition_uses_tau_intent(self):
        models = SimpleNamespace(
            classifier=Mock(
                return_value={
                    "labels": ["Known intent", "Other intent"],
                    "scores": [0.79, 0.21],
                }
            )
        )
        processor = NLUProcessor(models, tau_intent=0.8)

        label, score = processor.intent_recognition(
            "hello world",
            {"Known intent": {}, "Other intent": {}},
        )

        self.assertEqual(label, "UNK_INTENT")
        self.assertEqual(score, 0.79)

    def test_batch_intent_recognition_prefers_tau_intent(self):
        processor = SimpleNamespace(
            tau_intent=0.8,
            INTENT_THRESHOLD=0.1,
            models=SimpleNamespace(
                classifier=Mock(
                    return_value=[
                        {
                            "labels": ["Known intent"],
                            "scores": [0.79],
                        }
                    ]
                )
            ),
        )

        predictions = batch_eval_utils.batch_intent_recognition(
            processor,
            ["hello world"],
            ["Known intent"],
            batch_size=1,
        )

        self.assertEqual(predictions, [("UNK_INTENT", 0.79)])

    def test_slot_associations_uses_tau_slots(self):
        processor = NLUProcessor(SimpleNamespace(), tau_slots=0.6)
        df_score = pd.DataFrame(
            [[0.59, 0.61]],
            index=["slot"],
            columns=["below-threshold", "above-threshold"],
        )

        result = processor.slot_associations(df_score)
        self.assertEqual(result["slot"], ("above-threshold", 0.61))

        processor.SLOT_THRESHOLD = 0.7
        result = processor.slot_associations(df_score)
        self.assertEqual(result["slot"], ("NULL", 0.0))

    def test_postprocessing_uses_tau_slots_before_and_after_bias(self):
        processor = NLUProcessor(SimpleNamespace(), tau_slots=0.5)
        df_score = pd.DataFrame([[0.4]], index=["slot"], columns=["candidate"])

        with patch.object(
            batch_eval_utils,
            "build_alias_collapsed_grouping_view",
            return_value=(df_score, {}),
        ), patch.object(
            batch_eval_utils,
            "process_until_no_more_groups",
            return_value=["candidate"],
        ) as grouping_mock:
            result = batch_eval_utils._run_single_sentence_postprocessing(
                processor,
                "candidate",
                df_score,
                {"slot": 0.3},
                {"slot": ["candidate"]},
            )

        self.assertEqual(grouping_mock.call_args.kwargs["threshold"], 0.5)
        self.assertEqual(result["slot"], ("candidate", 0.7))

    def test_grouping_fallback_semantics_are_preserved(self):
        df_score = pd.DataFrame(
            [[0.4, 0.3]],
            index=["slot"],
            columns=["best-candidate", "backup-candidate"],
        )

        selected = process_until_no_more_groups(df_score, threshold=0.9, fallback_top_k=1)

        self.assertEqual(selected, ["best-candidate"])

    def test_grouping_threshold_none_keeps_all_candidates_before_grouping(self):
        df_score = pd.DataFrame(
            [[0.0, -0.1]],
            index=["slot"],
            columns=["zero-candidate", "negative-candidate"],
        )

        selected = process_until_no_more_groups(df_score, threshold=None, fallback_top_k=0)

        self.assertEqual(selected, ["zero-candidate", "negative-candidate"])

    @patch("src.core.pipeline.NLUProcessor")
    @patch("src.core.pipeline.ModelLoader")
    def test_initialize_nlu_supports_legacy_tau_and_specific_overrides(self, model_loader_cls, processor_cls):
        model_loader = model_loader_cls.return_value
        processor = processor_cls.return_value

        result = core_pipeline.initialize_nlu(
            index_path="custom-index",
            device=2,
            tau=0.25,
            tau_slots=0.65,
        )

        model_loader_cls.assert_called_once_with(index_path="custom-index", device=2)
        processor_cls.assert_called_once_with(
            model_loader,
            tau_intent=0.25,
            tau_slots=0.65,
        )
        self.assertIs(result, processor)

    def test_force_intent_bypasses_intent_threshold_gate(self):
        processor = Mock()
        processor.tau_slots = 0.5
        processor.get_slot_tokens_new.return_value = ["candidate"]
        processor.get_entities.return_value = []
        processor.retrieval_search.return_value = {"title": []}
        processor.filter_doctitles.return_value = []
        processor.get_intent_slot_answers_score.return_value = ([], {}, {})
        processor.slot_scores.return_value = pd.DataFrame(
            [[0.8]],
            index=["slot"],
            columns=["candidate"],
        )
        processor.slot_associations.return_value = {"slot": ("candidate", 0.8)}

        with patch.object(core_pipeline, "build_alias_collapsed_grouping_view", return_value=(processor.slot_scores.return_value, {})), patch.object(
            core_pipeline,
            "process_until_no_more_groups",
            return_value=["candidate"],
        ), patch.object(
            core_pipeline,
            "add_row_scores_masked",
            side_effect=lambda df, *_: df,
        ):
            result = core_pipeline.intent_slot_pipeline(
                processor,
                "hello world",
                {"Forced intent": {"slots": {"slot": {"hypothesis": "h", "question": "q"}}}},
                force_intent="Forced intent",
            )

        processor.intent_recognition.assert_not_called()
        self.assertEqual(result["intent"], ("Forced intent", 1.0))

    def test_use_gold_intent_for_slots_keeps_slot_flow_independent_from_intent_abstention(self):
        semantic_view = pd.DataFrame(
            {
                "dataset_name": ["snips"],
                "sentence": ["hello world"],
                "semantic_intent": ["gold_intent"],
            }
        )
        intent_slot_dict = {
            "gold_intent": {
                "slots": {
                    "slot": {
                        "hypothesis": "The slot is {}",
                        "question": "What is the slot?",
                    }
                }
            }
        }
        nlu_processor = SimpleNamespace(
            tau_intent=0.9,
            tau_slots=0.5,
            models=SimpleNamespace(
                nlp=SimpleNamespace(
                    pipe=lambda sentences, batch_size, n_process: [SimpleNamespace(ents=[])] * len(sentences)
                )
            ),
        )

        slot_frame = pd.DataFrame([[0.8]], index=["slot"], columns=["candidate"])
        slot_assignment = {"slot": ("candidate", 0.8)}

        with patch.object(batch_eval_utils, "batch_intent_recognition", return_value=[("UNK_INTENT", 0.1)]), patch.object(
            batch_eval_utils,
            "_extract_slot_tokens_from_doc",
            return_value=["candidate"],
        ), patch.object(
            batch_eval_utils,
            "_extract_entities_from_doc",
            return_value=[],
        ), patch.object(
            batch_eval_utils,
            "_batch_retrieval_titles",
            return_value=[[]],
        ), patch.object(
            batch_eval_utils,
            "_batch_qa_for_intent",
            return_value=[([], {}, {})],
        ), patch.object(
            batch_eval_utils,
            "_build_slot_score_matrices_batch",
            return_value=[slot_frame],
        ), patch.object(
            batch_eval_utils,
            "_run_single_sentence_postprocessing",
            return_value=slot_assignment,
        ):
            gold_view, _ = batch_eval_utils.run_semantic_batch_inference(
                nlu_processor,
                semantic_view,
                intent_slot_dict,
                use_gold_intent_for_slots=True,
            )
            pred_view, _ = batch_eval_utils.run_semantic_batch_inference(
                nlu_processor,
                semantic_view,
                intent_slot_dict,
                use_gold_intent_for_slots=False,
            )

        self.assertEqual(
            json.loads(gold_view.loc[0, "pred_slots_semantic_json"]),
            {"slot": ["candidate"]},
        )
        self.assertEqual(
            json.loads(pred_view.loc[0, "pred_slots_semantic_json"]),
            {},
        )

    def test_run_semantic_batch_inference_single_dataset_keeps_legacy_behavior_without_candidate_map(self):
        semantic_view = pd.DataFrame(
            {
                "dataset_name": ["snips", "snips"],
                "sentence": ["hello", "world"],
                "semantic_intent": ["gold_intent", "gold_intent"],
            }
        )

        nlu_processor = SimpleNamespace()

        with patch.object(
            batch_eval_utils,
            "batch_intent_recognition",
            return_value=[("UNK_INTENT", 0.1), ("UNK_INTENT", 0.2)],
        ) as batch_mock:
            result_view, _ = batch_eval_utils.run_semantic_batch_inference(
                nlu_processor,
                semantic_view,
                {"gold_intent": {"slots": {}}},
                use_gold_intent_for_slots=False,
            )

        batch_mock.assert_called_once()
        self.assertEqual(result_view["pred_intent"].tolist(), ["UNK_INTENT", "UNK_INTENT"])
        self.assertEqual(result_view["pred_intent_score"].tolist(), [0.1, 0.2])

    def test_run_semantic_batch_inference_requires_candidate_map_for_multi_dataset_batches(self):
        semantic_view = pd.DataFrame(
            {
                "dataset_name": ["snips", "atis"],
                "sentence": ["hello", "world"],
                "semantic_intent": ["snips_intent", "atis_intent"],
            }
        )

        with self.assertRaisesRegex(ValueError, "intent_candidates_by_dataset"):
            batch_eval_utils.run_semantic_batch_inference(
                SimpleNamespace(),
                semantic_view,
                {},
                use_gold_intent_for_slots=False,
            )

    def test_run_semantic_batch_inference_uses_dataset_specific_candidates_and_preserves_order(self):
        semantic_view = pd.DataFrame(
            {
                "dataset_name": ["snips", "multiwoz", "snips", "atis"],
                "sentence": ["s1", "m1", "s2", "a1"],
                "semantic_intent": ["snips_gold", "multi_gold", "snips_gold", "atis_gold"],
            }
        )
        intent_candidates_by_dataset = {
            "snips": ["snips_intent_a", "snips_intent_b"],
            "multiwoz": ["multi_intent"],
            "atis": ["atis_intent"],
        }
        captured_candidates = []

        def fake_batch_intent_recognition(_processor, sentences, semantic_intents, batch_size=16, unknown_label="UNK_INTENT"):
            captured_candidates.append((list(sentences), list(semantic_intents), batch_size, unknown_label))
            semantic_intents = list(semantic_intents)
            if semantic_intents == intent_candidates_by_dataset["snips"]:
                return [("snips_pred_1", 0.91), ("snips_pred_2", 0.81)]
            if semantic_intents == intent_candidates_by_dataset["multiwoz"]:
                return [("multi_pred", 0.72)]
            if semantic_intents == intent_candidates_by_dataset["atis"]:
                return [("atis_pred", 0.63)]
            raise AssertionError(f"Unexpected semantic_intents: {semantic_intents}")

        with patch.object(
            batch_eval_utils,
            "batch_intent_recognition",
            side_effect=fake_batch_intent_recognition,
        ):
            result_view, stats = batch_eval_utils.run_semantic_batch_inference(
                SimpleNamespace(),
                semantic_view,
                {},
                use_gold_intent_for_slots=False,
                intent_candidates_by_dataset=intent_candidates_by_dataset,
            )

        self.assertEqual(
            captured_candidates,
            [
                (["s1", "s2"], ["snips_intent_a", "snips_intent_b"], 16, "UNK_INTENT"),
                (["m1"], ["multi_intent"], 16, "UNK_INTENT"),
                (["a1"], ["atis_intent"], 16, "UNK_INTENT"),
            ],
        )
        self.assertEqual(
            result_view["pred_intent"].tolist(),
            ["snips_pred_1", "multi_pred", "snips_pred_2", "atis_pred"],
        )
        self.assertEqual(
            stats["intent_candidate_counts_by_dataset"],
            {"snips": 2, "multiwoz": 1, "atis": 1},
        )


if __name__ == "__main__":
    unittest.main()
