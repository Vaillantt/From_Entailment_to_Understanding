from pathlib import Path
import os

from .nlu_processor import NLUProcessor
from .model_loader import ModelLoader
import torch._dynamo

from src.common.text_utils import (
    add_row_scores_masked,
    build_alias_collapsed_grouping_view,
    find_all_matching_substrings,
    process_until_no_more_groups,
    remove_duplicates,
    remove_duplicates_and_substrings,
)

torch._dynamo.config.suppress_errors = True

os.environ["TORCHDYNAMO_DISABLE"] = "1"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX_PATH = REPO_ROOT / "retrieval" / "2025_sparse_wikititle_index"


def initialize_nlu(index_path=None, device=0, tau=None, tau_intent=None, tau_slots=None):
    """
    Inicializa os modelos e retorna uma instancia do NLUProcessor.
    """
    resolved_index_path = Path(
        index_path or os.environ.get("WIKITITLE_INDEX_PATH", DEFAULT_INDEX_PATH)
    )
    resolved_tau_intent = tau if tau_intent is None else tau_intent
    resolved_tau_slots = tau if tau_slots is None else tau_slots
    model_loader = ModelLoader(index_path=str(resolved_index_path), device=device)
    nlu_processor = NLUProcessor(
        model_loader,
        tau_intent=resolved_tau_intent,
        tau_slots=resolved_tau_slots,
    )
    return nlu_processor


def _project_slot_score_columns(df_slot_score, selected_tokens):
    """
    Reaproveita a matriz de scores ja calculada, mantendo apenas as colunas
    presentes em `selected_tokens` e preservando essa ordem.
    """
    kept_tokens = [token for token in selected_tokens if token in df_slot_score.columns]
    return df_slot_score.loc[:, kept_tokens].copy()


def intent_slot_pipeline(
    nlu_processor: NLUProcessor,
    sentence,
    intent_slot_dict,
    only_intent=False,
    force_intent=None,
):
    if not force_intent:
        intent, intent_score = nlu_processor.intent_recognition(sentence, intent_slot_dict)
        print("Intent reconhecido:", intent, "com score:", intent_score)

        if intent == "UNK_INTENT":
            print("[ABORT] Intent desconhecido (UNK_INTENT). Encerrando antes do slot filling.")
            result = {"intent": (intent, intent_score), "slots": {}}
            print("\n[NEW] Resultado final:", result)
            return result

        if only_intent:
            return {"intent": (intent, intent_score), "slots": {}}

    else:
        intent = force_intent
        intent_score = 1.0

    pos_tokens = nlu_processor.get_slot_tokens_new(sentence, intent, intent_slot_dict)
    print("Tokens POS extraidos:", pos_tokens)

    entities = nlu_processor.get_entities(sentence)
    print("Entidades extraidas:", entities)

    retrieval_results = nlu_processor.retrieval_search(sentence)["title"]
    print("Resultados de recuperacao:", retrieval_results)

    matching_titles = find_all_matching_substrings(retrieval_results, sentence)
    matching_titles_filtered = nlu_processor.filter_doctitles(
        matching_titles,
        intent,
        intent_slot_dict,
    )
    print("Titulos recuperados filtrados:", matching_titles_filtered)

    relevant_spans = remove_duplicates_and_substrings(pos_tokens + matching_titles)
    print("Tokens relevantes combinados:", relevant_spans)

    qa_answers, qa_slot_scores, qa_slot_spans = nlu_processor.get_intent_slot_answers_score(
        intent,
        sentence,
        intent_slot_dict,
    )
    print(f"\n[6] Respostas QA por slot: {qa_answers}")
    print(f"[6] Scores QA por slot: {qa_slot_scores}")
    print(f"[6] Spans QA por slot (top-1): {qa_slot_spans}")

    final_spans = remove_duplicates(relevant_spans + entities + qa_answers)
    print("Tokens finais para slot filling:", final_spans)

    df_slot_score = nlu_processor.slot_scores(sentence, intent, final_spans, intent_slot_dict)
    print("DataFrame com scores dos slots:\n", df_slot_score)

    grouping_df_slot_score, alias_group_map = build_alias_collapsed_grouping_view(
        df_slot_score,
        sentence,
    )
    collapsed_alias_groups = {
        rep: members for rep, members in alias_group_map.items() if len(members) > 1
    }
    if collapsed_alias_groups:
        print("\n[NEW] Alias groups colapsados para grouping:", collapsed_alias_groups)
        print("[NEW] DataFrame alias-collapsed para grouping:\n", grouping_df_slot_score)

    slot_threshold = getattr(
        nlu_processor,
        "tau_slots",
        getattr(nlu_processor, "SLOT_THRESHOLD", NLUProcessor.DEFAULT_SLOT_THRESHOLD),
    )
    new_final_slot_tokens = process_until_no_more_groups(
        grouping_df_slot_score,
        threshold=slot_threshold,
    )
    print("\n[NEW] Tokens apos agrupamento iterativo:", new_final_slot_tokens)

    if not new_final_slot_tokens:
        print("[WARN] Nenhum token candidato para slot filling apos filtros/agrupamento. Retornando slots vazios.")
        return {"intent": (intent, intent_score), "slots": {}}

    final_df_slot_score = _project_slot_score_columns(
        grouping_df_slot_score,
        new_final_slot_tokens,
    )
    print("\n[NEW] DataFrame final de scores (pos-agrupamento):\n", final_df_slot_score)

    final_df_slot_score = add_row_scores_masked(
        final_df_slot_score,
        qa_slot_scores,
        qa_slot_spans,
    )
    print("\n[11.NEW] DF apos soma dos QA slot scores por linha:\n", final_df_slot_score)

    print("\n\n NORMAL HUNGARIAN \n\n ###################################\n\n")
    slots = nlu_processor.slot_associations(final_df_slot_score)

    result = {"intent": (intent, intent_score), "slots": slots}
    print("\n[NEW] Resultado final:", result)
    return result
