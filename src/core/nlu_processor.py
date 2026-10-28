import re

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .model_loader import ModelLoader
from src.common.text_utils import (
    fuse_special_tokens,
    is_string_not_present,
    remove_duplicates_and_substrings,
    sanitize_edges,
)


class NLUProcessor:
    DEFAULT_INTENT_THRESHOLD = 0.2
    DEFAULT_SLOT_THRESHOLD = 0.5
    INTENT_THRESHOLD = DEFAULT_INTENT_THRESHOLD
    SLOT_THRESHOLD = DEFAULT_SLOT_THRESHOLD

    def __init__(
        self,
        models: ModelLoader,
        tau_intent: float | None = None,
        tau_slots: float | None = None,
    ):
        """
        Recebe uma instancia do ModelLoader para acessar os modelos.
        """
        self.models = models
        self.tau_intent = (
            self.DEFAULT_INTENT_THRESHOLD if tau_intent is None else tau_intent
        )
        self.tau_slots = (
            self.DEFAULT_SLOT_THRESHOLD if tau_slots is None else tau_slots
        )

    def __setattr__(self, name, value):
        if name in {"tau_intent", "INTENT_THRESHOLD"}:
            threshold = float(value)
            object.__setattr__(self, "tau_intent", threshold)
            object.__setattr__(self, "INTENT_THRESHOLD", threshold)
            return
        if name in {"tau_slots", "SLOT_THRESHOLD"}:
            threshold = float(value)
            object.__setattr__(self, "tau_slots", threshold)
            object.__setattr__(self, "SLOT_THRESHOLD", threshold)
            return
        object.__setattr__(self, name, value)

    def intent_recognition(self, sentence, intent_dict, unknown_label: str = "UNK_INTENT"):
        """
        Retorna sempre (label:str, score:float).
        Se nao atingir o threshold, retorna (unknown_label, top_score).
        Em erro ou dados malformados, retorna (unknown_label, 0.0).
        """
        intents_list = list(intent_dict.keys()) if intent_dict else []
        if not intents_list:
            return unknown_label, 0.0

        try:
            results = self.models.classifier(
                sentence,
                intents_list,
                hypothesis_template="This command is intended to {}",
                multi_label=False,
            )

            scores = results.get("scores") if isinstance(results, dict) else None
            labels = results.get("labels") if isinstance(results, dict) else None
            if not scores or not labels:
                return unknown_label, 0.0

            max_index = max(range(len(scores)), key=lambda i: scores[i])
            top_score = float(scores[max_index])
            top_label = str(labels[max_index])

            thr = getattr(self, "tau_intent", getattr(self, "INTENT_THRESHOLD", 0.0))
            if top_score >= thr:
                return top_label, round(top_score, 3)
            return unknown_label, round(top_score, 3)
        except Exception:
            return unknown_label, 0.0

    def get_slot_tokens_new(self, sentence, intent, intent_slot_dict):
        """
        Extrai candidatos a tokens/expressoes para slot filling a partir da sentenca,
        incluindo formas fundidas com hifen/apostrofos.
        """
        doc = self.models.nlp(sentence)

        entities = [ent.text for ent in doc.ents]
        slot_labels = list(intent_slot_dict.get(intent, {}).get("slots", {}).keys())
        pos_desired = ("NOUN", "PROPN", "ADV", "PRON", "VERB")

        nouns_and_tokens = []
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
                    start_sub = subtree_tokens[0].i
                    end_sub = subtree_tokens[-1].i + 1
                    phrase = doc[start_sub:end_sub].text
                    for slot_label in slot_labels:
                        for word in slot_label.split():
                            pattern = re.compile(
                                r"\b" + re.escape(word) + r"\b",
                                re.IGNORECASE,
                            )
                            phrase = pattern.sub("", phrase)
                    phrase = phrase.strip()
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
            else:
                start = i
                while i < n and doc[i].pos_ in ("NOUN", "PROPN"):
                    if doc[i].dep_ != "compound":
                        i += 1
                        break
                    i += 1
                compound_phrase = doc[start:i].text
                for slot_label in slot_labels:
                    for word in slot_label.split():
                        pattern = re.compile(
                            r"\b" + re.escape(word) + r"\b",
                            re.IGNORECASE,
                        )
                        compound_phrase = pattern.sub("", compound_phrase)
                compound_phrase = compound_phrase.strip()
                if compound_phrase:
                    nouns_and_tokens.append(compound_phrase)

        tokens_list = []
        for token_str in nouns_and_tokens:
            if not any(token_str.lower() in ent.lower() for ent in entities):
                tokens_list.append(token_str)

        fused = fuse_special_tokens(doc)
        fused_candidates = []
        for fused_token in fused:
            if "-" in fused_token or "'" in fused_token or "’" in fused_token:
                if any(fused_token.lower() in ent.lower() for ent in entities):
                    continue
                cleaned = fused_token
                for slot_label in slot_labels:
                    for word in slot_label.split():
                        cleaned = re.sub(
                            r"\b" + re.escape(word) + r"\b",
                            "",
                            cleaned,
                            flags=re.IGNORECASE,
                        )
                cleaned = cleaned.strip()
                if cleaned and is_string_not_present(cleaned, intent_slot_dict):
                    fused_candidates.append(cleaned)

        tokens_list.extend(fused_candidates)
        tokens_list = remove_duplicates_and_substrings(tokens_list)
        return tokens_list

    def get_entities(self, sentence):
        doc = self.models.nlp(sentence)
        entities = [ent.text for ent in doc.ents]
        for ent in doc.ents:
            print(f"Texto: {ent.text} | Tipo: {ent.label_}")
        return entities

    def filter_doctitles(self, doc_titles, intent, intent_slot_dict):
        """
        Filtra uma lista de titulos, removendo titulos indesejados e
        eliminando substrings que correspondam aos slot_labels.
        """
        slot_labels = list(intent_slot_dict.get(intent, {}).get("slots", {}).keys())
        pos_discard = {"ADP", "ADV", "AUX", "INTJ", "CONJ", "CCONJ", "SCONJ", "VERB"}

        for i in reversed(range(len(doc_titles))):
            term = doc_titles[i]
            doc = self.models.nlp(term)
            if len(doc) == 1 and doc[0].pos_ in pos_discard:
                doc_titles.pop(i)

        for i, title in enumerate(doc_titles):
            for slot_label in slot_labels:
                for word in slot_label.split():
                    pattern = re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)
                    title = pattern.sub("", title)
            doc_titles[i] = title.strip()

        return doc_titles

    def slot_scores(self, sentence, intent, tokens, intent_slot_dict, validation_slots=None):
        """
        Calcula os scores dos slots para uma sentenca utilizando o classificador.
        """
        if intent in intent_slot_dict:
            if not validation_slots:
                slots = intent_slot_dict[intent]["slots"]
            else:
                slots = {
                    slot: info
                    for slot, info in intent_slot_dict[intent]["slots"].items()
                    if slot in validation_slots
                }
        else:
            slots = {}

        if not slots:
            return slots

        slot_token_scores = {}
        for slot_name, slot_info in slots.items():
            slot_token_scores[slot_name] = []
            output = self.models.classifier(
                sentence,
                tokens,
                hypothesis_template=slot_info["hypothesis"],
                multi_label=True,
            )
            print(f"\nOutput for hypothesis '{slot_info['hypothesis']}':\n{output}\n")
            for label, score in zip(output["labels"], output["scores"]):
                slot_token_scores[slot_name].append(
                    {
                        "token": label,
                        "score": score,
                    }
                )

        data_list = [
            (slot, item["token"], item["score"])
            for slot, items in slot_token_scores.items()
            for item in items
        ]
        df = pd.DataFrame(data_list, columns=["Slot", "Token", "Score"])
        df_score = df.pivot(index="Slot", columns="Token", values="Score")
        print(f"\n{df_score}")
        return df_score

    def slot_associations(self, df_score, threshold: float | None = None):
        """
        Realiza o slot filling a partir do DataFrame de scores.
        """
        threshold = (
            getattr(
                self,
                "tau_slots",
                getattr(self, "SLOT_THRESHOLD", self.DEFAULT_SLOT_THRESHOLD),
            )
            if threshold is None
            else float(threshold)
        )
        huge_cost = 1e6
        scores = df_score.to_numpy()
        cost_matrix = np.where(scores >= threshold, -scores, huge_cost)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assigned_pairs = []
        for i, j in zip(row_ind, col_ind):
            slot_name = df_score.index[i]
            token_name = df_score.columns[j]
            score = df_score.iloc[i, j]
            assigned_pairs.append((slot_name, token_name, score))

        slot_filling = {}
        for slot, token, score in assigned_pairs:
            if score < threshold:
                slot_filling[slot] = ("NULL", 0.0)
            else:
                slot_filling[slot] = (token, round(float(score), 3))

        for slot in df_score.index:
            if slot not in slot_filling:
                slot_filling[slot] = ("NULL", 0.0)

        return slot_filling

    def retrieval_search(self, query):
        return self.models.retrieval_engine.retrieve(query)

    def get_intent_slot_answers_score(self, intent, sentence, intent_slot_dict):
        """
        Executa QA por slot e retorna:
        - unique_answers: lista de respostas unicas, ja sanitizadas nas bordas.
        - slot_best_scores: dict {slot_label: melhor_score_encontrado_no_slot}.
        - slot_answer_spans: dict {slot_label: [span_escolhido_para_este_slot]}.
        """
        slots = intent_slot_dict.get(intent, {}).get("slots", {})
        slot_answers = {}

        for slot_label, slot_info in slots.items():
            question = slot_info["question"]
            results = self.models.qa_model(question=question, context=sentence)
            if isinstance(results, dict):
                results = [results]
            for result in results:
                if (
                    isinstance(result, dict)
                    and "answer" in result
                    and isinstance(result["answer"], str)
                ):
                    result["answer"] = sanitize_edges(result["answer"])
            slot_answers[slot_label] = results

        print("Respostas extraidas por slot:")
        for slot, answers in slot_answers.items():
            print(f"  Slot '{slot}':")
            for answer in answers:
                print(f"    Answer: {answer.get('answer')}, Score: {answer.get('score')}")

        items = []
        for slot_label, answers in slot_answers.items():
            for answer in answers:
                if isinstance(answer, dict):
                    text = answer.get("answer", "")
                    score = float(answer.get("score", 0.0) or 0.0)
                    items.append((slot_label, text, score))

        items_sorted = sorted(items, key=lambda item: item[2], reverse=True)
        print("\nRespostas ordenadas por score:")
        for slot, answer, score in items_sorted:
            print(f"  Slot: {slot}, Answer: '{answer}', Score: {score}")

        final_candidates = []
        for candidate in items_sorted:
            cand_text = (candidate[1] or "").lower()
            if not cand_text:
                continue
            redundant = False
            for final_candidate in final_candidates:
                final_text = (final_candidate[1] or "").lower()
                if cand_text in final_text or final_text in cand_text:
                    print(
                        f"Ignorando redundancia: '{candidate[1]}' "
                        f"(comparado com '{final_candidate[1]}')"
                    )
                    redundant = True
                    break
            if not redundant:
                print(f"Adicionando resposta: '{candidate[1]}'")
                final_candidates.append(candidate)

        unique_answers = []
        seen = set()
        for _, answer, _ in final_candidates:
            key = (answer or "").lower()
            if key and key not in seen:
                unique_answers.append(answer)
                seen.add(key)

        slot_best_scores = {}
        slot_answer_spans = {}
        for slot_label, answers in slot_answers.items():
            best = 0.0
            best_answer = ""
            if answers:
                best_item = max(
                    (answer for answer in answers if isinstance(answer, dict)),
                    key=lambda item: float(item.get("score", 0.0) or 0.0),
                    default=None,
                )
                if best_item:
                    best = float(best_item.get("score", 0.0) or 0.0)
                    best_answer = best_item.get("answer", "") or ""
            slot_best_scores[slot_label] = best
            slot_answer_spans[slot_label] = [best_answer] if best_answer else []

        print("\nRespostas unicas finais:", unique_answers)
        print("Melhor score por slot:", slot_best_scores)
        print("Span top-1 por slot (QA):", slot_answer_spans)

        return unique_answers, slot_best_scores, slot_answer_spans
