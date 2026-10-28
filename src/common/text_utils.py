# utils.py
import re
import unicodedata


def is_string_not_present(string, intent_slot_dict):
    """
    Retorna True se a string nao for encontrada como substring em nenhum nome de
    intent ou slot.
    """
    string_lower = string.lower()
    for intent, data in intent_slot_dict.items():
        if string_lower in intent.lower():
            return False
        slots = data.get("slots", {})
        for slot_name in slots.keys():
            if string_lower in slot_name.lower():
                return False
    return True


def remove_duplicates(strings):
    """
    Remove duplicatas de uma lista de strings, comparando em lower case e
    mantendo a ordem.
    """
    unique = []
    seen = set()

    for item in strings:
        cleaned = sanitize_edges(item)
        key = cleaned.lower()
        if key not in seen:
            unique.append(cleaned)
            seen.add(key)

    return unique


def sanitize_edges(s: str) -> str:
    """
    Remove apenas espacos ou pontuacoes no inicio/fim da string.
    """
    punctuation = set(".,!?;:…—–-'\"“”‘’()[]{}<>")
    text = "" if s is None else str(s)
    n = len(text)
    i, j = 0, n - 1

    while i < n and (text[i].isspace() or text[i] in punctuation):
        i += 1
    while j >= i and (text[j].isspace() or text[j] in punctuation):
        j -= 1

    return text[i : j + 1]


def remove_duplicates_and_substrings(strings):
    """
    Remove duplicatas e tambem strings que sao substrings de outras strings na lista.
    """
    unique_strings = remove_duplicates(strings)

    filtered = []
    for item in unique_strings:
        if not any(
            item.lower() in other.lower() and item.lower() != other.lower()
            for other in unique_strings
        ):
            filtered.append(item)
    return filtered


def normalize_text(text):
    words = re.findall(r"\w+", text.lower())
    return words


def find_substrings(lista, string_maior):
    """
    Encontra substrings da lista presentes em uma string maior sem redundancias.
    """
    string_maior_normalizada = " ".join(string_maior.lower().split())

    encontrados = []
    for item in lista:
        item_normalizado = " ".join(item.lower().split())
        if item_normalizado in string_maior_normalizada:
            encontrados.append(item)

    encontrados.sort(key=len, reverse=True)

    resultado = []
    for item in encontrados:
        item_normalizado = " ".join(item.lower().split())
        if any(item_normalizado in " ".join(result.lower().split()) for result in resultado):
            continue
        resultado.append(item)

    return resultado


def find_all_matching_substrings(lista, string_maior):
    """
    Encontra ocorrencias na sentenca que correspondem a algum dos titulos da lista.
    """
    matches = []

    for titulo in lista:
        titulo_limpo = " ".join(titulo.split())
        pattern = r"\b" + r"\s+".join(
            [re.escape(word) + r"\w*" for word in titulo_limpo.split()]
        ) + r"\b"
        for match in re.finditer(pattern, string_maior, flags=re.IGNORECASE):
            matches.append(match.group(0))

    unique_matches = []
    seen = set()
    for match in matches:
        key = match.lower()
        if key not in seen:
            seen.add(key)
            unique_matches.append(match)

    unique_matches.sort(key=len, reverse=True)
    final_matches = []
    for match in unique_matches:
        match_lower = match.lower()
        if not any(
            match_lower in other.lower() and match_lower != other.lower()
            for other in final_matches
        ):
            final_matches.append(match)

    return final_matches


def fuse_special_tokens(doc):
    """
    Retorna uma lista de strings com tokens fundidos para casos com hifen e apostrofos.
    """
    symbols_apos = {"'", "’"}
    fused = []
    i = 0
    n = len(doc)

    while i < n:
        token = doc[i]

        if token.pos_ in {"PART", "PUNCT"} and token.text == "-" and fused and i < n - 1:
            fused[-1] = fused[-1] + "-" + doc[i + 1].text
            i += 2
            continue

        if token.pos_ in {"PART", "PUNCT"} and token.text in {"'s", "’s"} and fused:
            fused[-1] = fused[-1] + token.text
            i += 1
            continue

        if token.pos_ in {"PART", "PUNCT"} and token.text in symbols_apos and fused and i < n - 1:
            fused[-1] = fused[-1] + token.text + doc[i + 1].text
            i += 2
            while (
                i < n - 1
                and doc[i].pos_ in {"PART", "PUNCT"}
                and doc[i].text in symbols_apos
            ):
                fused[-1] = fused[-1] + doc[i].text + doc[i + 1].text
                i += 2
            continue

        fused.append(token.text)
        i += 1

    return fused


def _strip_accents(text: str) -> str:
    clean_text = "" if text is None else str(text)
    return "".join(
        ch for ch in unicodedata.normalize("NFD", clean_text)
        if unicodedata.category(ch) != "Mn"
    )


def _normalize_alias_grouping_word(word: str) -> str:
    """
    Normaliza uma palavra apenas para detectar equivalencia de aliases no grouping.
    """
    normalized = _strip_accents(word).lower()
    normalized = normalized.strip(".,!?;:()[]{}<>\"'`“”‘’")
    normalized = re.sub(r"(?:'s|’s)$", "", normalized)
    normalized = normalized.strip(".,!?;:()[]{}<>\"'`“”‘’")
    return normalized


def _normalize_text_for_alias_grouping(text: str):
    """
    Retorna uma tupla de palavras normalizadas para detectar aliases equivalentes.
    """
    tokens = ("" if text is None else str(text)).split()
    normalized = []
    for token in tokens:
        cleaned = _normalize_alias_grouping_word(token)
        if cleaned:
            normalized.append(cleaned)
    return tuple(normalized)


def _normalize_surface_tokens_for_alignment(text: str):
    """
    Tokenizacao compativel com a avaliacao do notebook.
    """
    normalized = _strip_accents("" if text is None else str(text)).lower()
    return re.findall(r"\S+", normalized)


def _surface_aligns_to_sentence(surface: str, sentence: str) -> bool:
    """
    Retorna True se 'surface' aparece como uma janela contigua de tokens na sentenca.
    """
    surface_tokens = _normalize_surface_tokens_for_alignment(surface)
    if not surface_tokens:
        return False

    sentence_tokens = _normalize_surface_tokens_for_alignment(sentence)
    window = len(surface_tokens)
    for i in range(0, len(sentence_tokens) - window + 1):
        if sentence_tokens[i : i + window] == surface_tokens:
            return True
    return False


def build_alias_collapsed_grouping_view(df_slot_score, sentence: str):
    """
    Construi uma visao temporaria do DataFrame para o grouping, colapsando colunas
    que representam o mesmo alias e preservando uma surface original como representante.
    """
    if df_slot_score is None or getattr(df_slot_score, "empty", True):
        return df_slot_score, {}

    try:
        import pandas as pd
    except Exception:
        fallback_map = {col: [col] for col in getattr(df_slot_score, "columns", [])}
        return df_slot_score, fallback_map

    df = df_slot_score.copy()
    column_order = {col: idx for idx, col in enumerate(df.columns)}
    alias_groups = {}

    for col in df.columns:
        alias_key = _normalize_text_for_alias_grouping(col)
        alias_groups.setdefault(alias_key, []).append(col)

    collapsed_columns = []
    representative_to_members = {}

    for members in alias_groups.values():
        if len(members) == 1:
            representative = members[0]
        else:
            member_scores = {col: float(df[col].sum()) for col in members}
            aligned_members = [
                col for col in members if _surface_aligns_to_sentence(col, sentence)
            ]
            candidate_pool = aligned_members if aligned_members else members
            representative = sorted(
                candidate_pool,
                key=lambda col: (-member_scores[col], column_order[col]),
            )[0]

        representative_to_members[representative] = list(members)
        collapsed_series = df[members].max(axis=1).rename(representative)
        collapsed_columns.append(collapsed_series)

    collapsed_df = pd.concat(collapsed_columns, axis=1)
    return collapsed_df, representative_to_members


def normalize_text_span(text: str):
    """
    Split simples (case-sensitive), sem remover pontuacao.
    """
    return ("" if text is None else str(text)).split()


def share_word(token_a: str, token_b: str) -> bool:
    """
    Retorna True se token_a e token_b compartilham ao menos uma palavra.
    """
    words_a = set(normalize_text_span(token_a))
    words_b = set(normalize_text_span(token_b))
    return len(words_a & words_b) > 0


def can_form_groups(items):
    """
    Retorna True se existir algum grupo com tamanho >= 2.
    """
    items_sorted = sorted(items, key=lambda item: item[1], reverse=True)
    grupos = []
    for span, score in items_sorted:
        colocado = False
        for grupo in grupos:
            if all(share_word(span, grouped[0]) for grouped in grupo):
                grupo.append((span, score))
                colocado = True
                break
        if not colocado:
            grupos.append([(span, score)])
    return any(len(grupo) >= 2 for grupo in grupos)


def form_groups(items):
    """
    Retorna lista de grupos seguindo a mesma regra de intersecao total.
    """
    items_sorted = sorted(items, key=lambda item: item[1], reverse=True)
    grupos = []
    for span, score in items_sorted:
        colocado = False
        for grupo in grupos:
            if all(share_word(span, grouped[0]) for grouped in grupo):
                grupo.append((span, score))
                colocado = True
                break
        if not colocado:
            grupos.append([(span, score)])
    return grupos


def process_until_no_more_groups(df, threshold=0.2, fallback_top_k=5):
    """
    Filtra colunas, forma grupos por intersecao de palavras e mantem o melhor span
    de cada grupo ate nao haver mais agrupamentos.

    Quando ``threshold`` e None, preserva todas as colunas antes do agrupamento.
    """
    if df is None or getattr(df, "empty", True):
        return []

    df = df.copy()
    df = df.fillna(0.0)

    col_max = df.max(axis=0)
    if threshold is None:
        keep_cols = list(df.columns)
    else:
        keep_cols = [col for col in df.columns if col_max.get(col, 0.0) > threshold]

        if not keep_cols:
            if fallback_top_k and len(df.columns) > 0:
                keep_cols = (
                    col_max.sort_values(ascending=False)
                    .head(fallback_top_k)
                    .index.tolist()
                )
            else:
                return []

    df_filtered = df[keep_cols]

    items = []
    for span in df_filtered.columns:
        total_score = float(df_filtered[span].sum())
        items.append((span, total_score))

    while True:
        if not can_form_groups(items):
            break
        grupos = form_groups(items)
        final_spans = []
        for grupo in grupos:
            best_tuple = max(grupo, key=lambda item: item[1])
            final_spans.append(best_tuple)
        items = final_spans

    return [span for span, _ in items]


def normalize_text_span_lower(text: str):
    """
    Split simples, case-insensitive, sem remover pontuacao interna.
    """
    clean_text = "" if text is None else str(text)
    return clean_text.lower().split()


def share_word_ci(a: str, b: str) -> bool:
    """
    Compartilha ao menos uma palavra (case-insensitive) entre a e b.
    """
    wa = set(normalize_text_span_lower(a))
    wb = set(normalize_text_span_lower(b))
    return len(wa & wb) > 0


def add_row_scores_masked(df, row_scores, row_spans_map):
    """
    Soma vies apenas nas celulas (s, t) em que o token da coluna t compartilha
    ao menos uma palavra com algum span de resposta do slot s.
    """
    if df is None or getattr(df, "empty", True):
        return df
    try:
        import pandas as pd
    except Exception:
        return df

    bias = pd.DataFrame(0.0, index=df.index, columns=df.columns)

    for slot_name, qscore in (row_scores or {}).items():
        if slot_name not in df.index or qscore is None:
            continue
        spans = (row_spans_map or {}).get(slot_name, []) or []
        if not spans:
            continue
        for token in df.columns:
            if any(share_word_ci(token, span) for span in spans):
                bias.at[slot_name, token] += float(qscore)

    return df + bias
