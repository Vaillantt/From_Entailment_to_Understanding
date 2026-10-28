import json
import re
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ARQUIVO = REPO_ROOT / "archive" / "datasets" / "field_operations" / "generated_dataset" / "confirm_movement_milestone_300_FINAL.json"
INTENT_KEY = "Confirm Movement Milestone"

# Se True: ignora pontuação na comparação (mais agressivo)
IGNORAR_PONTUACAO = False

def reconstruir_utterance(exemplo: dict) -> str:
    return "".join(seg["text"] for seg in exemplo["data"]).strip()

def normalizar(texto: str) -> str:
    texto = " ".join(texto.split()).strip().lower()
    if IGNORAR_PONTUACAO:
        # remove pontuação simples mantendo letras/números/espaços
        texto = re.sub(r"[^\w\s]", "", texto)
        texto = " ".join(texto.split()).strip()
    return texto

def main():
    data = json.loads(ARQUIVO.read_text(encoding="utf-8"))
    exemplos = data[INTENT_KEY]

    mapa = defaultdict(list)  # norm_utt -> [indices...]
    originais = {}            # norm_utt -> uma versão original (pra print)

    for i, ex in enumerate(exemplos):
        utt = reconstruir_utterance(ex)
        key = normalizar(utt)
        mapa[key].append(i)
        # guarda um original representativo
        originais.setdefault(key, utt)

    duplicatas = {k: idxs for k, idxs in mapa.items() if len(idxs) > 1}

    print(f"Total de exemplos: {len(exemplos)}")
    print(f"Total de sentenças únicas (normalizadas): {len(mapa)}")
    print(f"Total de grupos duplicados: {len(duplicatas)}\n")

    # Ordena por “tamanho do grupo” desc, depois pelo menor índice
    grupos_ordenados = sorted(
        duplicatas.items(),
        key=lambda kv: (-len(kv[1]), min(kv[1]))
    )

    for key, idxs in grupos_ordenados:
        print("=" * 80)
        print(f"Sentença: {originais[key]}")
        print(f"Índices: {idxs}")
        # Se quiser ver TODAS as versões originais (caso IGNORAR_PONTUACAO=True),
        # descomente o bloco abaixo:
        # print("Variantes originais:")
        # for j in idxs:
        #     print(f"  [{j}] {reconstruir_utterance(exemplos[j])}")

    # Lista “pares” explícitos (opcional)
    # print("\nPares duplicados:")
    # for key, idxs in grupos_ordenados:
    #     for a in range(len(idxs)):
    #         for b in range(a+1, len(idxs)):
    #             print(f"{idxs[a]} <-> {idxs[b]} :: {originais[key]}")

if __name__ == "__main__":
    main()
