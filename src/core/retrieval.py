#!/usr/bin/env python
"""
retrieval_engine.py

Este módulo contém funções e uma classe para carregar os índices e modelos de recuperação
uma única vez e, posteriormente, realizar retrieval para diferentes queries sem
necessidade de recarregar os recursos a cada chamada.
"""

import os
import time
import pandas as pd
import pyterrier as pt
import string
import re


def init_pyterrier(mem: int = 10000):
    """
    Inicializa o PyTerrier se ainda não estiver iniciado.
    
    Args:
        mem (int): Memória alocada para o PyTerrier (em MB).
    """
    if not pt.started():
        pt.init(mem=mem)
        pt.logging("DEBUG")

def load_index(index_path: str):
    """
    Carrega o índice do PyTerrier a partir do diretório informado.
    
    Args:
        index_path (str): Caminho para o diretório do índice.
    
    Returns:
        pt.Index: Objeto do índice carregado.
    
    Raises:
        FileNotFoundError: Se o diretório do índice não existir ou estiver vazio.
    """
    props = os.path.join(index_path, "data.properties")
    if not os.path.exists(index_path) or not os.listdir(index_path):
        raise FileNotFoundError(f"Índice não encontrado ou vazio: {index_path}")
    index = pt.IndexFactory.of(props)
    print("Índice carregado:")
    print(index.getCollectionStatistics().toString())
    return index

def load_retriever(index_path: str, top_k: int = 10, model: str = 'PL2'):
    """
    Carrega o retriever a partir do diretório do índice, definindo o número de resultados e o modelo.
    
    Args:
        index_path (str): Caminho para o diretório do índice.
        top_k (int): Número de resultados a retornar.
        model (str): Modelo de recuperação (por exemplo, 'PL2').
    
    Returns:
        pt.terrier.Retriever: Objeto de recuperação carregado.
    
    Raises:
        FileNotFoundError: Se o índice não for encontrado ou estiver vazio.
    """
    init_pyterrier()
    props = os.path.join(index_path, "data.properties")

    if not os.path.exists(props ):
        raise FileNotFoundError(f"data.properties não encontrado em: {props}")
    
    retriever = pt.terrier.Retriever(
        props, 
        num_results=top_k, 
        wmodel=model, 
        metadata=["docno", "title", "categories"]
    )
    print("Retriever carregado.")
    return retriever

def retrieve_query(retriever, query: str) -> pd.DataFrame:
    """
    Executa o retrieval para a query fornecida utilizando o retriever previamente carregado.
    
    Args:
        retriever: Objeto de recuperação previamente carregado.
        query (str): Consulta a ser realizada.
    
    Returns:
        pd.DataFrame: DataFrame com os resultados da recuperação.
    """
    # Converte a query em DataFrame no formato esperado
    query = query.translate(str.maketrans('', '', string.punctuation))
    query_df = pd.DataFrame({'qid': [1], 'query': [query]})
    start_time = time.time()
    results = retriever.transform(query_df)
    elapsed_time = time.time() - start_time
    print(f"Recuperação concluída em {elapsed_time:.2f} segundos.")
    return results

def find_substrings(lista, string_maior):
    """
    Encontra substrings da lista presentes em uma string maior sem redundâncias.
    
    Args:
        lista (list): Lista de strings.
        string_maior (str): String na qual as substrings serão procuradas.
    
    Returns:
        list: Lista de substrings encontradas sem redundância.
    """
    # Normaliza a string maior: remove espaços extras e converte para minúsculas
    string_maior_normalizada = " ".join(string_maior.lower().split())
    
    # Encontra os itens da lista que estão presentes na string maior
    encontrados = []
    for item in lista:
        item_normalizado = " ".join(item.lower().split())
        if item_normalizado in string_maior_normalizada:
            encontrados.append(item)
    
    # Ordena em ordem decrescente de tamanho para priorizar strings maiores
    encontrados.sort(key=len, reverse=True)
    
    # Remove redundâncias: se um item for substring de outro já adicionado, ele é descartado.
    resultado = []
    for item in encontrados:
        item_normalizado = " ".join(item.lower().split())
        if any(item_normalizado in " ".join(r.lower().split()) for r in resultado):
            continue
        resultado.append(item)
    
    return resultado


def find_all_matching_substrings(lista, string_maior):
    """
    Encontra todas as ocorrências na sentença que correspondem a algum dos títulos da lista,
    permitindo pequenas variações (como letras extras no final) e retornando as ocorrências
    exatamente como aparecem na sentença (preservando acentos, case, etc.).
    
    Em seguida, remove duplicatas e elimina redundâncias, ou seja, se uma ocorrência estiver
    contida em outra maior, a menor é descartada.
    
    Args:
        lista (list): Lista de títulos (strings) vindos do retrieval.
        string_maior (str): Sentença na qual os títulos serão procurados.
        
    Returns:
        list: Lista com todas as ocorrências únicas e não redundantes.
    """
    matches = []
    
    # Para cada título, constrói um padrão regex que permite extensões nas palavras (ex.: 'completo' bate com 'completosos')
    for titulo in lista:
        titulo_limpo = " ".join(titulo.split())
        pattern = r'\b' + r'\s+'.join([re.escape(word) + r'\w*' for word in titulo_limpo.split()]) + r'\b'
        # Coleta todas as ocorrências na sentença (busca sem case-sensitive, mas retorna o trecho literal)
        for match in re.finditer(pattern, string_maior, flags=re.IGNORECASE):
            matches.append(match.group(0))
    
    # Remove duplicatas (comparação feita em lower-case)
    unique_matches = []
    seen = set()
    for m in matches:
        key = m.lower()
        if key not in seen:
            seen.add(key)
            unique_matches.append(m)
    
    # Remove redundâncias: se uma ocorrência estiver contida em outra já selecionada, descarta a menor.
    unique_matches.sort(key=len, reverse=True)  # Ordena do maior para o menor
    final_matches = []
    for m in unique_matches:
        m_lower = m.lower()
        # Se m estiver contido em alguma ocorrência já adicionada (e não for idêntico), ignora
        if not any(m_lower in other.lower() and m_lower != other.lower() for other in final_matches):
            final_matches.append(m)
    
    return final_matches

class RetrievalEngine:
    """
    Classe para encapsular o motor de recuperação.
    O índice e o retriever são carregados uma única vez e podem ser reutilizados para diversas queries.
    """
    def __init__(self, index_path: str, top_k: int = 10, model: str = 'PL2'):
        """
        Inicializa a engine carregando o índice e o retriever.
        
        Args:
            index_path (str): Caminho para o diretório do índice.
            top_k (int): Número de resultados a retornar.
            model (str): Modelo de recuperação (por exemplo, 'PL2').
        """
        self.index_path = index_path
        self.top_k = top_k
        self.model = model
        
        # Carrega o índice e o retriever uma única vez
        self.index = load_index(index_path)
        self.retriever = load_retriever(index_path, top_k, model)
    
    def retrieve(self, query: str) -> pd.DataFrame:
        """
        Executa a recuperação para a query utilizando o retriever carregado.
        
        Args:
            query (str): Consulta para recuperação.
        
        Returns:
            pd.DataFrame: DataFrame com os resultados da recuperação.
        """
        return retrieve_query(self.retriever, query)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Script para realizar retrieval utilizando PyTerrier."
    )
    parser.add_argument("--query", type=str, required=True, help="Consulta para recuperação.")
    parser.add_argument("--index_path", type=str, required=True, help="Caminho do índice.")
    parser.add_argument("--top_k", type=int, default=10, help="Número de resultados a retornar.")
    parser.add_argument("--model", type=str, default="PL2", help="Modelo de recuperação (ex: 'PL2').")
    
    args = parser.parse_args()
    
    # Carrega os recursos apenas uma vez
    engine = RetrievalEngine(index_path=args.index_path, top_k=args.top_k, model=args.model)
    
    # Realiza o retrieval para a query fornecida
    results = engine.retrieve(args.query)
    print("Resultados da recuperação:")
    print(results)
    
    # Exemplo de uso da função auxiliar para extrair substrings sem redundância
    if 'title' in results.columns:
        titles = results['title'].tolist()
        substrings = find_substrings(titles, args.query)
        print("Substrings encontradas sem redundância:")
        print(substrings)
