

import pandas as pd
import numpy as np
import nltk

#Biblioteca de PLN mais moderna e rápida
import spacy

#Divide seus dados em conjunto de treino
from sklearn.model_selection import train_test_split

#algoritmo de classificação que cria muitas árvores de decisão e combina os votos delas.
from sklearn.ensemble import RandomForestClassifier

#algoritmo de classificação que busca a "fronteira" que melhor separa as classes.
from sklearn.svm import SVC

#algoritmo de classificação regressão logistica
from sklearn.linear_model import LogisticRegression

#algoritmo de classificação 
from sklearn.naive_bayes import MultinomialNB

#Gera um relatório de avaliação com precisão, recall, F1-score e acurácia para cada classe.
from sklearn.metrics import classification_report

#Salva e carrega o modelo treinado em arquivo
import joblib

# trabalha com expressões regulares(REGEX): uma linguagem de padrões para procurar, extrair e substituir trechos de texto.
import re

# normalização de caracteres
import unicodedata

# tokenização
from nltk.tokenize import word_tokenize


# Downloads essenciais para viabilizar as ferramentas de NLP do NLTK
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')

# Carregamento do modelo de português do spaCy (exige instalação prévia: python -m spacy download pt_core_news_sm)
nlp = spacy.load("pt_core_news_sm")



# ============================== carregar e dividir dados ==================================================================



#Funções para leitura dos dados 
from IPython.display import display
from pathlib import Path
import os


def carregar_dados(caminho_base_dataset):
    """
    Lê os arquivos .txt locais, estrutura em um DataFrame e divide em Treino, Validação e Teste.
    """
    dados = []
    erros = []  
    
    # 1. Mapeamento das categorias exigidas pelo projeto
    categorias = [
        "1_Ensino_Fundamental_I", 
        "2_Ensino_Fundamental_II", 
        "3_Ensino_Medio", 
        "4_Ensino_Superior"
    ]
    
    # 2. Leitura e Estruturação dos Dados Locais
    for categoria in categorias:
        caminho_pasta = os.path.join(caminho_base_dataset, categoria) #caminho_pasta = dataset/Ensino_Fundamental_I
        
        # Verifica se a pasta existe antes de iterar
        if os.path.exists(caminho_pasta):
            for nome_arquivo in os.listdir(caminho_pasta):
                if nome_arquivo.endswith('.txt'):
                    caminho_arquivo = os.path.join(caminho_pasta, nome_arquivo) #caminho arquivo = dataset/Ensino_Fundamental_I/arquivo1.txt
                    
                    try:
                        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
                            texto = f.read().strip()
                    except UnicodeDecodeError:
                        erros.append(nome_arquivo)
                        continue

                    if texto:
                        dados.append({"texto": texto, "complexidade": categoria})
    

    # aviso de arquivos ignorados por erro de encoding
    if erros:
        print(f"Atenção: {len(erros)} arquivo(s) com erro de encoding ignorados.")


    # Converte a lista de dicionários para um DataFrame do Pandas
    df = pd.DataFrame(dados)
    
    # aviso de dados vazios
    if df.empty:
        raise ValueError("Nenhum dado foi carregado. Verifique os caminhos dos arquivos.")

    # tratatando duplicados no df
    n_dup = df.duplicated(subset="texto").sum()
    if n_dup:
        print(f"Atenção: {n_dup} texto(s) duplicado(s). Removendo para evitar vazamento.")
        df = df.drop_duplicates(subset="texto")

    return df.reset_index(drop=True)


def separar_dados(df):

    # 3. Divisão dos Dados (Treino, Validação e Teste) --> ordem: (maior,menor)
    # Primeiro, separamos o Teste (ex: 15% dos dados) do resto (Treino + Validação)
    # O parâmetro stratify garante que a proporção das 4 categorias se mantenha igual
    df_resto, df_teste = train_test_split(
        df, 
        test_size=0.15, 
        stratify=df['complexidade'], 
        random_state=42                # fixa a semente do sorteio, então a divisão é sempre a mesma toda vez que você rodar. O número 42 é só uma convenção, qualquer inteiro serve.
    )
    
    # Agora, dividimos o "resto" em Treino (ex: 70% do total) e Validação (ex: 15% do total)
    # Como tiramos 15% antes, restaram 85%. Para tirar ~15% do total a partir dos 85%, usamos test_size=0.176 (0.15/0.85)
    df_treino, df_validacao = train_test_split(
        df_resto, 
        test_size=0.1764, 
        stratify=df_resto['complexidade'], 
        random_state=42
    )
    
    print(f"Dados carregados com sucesso!")
    print(f"Total: {len(df)} instâncias.")
    print(f"Treino: {len(df_treino)} | Validação: {len(df_validacao)} | Teste: {len(df_teste)}")
    
    
    return (d.reset_index(drop=True) for d in (df_treino, df_validacao, df_teste))



# Caminho da pasta principal
caminho_base_dataset = Path(r"c:\Users\pedro\Downloads\Prova de seleção\dataset_caed")

df = carregar_dados(caminho_base_dataset)
df_treino, df_validacao, df_teste = separar_dados(df)

#display(df_treino, df_validacao, df_teste)



# ============================ criando feature de palavras canonicas ===============================================================



# Conjuntos de busca otimizada O(1), criados uma única vez 
VOGAIS = set("aeiouáéíóúâêôãõà")
CONSOANTES = set("bcçdfghjklmnpqrstvwxyz")  

# A sílaba canônica é CV em termos de som, digrafos representam um único som consonantal
# cria e guarda um padrão pronto para uso transforma o texto do padrão em um objeto "compilado"
DIGRAFOS = re.compile(r"ch|lh|nh|rr|ss|qu(?=[eéêií])|gu(?=[eéêií])")


def calcular_canonicidade(texto):
    """
    Recebe um texto bruto, tokeniza, avalia a estrutura CV de cada palavra
    e retorna a proporção de palavras canônicas.
    """
    # padroniza a forma como os caracteres acentuados são armazenados.
    # junta tudo na forma composta, deixando cada letra acentuada como um único caractere.
    texto = unicodedata.normalize("NFC", texto.lower())

    tokens = word_tokenize(texto, language='portuguese')

    palavras = [p for p in tokens if p.isalpha()]

    # Prevenção de divisão por zero caso o texto só contenha símbolos
    if not palavras:
        return 0.0

    total_canonicas = 0

    # 2. Avaliação estrutural de cada palavra isolada
    for palavra in palavras:
      
        # procura todas as ocorrências do padrão em palavra e troca cada uma por "b"
        # "b" é só um marcador o objetivo é que o dígrafo (dois caracteres) passe a ocupar a posição de uma consoante só
        palavra = DIGRAFOS.sub("b", palavra)

        # Regra 1: Palavras com formato estrito CV possuem tamanho par
        if len(palavra) % 2 != 0:
            continue

        e_canonica = True

        # Regra 2 e 3: Verificação alternada de Consoante e Vogal
        for i, letra in enumerate(palavra):
            # Se o índice é par (0, 2, 4...), a letra TEM que ser consoante
            if i % 2 == 0 and letra not in CONSOANTES:
                e_canonica = False
                break
            # Se o índice é ímpar (1, 3, 5...), a letra TEM que ser vogal
            elif i % 2 != 0 and letra not in VOGAIS:
                e_canonica = False
                break

        if e_canonica:
            total_canonicas += 1

    # 3. Retorna a proporção de palavras canônicas (Feature para o modelo ML)
    return total_canonicas / len(palavras)




# ================== Criando Features =========================================================================================



# utilizando biblioteca de NLP spaCY pois aqui ela facilita o uso do pos_taggin para classificar gramaticalmente 
nlp = spacy.load("pt_core_news_sm", disable=["ner"])  


# ---------- Constantes (criadas uma única vez) ----------

# Passo 1: padrões de pontuação
PADROES_PONTUACAO = {
    "virgula":       re.compile(r","),
    "ponto_final":   re.compile(r"(?<!\.)\.(?!\.)"),   # ignora pontos de reticências
    "reticencias":   re.compile(r"\.{3}|…"),
    "interrogacao":  re.compile(r"\?"),
    "exclamacao":    re.compile(r"!"),
    "ponto_virgula": re.compile(r";"),
    "dois_pontos":   re.compile(r":"),
    "aspas":         re.compile(r"[\"“”«»]"),
    "parenteses":    re.compile(r"[()]"),
    "travessao":     re.compile(r"[—–]"),
}

# Passo 3: classes gramaticais (etiquetas universais do spaCy)
CLASSES_POS = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP",
               "CCONJ", "SCONJ", "AUX", "NUM", "PROPN", "INTJ"]

# Nomes de todas as features, em ordem fixa
NOMES_FEATURES = (
    [f"punct_{nome}" for nome in PADROES_PONTUACAO]
    + ["sent_tam_medio", "sent_tam_desvio"]
    + [f"pos_{tag}" for tag in CLASSES_POS]
    + ["lex_tam_medio_palavra", "lex_desvio_tam_palavra",
       "lex_prop_palavras_longas", "lex_ttr_300"]
    + ["canonicidade"]
)


def extrair_features_nlp(texto):

    # ---------------- PASSO 0 ----------------
    # nlp(texto) roda o pipeline do spaCy (tokenizador, etiquetador gramatical,
    # parser de sentenças) e devolve um objeto "doc": o texto já analisado.
    # Fazemos isso UMA vez e reaproveitamos o doc em todos os passos.
    doc = nlp(texto)

    # "for t in doc" percorre todos os tokens (palavras E pontuação).
    # "if t.is_alpha" mantém só os que são formados por letras, descartando
    # vírgulas, pontos, números etc. O resultado é uma lista de tokens-palavra.
    palavras = [t for t in doc if t.is_alpha]

    # Quantidade de palavras: será o denominador de todas as proporções.
    n_palavras = len(palavras)

    # Texto sem nenhuma palavra: devolve o vetor zerado, com as mesmas chaves,
    # para não quebrar o DataFrame depois.
    if n_palavras == 0:
        return {nome: 0.0 for nome in NOMES_FEATURES}

    features = {}   # dicionário que vai guardar "nome da feature -> valor"


    # ================== Features Estruturais =========================================


    # ---------------- Feature de pontução  ----------------

    # Conta a pontuação com regex no texto cru, dividindo por n_palavras.
    for nome, padrao in PADROES_PONTUACAO.items():
        features[f"punct_{nome}"] = len(padrao.findall(texto)) / n_palavras


    # ---------------- Feature de tamanho medio sentenças (spaCy) ----------------

    # doc.sents entrega as sentenças que o spaCy identificou.
    # Para cada sentença ("sent"), percorremos seus tokens ("for t in sent")
    # e somamos 1 para cada token que é palavra (t.is_alpha).
    # Resultado: uma lista com o nº de palavras de cada sentença, ex.: [4, 6, 3].
    tamanhos = [sum(1 for t in sent if t.is_alpha) for sent in doc.sents]

    # Descarta sentenças sem palavras (ex.: uma sentença que é só "...").
    tamanhos = [n for n in tamanhos if n > 0]

    # Média e desvio-padrão do tamanho das sentenças.
    features["sent_tam_medio"] = float(np.mean(tamanhos))
    features["sent_tam_desvio"] = float(np.std(tamanhos))


    # ----------------- Feature de classes gramaticais (spaCy) ----------------

    # Cria um "placar" com todas as classes da nossa lista começando em 0:
    # {"NOUN": 0, "VERB": 0, "ADJ": 0, ...}
    # Fixar as classes garante que TODO texto gere as mesmas colunas.
    contagem_pos = {tag: 0 for tag in CLASSES_POS}

    # Percorre cada palavra do texto, uma de cada vez.
    for token in palavras:
        # token.pos_ é a classe gramatical que o spaCy atribuiu (ex.: "NOUN").
        # Só contamos se ela estiver na nossa lista; classes fora dela
        # (PART, SYM, X) são ignoradas, o que evita erro de chave inexistente.
        if token.pos_ in contagem_pos:
            # Soma 1 ao placar daquela classe.
            contagem_pos[token.pos_] += 1

    # Converte o placar em proporções e guarda no dicionário de features.
    # contagem_pos.items() devolve pares (chave, valor), ex.: ("NOUN", 5).
    # O "for tag, contagem in ..." desempacota cada par em duas variáveis.
    for tag, contagem in contagem_pos.items():
        features[f"pos_{tag}"] = contagem / n_palavras


    # ================== Features Lexicais ===============================================

    # padronizando minusculas
    formas = [t.text.lower() for t in palavras]
    # Tamanho das palavras 
    tamanhos_pal = [len(f) for f in formas]


    # --------- Feature Comprimento médio e variação do comprimento das palavras ----------------
    features["lex_tam_medio_palavra"] = float(np.mean(tamanhos_pal))
    features["lex_desvio_tam_palavra"] = float(np.std(tamanhos_pal))


    # -------- Feature Proporção de palavras longas (mais de 6 letras) -----------------------
    features["lex_prop_palavras_longas"] = sum(1 for n in tamanhos_pal if n > 6) / n_palavras


    # ---------- Feature TTR nas primeiras 300 palavras: palavras distintas / palavras da janela ---------
    janela = formas[:300]
    features["lex_ttr_300"] = len(set(janela)) / len(janela)


    # ---------------- Feature de canonicidade (NLTK, via função própria) ----------------
    features["canonicidade"] = calcular_canonicidade(texto)


    return features



# ======================= Adicionar Features ao DFs ===========================================================



def adicionar_features(df):
    """Aplica extrair_features_nlp a cada texto e anexa as colunas ao DataFrame."""
    df = df.drop(columns=[c for c in NOMES_FEATURES if c in df.columns])  # garatinr que colunas de features não serão sobreescritas 
    features = pd.DataFrame(
        [extrair_features_nlp(t) for t in df["texto"]],
        index=df.index,
    )
    return pd.concat([df, features], axis=1)

df_treino    = adicionar_features(df_treino)
df_validacao = adicionar_features(df_validacao)
df_teste     = adicionar_features(df_teste)
