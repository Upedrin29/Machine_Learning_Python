# Módulo de extração de features: transforma um texto bruto no vetor de 30 features
# que o modelo SVC espera receber. Usado pelo app (app.py) na hora da classificação.


# ============================== BIBLIOTECAS E IMPORTAÇÕES ===========================================================================


# Biblioteca de cálculo numérico: usada para média (np.mean) e desvio-padrão (np.std) dos tamanhos
import numpy as np

# Biblioteca de PLN (NLTK): usada aqui para baixar os recursos de tokenização e para o word_tokenize
import nltk

#Biblioteca de PLN mais moderna e rápida
import spacy

# trabalha com expressões regulares(REGEX): uma linguagem de padrões para procurar, extrair e substituir trechos de texto.
import re

# normalização de caracteres
import unicodedata

# tokenização
from nltk.tokenize import word_tokenize


# Downloads essenciais para viabilizar as ferramentas de NLP do NLTK
nltk.download('punkt')
nltk.download('punkt_tab')

# Carregamento do modelo de português do spaCy (exige instalação prévia: python -m spacy download pt_core_news_sm)
# utilizando biblioteca de NLP spaCY pois aqui ela facilita o uso do pos_taggin para classificar gramaticalmente 
nlp = spacy.load("pt_core_news_sm", disable=["ner"])  



# ============================ CRIANDO FEATURES DE PALAVRAS CANÔNICAS ===============================================================



# Conjuntos de busca otimizada O(1), criados uma única vez 
VOGAIS = set("aeiouáéíóúâêôãõà")
CONSOANTES = set("bcçdfghjklmnpqrstvwxyz")  

# A sílaba canônica é CV em termos de som, digrafos representam um único som consonantal
# cria e guarda um padrão pronto para uso transforma o texto do padrão em um objeto "compilado"
DIGRAFOS = re.compile(r"ch|lh|nh|rr|ss|qu(?=[eéêií])|gu(?=[eéêií])")


# ------------- função para criar feature de canonicidade ----------------------------- 


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




# ===================================== CRIANDO FEATURES ====================================================================



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


    # ========================== Features Estruturais =================================


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


    # ========================== Features Lexicais ===================================

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