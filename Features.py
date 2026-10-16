# Módulo de extração de features: transforma um texto bruto no vetor de 30 features
# que o modelo SVC espera receber. Usado pelo app (app.py) na hora da classificação.


# ============================== BIBLIOTECAS E IMPORTAÇÕES ===========================================================================



import numpy as np


import nltk


import spacy


import re


import unicodedata


from nltk.tokenize import word_tokenize



nltk.download('punkt')
nltk.download('punkt_tab')
nlp = spacy.load("pt_core_news_sm", disable=["ner"])  



# ============================ CRIANDO FEATURES DE PALAVRAS CANÔNICAS ===============================================================




VOGAIS = set("aeiouáéíóúâêôãõà")
CONSOANTES = set("bcçdfghjklmnpqrstvwxyz")  
DIGRAFOS = re.compile(r"ch|lh|nh|rr|ss|qu(?=[eéêií])|gu(?=[eéêií])")


# ------------- função para criar feature de canonicidade ----------------------------- 


def calcular_canonicidade(texto):
    """
    Recebe um texto bruto, tokeniza, avalia a estrutura CV de cada palavra
    e retorna a proporção de palavras canônicas.
    """

    texto = unicodedata.normalize("NFC", texto.lower())

    tokens = word_tokenize(texto, language='portuguese')

    palavras = [p for p in tokens if p.isalpha()]


    if not palavras:
        return 0.0

    total_canonicas = 0


    for palavra in palavras:
      
        palavra = DIGRAFOS.sub("b", palavra)


        if len(palavra) % 2 != 0:
            continue

        e_canonica = True

    
        for i, letra in enumerate(palavra):
         
            if i % 2 == 0 and letra not in CONSOANTES:
                e_canonica = False
                break

            elif i % 2 != 0 and letra not in VOGAIS:
                e_canonica = False
                break

        if e_canonica:
            total_canonicas += 1


    return total_canonicas / len(palavras)




# ===================================== CRIANDO FEATURES ====================================================================



# ---------- Constantes (criadas uma única vez) ----------


PADROES_PONTUACAO = {
    "virgula":       re.compile(r","),
    "ponto_final":   re.compile(r"(?<!\.)\.(?!\.)"),   
    "reticencias":   re.compile(r"\.{3}|…"),
    "interrogacao":  re.compile(r"\?"),
    "exclamacao":    re.compile(r"!"),
    "ponto_virgula": re.compile(r";"),
    "dois_pontos":   re.compile(r":"),
    "aspas":         re.compile(r"[\"“”«»]"),
    "parenteses":    re.compile(r"[()]"),
    "travessao":     re.compile(r"[—–]"),
}


CLASSES_POS = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP",
               "CCONJ", "SCONJ", "AUX", "NUM", "PROPN", "INTJ"]


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

    doc = nlp(texto)


    palavras = [t for t in doc if t.is_alpha]
    palavras = [t for t in doc if t.is_alpha]


    n_palavras = len(palavras)

    if n_palavras == 0:
        return {nome: 0.0 for nome in NOMES_FEATURES}

    features = {}   


    # ========================== Features Estruturais =================================


    # ---------------- Feature de pontução  ----------------


    for nome, padrao in PADROES_PONTUACAO.items():
        features[f"punct_{nome}"] = len(padrao.findall(texto)) / n_palavras


    # ---------------- Feature de tamanho medio sentenças (spaCy) ----------------

   
    tamanhos = [sum(1 for t in sent if t.is_alpha) for sent in doc.sents]

    tamanhos = [n for n in tamanhos if n > 0]

    
    features["sent_tam_medio"] = float(np.mean(tamanhos))
    features["sent_tam_desvio"] = float(np.std(tamanhos))


    # ----------------- Feature de classes gramaticais (spaCy) ----------------

    contagem_pos = {tag: 0 for tag in CLASSES_POS}


    for token in palavras:
       
        if token.pos_ in contagem_pos:
            contagem_pos[token.pos_] += 1

  
    for tag, contagem in contagem_pos.items():
        features[f"pos_{tag}"] = contagem / n_palavras


    # ========================== Features Lexicais ===================================

    
    formas = [t.text.lower() for t in palavras]

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