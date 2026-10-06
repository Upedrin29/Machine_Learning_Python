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


# Downloads essenciais para viabilizar as ferramentas de NLP do NLTK
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')

# Carregamento do modelo de português do spaCy (exige instalação prévia: python -m spacy download pt_core_news_sm)
nlp = spacy.load("pt_core_news_sm")



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

display(df_treino, df_validacao, df_teste)
