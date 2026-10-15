from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Extração das 30 features (mesmo código usado no treino do modelo)
from Features import extrair_features_nlp, NOMES_FEATURES


# ============================== configurações ==============================


# Caminho do modelo salvo: procura o .joblib na mesma pasta deste arquivo
CAMINHO_MODELO = Path(__file__).parent / "modelo_svc_complexidade.joblib"

# Os textos de treino tinham ~300 palavras ou mais; abaixo disso o resultado é menos confiável
MIN_PALAVRAS_RECOMENDADO = 300

# Limite de segurança: o spaCy rejeita textos com mais de 1.000.000 de caracteres
MAX_CARACTERES = 500_000

# Normalizando nomes das categorias devolvidas pelo modelo
NOMES_CATEGORIAS = {
    "1_Ensino_Fundamental_I": "Ensino Fundamental I",
    "2_Ensino_Fundamental_II": "Ensino Fundamental II",
    "3_Ensino_Medio": "Ensino Médio",
    "4_Ensino_Superior": "Ensino Superior",
}

# layout="centered" mantém o conteúdo em uma coluna central da página
st.set_page_config(page_title="Modelo Predição", page_icon="📝", layout="centered")

# CSS: centraliza títulos e textos de apoio
st.markdown(
    """
    <style>
        h1, h3, .centro { text-align: center; }
        .resultado {
            text-align: center;
            padding: 1.2rem;
            margin-top: 1rem;
            border-radius: 12px;
            border: 1px solid rgba(128, 128, 128, 0.4);
            background: rgba(128, 128, 128, 0.12);
        }
        .resultado .rotulo { font-size: 0.95rem; opacity: 0.8; }
        .resultado .categoria { font-size: 1.8rem; font-weight: 700; margin-top: 0.3rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================== funções auxiliares ==============================


# Carrega o modelo uma única vez (e não a cada clique)
@st.cache_resource
def carregar_modelo():
    return joblib.load(CAMINHO_MODELO)


def ler_arquivo_txt(arquivo):
    """
    Extrai o texto de um arquivo .txt enviado pelo usuário.
    Tenta várias codificações, pois arquivos em português nem sempre são UTF-8.
    """
    conteudo = arquivo.getvalue()   # bytes do arquivo

    # utf-8-sig: UTF-8 que também remove o BOM (marca invisível que o Bloco de Notas pode inserir)
    for codificacao in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return conteudo.decode(codificacao).strip()
        except UnicodeDecodeError:
            continue
    return ""


# ============================== interface ==============================


st.title("Modelo Predição")

st.markdown(
    """
    <p class="centro">
    Esta ferramenta estima o <b>nível de escolaridade</b> de um texto em português
    (Ensino Fundamental I, Fundamental II, Médio ou Superior), com base em características
    como tamanho das frases, classes gramaticais, pontuação e vocabulário.<br>
    Digite ou cole um texto, ou envie um arquivo <b>.txt</b>, e clique em <b>Classificar texto</b>.
    </p>
    """,
    unsafe_allow_html=True,
)

st.write("")

# Escolha da forma de entrada (centralizada)
_, col_opcao, _ = st.columns([1, 3, 1])
with col_opcao:
    modo = st.radio(
        "Como deseja enviar o texto?",
        ["Digitar ou colar texto", "Enviar arquivo .txt"],
        horizontal=True,
        label_visibility="collapsed",
    )

texto = ""

if modo == "Digitar ou colar texto":
    texto = st.text_area(
        "Texto a ser classificado",
        height=250,
        placeholder="Cole ou digite o texto aqui...",
    )
else:
    arquivo = st.file_uploader("Selecione um arquivo .txt", type=["txt"])
    if arquivo is not None:
        texto = ler_arquivo_txt(arquivo)
        if texto:
            st.caption(f"Arquivo carregado: {arquivo.name} ({len(texto.split())} palavras)")
        else:
            st.error("Não foi possível ler o conteúdo do arquivo.")

# Botão centralizado
_, col_botao, _ = st.columns([1, 2, 1])
with col_botao:
    clicou = st.button("Classificar texto", type="primary", use_container_width=True)


# ============================== classificação ==============================


if clicou:
    n_palavras = len(texto.split())

    if not texto.strip():
        st.warning("Envie um texto ou um arquivo .txt antes de classificar.")
    elif len(texto) > MAX_CARACTERES:
        st.error(f"Texto muito grande (máximo de {MAX_CARACTERES:,} caracteres).".replace(",", "."))
    else:
        try:
            with st.spinner("Classificando..."):
                modelo = carregar_modelo()
                # Mesmo caminho do treino: texto -> 30 features (mesma ordem) -> modelo
                vetor = pd.DataFrame([extrair_features_nlp(texto)])[NOMES_FEATURES]
                categoria = modelo.predict(vetor)[0]

            nome_categoria = NOMES_CATEGORIAS.get(categoria, categoria)

            st.markdown(
                f"""
                <div class="resultado">
                    <div class="rotulo">Nível previsto</div>
                    <div class="categoria">{nome_categoria}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if n_palavras < MIN_PALAVRAS_RECOMENDADO:
                st.markdown(
                    f"<p class='centro'><small>⚠️ O texto tem {n_palavras} palavras. O modelo foi treinado com "
                    f"textos de ~{MIN_PALAVRAS_RECOMENDADO} palavras ou mais, então o resultado pode ser "
                    f"menos confiável.</small></p>",
                    unsafe_allow_html=True,
                )

        except FileNotFoundError:
            st.error(f"Modelo não encontrado em: {CAMINHO_MODELO}")
        except Exception as erro:
            st.error(f"Erro ao classificar o texto: {erro}")