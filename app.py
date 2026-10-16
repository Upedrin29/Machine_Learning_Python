from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


from Features import extrair_features_nlp, NOMES_FEATURES
from mudança_texto import main as aplicar_mutacoes_texto  



# ================================================== CONFIGURAÇÕES ============================================================




CAMINHO_MODELO = Path(__file__).parent / "modelo_svc_complexidade.joblib"


EFICIENCIA_MODELO = 0.785


MIN_PALAVRAS_RECOMENDADO = 300
MAX_CARACTERES = 500_000


NOMES_CATEGORIAS = {
    "1_Ensino_Fundamental_I": "Ensino Fundamental I",
    "2_Ensino_Fundamental_II": "Ensino Fundamental II",
    "3_Ensino_Medio": "Ensino Médio",
    "4_Ensino_Superior": "Ensino Superior",
}


st.set_page_config(page_title="Modelo Predição", page_icon="📝", layout="centered")


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
        .resultado .eficiencia {
            font-size: 1.05rem;
            margin-top: 0.6rem;
            opacity: 0.9;
        }
        .resultado .eficiencia strong {
            color: #2e7d32;
        }
    </style>
    """,
    unsafe_allow_html=True,
)



# ======================================== FUNÇÕES AUXILIARES ================================================================



# Carrega o modelo uma única vez (e não a cada clique)
@st.cache_resource
def carregar_modelo():
    return joblib.load(CAMINHO_MODELO)


def ler_arquivo_txt(arquivo):
    """
    Extrai o texto de um arquivo .txt enviado pelo usuário.
    Tenta várias codificações, pois arquivos em português nem sempre são UTF-8.
    """
    conteudo = arquivo.getvalue()  

 
    for codificacao in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return conteudo.decode(codificacao).strip()
        except UnicodeDecodeError:
            continue
    return ""



# =============================================== INTERFACE ==================================================================



st.title("Modelo Predição")

st.markdown(
    """
    <p class="centro">
    Esta ferramenta estima o <b>nível de escolaridade</b> de um texto em português
    (Ensino Fundamental I, Fundamental II, Médio ou Superior)<br>
    Digite ou cole um texto, ou envie um arquivo <b>.txt</b>, e clique em <b>Classificar texto</b>.
    </p>
    """,
    unsafe_allow_html=True,
)

st.write("")


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


col_b1, col_b2 = st.columns(2)  
with col_b1: 
    clicou = st.button("Classificar texto", type="primary", use_container_width=True) 
with col_b2: 
    clicou_mutar = st.button("🧬 Gerar mutação do texto", use_container_width=True) 




if "texto_modificado" not in st.session_state:  
    st.session_state["texto_modificado"] = None  
if "ultimo_texto_mutado" not in st.session_state: 
    st.session_state["ultimo_texto_mutado"] = None  


if st.session_state.get("texto_modificado") and texto != st.session_state.get("ultimo_texto_mutado"):  
    st.session_state["texto_modificado"] = None  
    st.session_state["ultimo_texto_mutado"] = None  



# ======================================================= MUTAÇÃO DE TEXTO ===================================================================



if clicou_mutar:  
    if not texto.strip(): 
        st.warning("Envie ou digite um texto antes de gerar a mutação.") 
    else:  
        try: 
            with st.spinner("Aplicando mutações ao texto..."):  
                texto_gerado = aplicar_mutacoes_texto(texto)  
                st.session_state["texto_modificado"] = texto_gerado  
                st.session_state["ultimo_texto_mutado"] = texto  

            st.success("Mutações aplicadas! Ao clicar em 'Classificar texto', o modelo analisará o texto modificado.") 

        except Exception as erro_mutacao:  
            st.error(f"Erro ao processar a mutação do texto: {erro_mutacao}")  


if st.session_state.get("texto_modificado"):  
    st.markdown("### 📝 Texto com Mutações Geradas")  
    col_aviso, col_limpar = st.columns([3, 1]) 

    with col_aviso: 
        st.info("ℹ️ O texto modificado abaixo é o que será classificado.") 
        
    with col_limpar:
        if st.button("Voltar ao original", use_container_width=True): 
            st.session_state["texto_modificado"] = None  
            st.session_state["ultimo_texto_mutado"] = None  
            st.rerun() 

    if st.session_state.get("texto_modificado"):  
        st.text_area(  
            "Texto modificado ativo:",  
            value=st.session_state["texto_modificado"],  
            height=200,  
        ) 



# ========================================================= CLASSIFICAÇÃO ==================================================================



if clicou:  
    texto_alvo = st.session_state["texto_modificado"] if st.session_state.get("texto_modificado") else texto  
    tipo_texto = "Texto Modificado" if st.session_state.get("texto_modificado") else "Texto Original"  
    n_palavras = len(texto_alvo.split())  

    if not texto_alvo.strip(): 
        st.warning("Envie um texto ou um arquivo .txt antes de classificar.")  

    elif len(texto_alvo) > MAX_CARACTERES:  
        st.error(f"Texto muito grande (máximo de {MAX_CARACTERES:,} caracteres).".replace(",", ".")) 

    else:  
        try: 
            with st.spinner(f"Classificando ({tipo_texto})..."):  
                modelo = carregar_modelo()  
                vetor = pd.DataFrame([extrair_features_nlp(texto_alvo)])[NOMES_FEATURES]  
                categoria = modelo.predict(vetor)[0]  

            nome_categoria = NOMES_CATEGORIAS.get(categoria, categoria)  
            eficiencia_str = f"{EFICIENCIA_MODELO * 100:.1f}%".replace(".", ",")  

            st.markdown(  
                f"""
                <div class="resultado">
                    <div class="rotulo">Nível previsto ({tipo_texto})</div>
                    <div class="categoria">{nome_categoria}</div>
                    <div class="eficiencia">Eficiência na previsão do modelo: <strong>{eficiencia_str}</strong></div>
                </div>
                """,
                unsafe_allow_html=True, 
            )  

            if n_palavras < MIN_PALAVRAS_RECOMENDADO:  
                st.markdown(  
                    f"<p class='centro'><small>⚠️ O {tipo_texto.lower()} tem {n_palavras} palavras. O modelo foi treinado com "
                    f"textos de ~{MIN_PALAVRAS_RECOMENDADO} palavras ou mais, então o resultado pode ser "
                    f"menos confiável.</small></p>",
                    unsafe_allow_html=True,  
                )  


        except FileNotFoundError: 
            st.error(f"Modelo não encontrado em: {CAMINHO_MODELO}")  
        except Exception as erro: 
            st.error(f"Erro ao classificar o texto: {erro}") 

