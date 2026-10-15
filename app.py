from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Extração das 30 features (mesmo código usado no treino do modelo)
from Features import extrair_features_nlp, NOMES_FEATURES
from mudança_texto import main as aplicar_mutacoes_texto  # Importa a função main de mudança_texto para realizar mutações



# ================================================== CONFIGURAÇÕES ============================================================



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
    conteudo = arquivo.getvalue()   # bytes do arquivo

    # utf-8-sig: UTF-8 que também remove o BOM (marca invisível que o Bloco de Notas pode inserir)
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

# Botões de ação centralizados (Classificar e Gerar Mutação)
col_b1, col_b2 = st.columns(2)  # Divide a área de botões em duas colunas iguais
with col_b1:  # Coluna da esquerda para a classificação original
    clicou = st.button("Classificar texto", type="primary", use_container_width=True)  # Botão de classificação
with col_b2:  # Coluna da direita para a geração de mutações de texto
    clicou_mutar = st.button("🧬 Gerar mutação do texto", use_container_width=True)  # Botão de mutação do texto



# Inicializa as variáveis no estado da sessão para armazenar o texto modificado e o texto de referência
if "texto_modificado" not in st.session_state:  # Verifica se a chave do texto mutado existe na sessão do Streamlit
    st.session_state["texto_modificado"] = None  # Define o valor inicial como None para o texto mutado
if "ultimo_texto_mutado" not in st.session_state:  # Verifica se a chave do texto base existe na sessão
    st.session_state["ultimo_texto_mutado"] = None  # Define o valor inicial como None para o texto base

# Se o usuário alterou o texto original na entrada, descarta a mutação anterior para considerar o novo texto
if st.session_state.get("texto_modificado") and texto != st.session_state.get("ultimo_texto_mutado"):  # Detecta mudança no texto
    st.session_state["texto_modificado"] = None  # Limpa o texto modificado que ficou desatualizado
    st.session_state["ultimo_texto_mutado"] = None  # Limpa a referência do texto que gerou a mutação



# ======================================================= MUTAÇÃO DE TEXTO ===================================================================



if clicou_mutar:  # Verifica se o usuário pressionou o botão para gerar a mutação do texto
    if not texto.strip():  # Valida se o usuário inseriu algum texto para sofrer mutação
        st.warning("Envie ou digite um texto antes de gerar a mutação.")  # Alerta o usuário caso o campo esteja vazio
    else:  # Caso o texto fornecido pelo usuário seja válido
        try:  # Abre bloco protegido para o processamento das mutações
            with st.spinner("Aplicando mutações ao texto..."):  # Exibe indicador visual de carregamento
                texto_gerado = aplicar_mutacoes_texto(texto)  # Invoca o algoritmo de mutação passando o texto recebido
                st.session_state["texto_modificado"] = texto_gerado  # Salva o texto modificado no session_state para persistir
                st.session_state["ultimo_texto_mutado"] = texto  # Registra o texto de origem que gerou esta mutação

            st.success("Mutações aplicadas! Ao clicar em 'Classificar texto', o modelo analisará o texto modificado.")  # Feedback positivo

        except Exception as erro_mutacao:  # Captura eventuais erros durante a mutação
            st.error(f"Erro ao processar a mutação do texto: {erro_mutacao}")  # Exibe mensagem de erro na interface


# Exibe a área do texto modificado se ele tiver sido gerado na sessão
if st.session_state.get("texto_modificado"):  # Verifica se há um texto mutado armazenado na sessão
    st.markdown("### 📝 Texto com Mutações Geradas")  # Exibe título da seção de resultado da mutação
    col_aviso, col_limpar = st.columns([3, 1])  # Divide espaço entre aviso e botão de descarte

    with col_aviso:  # Coluna informativa
        st.info("ℹ️ O texto modificado abaixo é o que será classificado.")  # Informa que a classificação usará o texto mutado
        
    with col_limpar:  # Coluna da ação de resetar
        if st.button("Voltar ao original", use_container_width=True):  # Botão para descartar a mutação
            st.session_state["texto_modificado"] = None  # Limpa o texto mutado da sessão
            st.session_state["ultimo_texto_mutado"] = None  # Limpa também a referência do texto base
            st.rerun()  # Recarrega a página para refletir a alteração imediatamente

    if st.session_state.get("texto_modificado"):  # Se a mutação ainda estiver ativa
        st.text_area(  # Renderiza uma área de texto interativa com o resultado da mutação
            "Texto modificado ativo:",  # Rótulo descritivo do campo
            value=st.session_state["texto_modificado"],  # Atribui o conteúdo gerado salvo na sessão
            height=200,  # Define a altura visual do componente na interface
        ) 



# ========================================================= CLASSIFICAÇÃO ==================================================================



if clicou:  # Verifica se o botão de classificação foi clicado
    # Define o texto alvo: usa o texto modificado se existir na sessão, caso contrário utiliza o texto original
    texto_alvo = st.session_state["texto_modificado"] if st.session_state.get("texto_modificado") else texto  # Seleciona o texto apropriado
    tipo_texto = "Texto Modificado" if st.session_state.get("texto_modificado") else "Texto Original"  # Define rótulo da origem do texto
    n_palavras = len(texto_alvo.split())  # Conta a quantidade de palavras do texto alvo

    if not texto_alvo.strip():  # Valida se o texto a classificar está vazio
        st.warning("Envie um texto ou um arquivo .txt antes de classificar.")  # Exibe alerta para texto não preenchido

    elif len(texto_alvo) > MAX_CARACTERES:  # Verifica se o texto excede o limite máximo suportado
        st.error(f"Texto muito grande (máximo de {MAX_CARACTERES:,} caracteres).".replace(",", "."))  # Alerta sobre excesso de caracteres

    else:  # Caso o texto atenda aos critérios de validação
        try:  # Abre bloco protegido para inferência com o modelo
            with st.spinner(f"Classificando ({tipo_texto})..."):  # Exibe animação de carregamento informando qual texto é classificado
                modelo = carregar_modelo()  # Carrega o modelo de classificação treinado

                # Mesmo caminho do treino: texto -> 30 features (mesma ordem) -> modelo
                vetor = pd.DataFrame([extrair_features_nlp(texto_alvo)])[NOMES_FEATURES]  # Extrai as 30 features NLP do texto alvo
                categoria = modelo.predict(vetor)[0]  # Executa a previsão do nível de escolaridade

            nome_categoria = NOMES_CATEGORIAS.get(categoria, categoria)  # Obtém o nome legível da categoria prevista

            st.markdown(  # Exibe o painel com o resultado previsto
                f"""
                <div class="resultado">
                    <div class="rotulo">Nível previsto ({tipo_texto})</div>
                    <div class="categoria">{nome_categoria}</div>
                </div>
                """,
                unsafe_allow_html=True,  # Permite renderização de HTML personalizado
            )  

            if n_palavras < MIN_PALAVRAS_RECOMENDADO:  # Avisa caso o texto tenha menos palavras que o recomendado
                st.markdown(  # Exibe aviso sobre a confiabilidade do resultado
                    f"<p class='centro'><small>⚠️ O {tipo_texto.lower()} tem {n_palavras} palavras. O modelo foi treinado com "
                    f"textos de ~{MIN_PALAVRAS_RECOMENDADO} palavras ou mais, então o resultado pode ser "
                    f"menos confiável.</small></p>",
                    unsafe_allow_html=True,  # Permite HTML no aviso
                )  


        except FileNotFoundError:  # Trata ausência do arquivo de modelo .joblib
            st.error(f"Modelo não encontrado em: {CAMINHO_MODELO}")  # Notifica que o modelo não foi encontrado
        except Exception as erro:  # Trata qualquer outra exceção no fluxo de inferência
            st.error(f"Erro ao classificar o texto: {erro}")  # Exibe detalhes do erro ocorrido

