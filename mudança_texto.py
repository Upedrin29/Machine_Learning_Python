import random  # Importa o módulo random para seleções e sorteios aleatórios de termos
import spacy  # Importa a biblioteca spaCy para processamento de linguagem natural e análise sintática


# Inicializa o pipeline de análise morfossintática do modelo de português desativando NER
nlp = spacy.load("pt_core_news_sm", disable=["ner"])  # Carrega o modelo pt_core_news_sm


# Mapeamento bidirecional de gênero para substantivos e pronomes (masculino <-> feminino)
MAPA_SUBSTANTIVOS = {  
    "cantor": "cantora",  # Converte substantivo masculino 'cantor' para feminino 'cantora'
    "cantora": "cantor",  # Converte substantivo feminino 'cantora' para masculino 'cantor'
    "aluno": "aluna",  # Converte substantivo masculino 'aluno' para feminino 'aluna'
    "aluna": "aluno",  # Converte substantivo feminino 'aluna' para masculino 'aluno'
    "professor": "professora",  # Converte masculino 'professor' para feminino 'professora'
    "professora": "professor",  # Converte feminino 'professora' para masculino 'professor'
    "escritor": "escritora",  # Converte masculino 'escritor' para feminino 'escritora'
    "escritora": "escritor",  # Converte feminino 'escritora' para masculino 'escritor'
    "menino": "menina",  # Converte masculino 'menino' para feminino 'menina'
    "menina": "menino",  # Converte feminino 'menina' para masculino 'menino'
    "amigo": "amiga",  # Converte masculino 'amigo' para feminino 'amiga'
    "amiga": "amigo",  # Converte feminino 'amiga' para masculino 'amigo'
    "médico": "médica",  # Converte masculino 'médico' para feminino 'médica'
    "médica": "médico",  # Converte feminino 'médica' para masculino 'médico'
    "ator": "atriz",  # Converte masculino 'ator' para feminino 'atriz'
    "atriz": "ator",  # Converte feminino 'atriz' para masculino 'ator'
    "homem": "mulher",  # Converte masculino 'homem' para feminino 'mulher'
    "mulher": "homem",  # Converte feminino 'mulher' para masculino 'homem'
    "ele": "ela",  # Converte pronome pessoal masculino 'ele' para feminino 'ela'
    "ela": "ele",  # Converte pronome pessoal feminino 'ela' para masculino 'ele'
    "eles": "elas",  # Converte pronome plural 'eles' para 'elas'
    "elas": "eles",  # Converte pronome plural 'elas' para 'eles'
    "filho": "filha",  # Converte masculino 'filho' para 'filha'
    "filha": "filho",  # Converte feminino 'filha' para 'filho'
    "irmão": "irmã",  # Converte masculino 'irmão' para 'irmã'
    "irmã": "irmão",  # Converte feminino 'irmã' para 'irmão'
}  

# Mapeamento bidirecional de determinantes, artigos e pronomes possessivos/demonstrativos
MAPA_DETERMINANTES = { 
    "o": "a",  # Troca artigo definido singular 'o' por 'a'
    "a": "o",  # Troca artigo definido singular 'a' por 'o'
    "os": "as",  # Troca artigo definido plural 'os' por 'as'
    "as": "os",  # Troca artigo definido plural 'as' por 'os'
    "um": "uma",  # Troca artigo indefinido singular 'um' por 'uma'
    "uma": "um",  # Troca artigo indefinido singular 'uma' por 'um'
    "uns": "umas",  # Troca artigo indefinido plural 'uns' por 'umas'
    "umas": "uns",  # Troca artigo indefinido plural 'umas' por 'uns'
    "este": "esta",  # Troca pronome demonstrativo 'este' por 'esta'
    "esta": "este",  # Troca pronome demonstrativo 'esta' por 'este'
    "esse": "essa",  # Troca pronome demonstrativo 'esse' por 'essa'
    "essa": "esse",  # Troca pronome demonstrativo 'essa' por 'esse'
    "aquele": "aquela",  # Troca pronome demonstrativo 'aquele' por 'aquela'
    "aquela": "aquele",  # Troca pronome demonstrativo 'aquela' por 'aquele'
    "meu": "minha",  # Troca pronome possessivo 'meu' por 'minha'
    "minha": "meu",  # Troca pronome possessivo 'minha' por 'meu'
    "seu": "sua",  # Troca pronome possessivo 'seu' por 'sua'
    "sua": "seu",  # Troca pronome possessivo 'sua' por 'seu'
    "nosso": "nossa",  # Troca pronome possessivo 'nosso' por 'nossa'
    "nossa": "nosso",  # Troca pronome possessivo 'nossa' por 'nosso'
    "pelo": "pela",  # Troca preposição contraída 'pelo' por 'pela'
    "pela": "pelo",  # Troca preposição contraída 'pela' por 'pelo'
    "pelos": "pelas",  # Troca preposição contraída no plural 'pelos' por 'pelas'
    "pelas": "pelos",  # Troca preposição contraída no plural 'pelas' por 'pelos'
    "dele": "dela",  # Troca contração 'dele' por 'dela'
    "dela": "dele",  # Troca contração 'dela' por 'dele'
    "no": "na",  # Troca em+o por em+a
    "na": "no",  # Troca em+a por em+o
    "nos": "nas",  # Troca em+os por em+as
    "nas": "nos",  # Troca em+as por em+os
    "ao": "à",  # Troca a+o por a+a (crase)
    "à": "ao",  # Troca a+a por a+o
    "aos": "às",  # Troca a+os por a+as
    "às": "aos",  # Troca a+as por a+os
} 

# Mapeamento bidirecional para flexão de gênero em adjetivos e particípios
MAPA_ADJETIVOS = {  
    "famoso": "famosa",  # Flexiona 'famoso' para 'famosa'
    "famosa": "famoso",  # Flexiona 'famosa' para 'famoso'
    "famosos": "famosas",  # Flexiona no plural 'famosos' para 'famosas'
    "famosas": "famosos",  # Flexiona no plural 'famosas' para 'famosos'
    "bonito": "bonita",  # Flexiona 'bonito' para 'bonita'
    "bonita": "bonito",  # Flexiona 'bonita' para 'bonito'
    "alto": "alta",  # Flexiona 'alto' para 'alta'
    "alta": "alto",  # Flexiona 'alta' para 'alto'
    "baixo": "baixa",  # Flexiona 'baixo' para 'baixa'
    "baixa": "baixo",  # Flexiona 'baixa' para 'baixo'
    "novo": "nova",  # Flexiona 'novo' para 'nova'
    "nova": "novo",  # Flexiona 'nova' para 'novo'
    "novos": "novas",  # Flexiona no plural 'novos' para 'novas'
    "novas": "novos",  # Flexiona no plural 'novas' para 'novos'
    "antigo": "antiga",  # Flexiona 'antigo' para 'antiga'
    "antiga": "antigo",  # Flexiona 'antiga' para 'antigo'
    "dedicado": "dedicada",  # Flexiona 'dedicado' para 'dedicada'
    "dedicada": "dedicado",  # Flexiona 'dedicada' para 'dedicado'
    "cansado": "cansada",  # Flexiona 'cansado' para 'cansada'
    "cansada": "cansado",  # Flexiona 'cansada' para 'cansado'
    "animado": "animada",  # Flexiona 'animado' para 'animada'
    "animada": "animado",  # Flexiona 'animada' para 'animado'
    "conhecido": "conhecida",  # Flexiona 'conhecido' para 'conhecida'
    "conhecida": "conhecido",  # Flexiona 'conhecida' para 'conhecido'
    "aplaudido": "aplaudida",  # Flexiona 'aplaudido' para 'aplaudida'
    "aplaudida": "aplaudido",  # Flexiona 'aplaudida' para 'aplaudido'
    "lido": "lida",  # Flexiona particípio 'lido' para 'lida'
    "lida": "lido",  # Flexiona particípio 'lida' para 'lido'
}  # Finaliza o dicionário de adjetivos


# Dicionário de sinônimos contextualizados por classe gramatical (POS)
SINONIMOS_POS = {  
    ("famoso", "ADJ"): ["conhecido", "notável", "célebre", "ilustre"],  # Sinônimos de adjetivo masculino
    ("famosa", "ADJ"): ["conhecida", "notável", "célebre", "ilustre"],  # Sinônimos de adjetivo feminino
    ("aluno", "NOUN"): ["estudante", "discípulo", "aprendiz"],  # Sinônimos de substantivo masculino
    ("aluna", "NOUN"): ["estudante", "discípula", "aprendiz"],  # Sinônimos de substantivo feminino
    ("cantor", "NOUN"): ["vocalista", "músico", "intérprete"],  # Sinônimos de substantivo masculino
    ("cantora", "NOUN"): ["vocalista", "música", "intérprete"],  # Sinônimos de substantivo feminino
    ("texto", "NOUN"): ["artigo", "escrito", "documento"],  # Sinônimos para texto
    ("livro", "NOUN"): ["obra", "volume", "exemplar"],  # Sinônimos para livro
    ("rapidamente", "ADV"): ["depressa", "velozmente", "ligeiramente"],  # Sinônimos de advérbio
    ("grande", "ADJ"): ["amplo", "extenso", "imenso"],  # Sinônimos de adjetivo invariável
    ("pequeno", "ADJ"): ["reduzido", "curto", "modesto"],  # Sinônimos de adjetivo pequeno
    ("pequena", "ADJ"): ["reduzida", "curta", "modesta"],  # Sinônimos de adjetivo pequena
    ("feliz", "ADJ"): ["contente", "alegre", "satisfeito"],  # Sinônimos de feliz
    ("bonito", "ADJ"): ["belo", "lindo", "atraente"],  # Sinônimos de adjetivo bonito
    ("bonita", "ADJ"): ["bela", "linda", "atraente"],  # Sinônimos de adjetivo bonita
}  

# Dicionário de antônimos moderados contextualizados por classe gramatical
ANTONIMOS_POS = {  
    ("famoso", "ADJ"): ["desconhecido", "anônimo"],  # Antônimo de adjetivo masculino
    ("famosa", "ADJ"): ["desconhecida", "anônima"],  # Antônimo de adjetivo feminino
    ("grande", "ADJ"): ["pequeno", "reduzido"],  # Antônimo de adjetivo grande
    ("pequeno", "ADJ"): ["grande", "amplo"],  # Antônimo de adjetivo pequeno
    ("feliz", "ADJ"): ["triste", "descontente"],  # Antônimo de feliz
    ("bonito", "ADJ"): ["feio", "desagradável"],  # Antônimo de bonito
    ("bonita", "ADJ"): ["feia", "desagradável"],  # Antônimo de bonita
    ("novo", "ADJ"): ["antigo", "velho"],  # Antônimo de novo
    ("antigo", "ADJ"): ["recente", "moderno"],  # Antônimo de antigo
} 


# Função auxiliar para espelhar a caixa alta/baixa
def manter_capitalizacao(palavra_original, nova_palavra):  

    if palavra_original.isupper():  # Se a palavra original for toda em maiúsculas
        return nova_palavra.upper()  # Devolve a nova palavra em letras maiúsculas
    elif palavra_original.istitle():  # Se apenas a primeira letra da original for maiúscula
        return nova_palavra.capitalize()  # Devolve com a inicial maiúscula


    return nova_palavra.lower()  # Devolve em letras minúsculas por padrão


# Função que realiza a troca de gênero com concordância sintática em bloco
def mutacao_troca_genero_inteligente(doc):  

    substituicoes = {}  # Dicionário mapeando índices de tokens para suas respectivas formas alteradas

    for token in doc:  # Itera sobre cada token analisado morfossintaticamente pelo spaCy
        chave_token = token.text.lower()  # Obtém o texto do token em formato minúsculo
        if token.pos_ in ("NOUN", "PRON") and chave_token in MAPA_SUBSTANTIVOS:  # Identifica núcleo nominal elegível
            novo_substantivo = MAPA_SUBSTANTIVOS[chave_token]  # Resgata a forma no gênero inverso para o núcleo
            substituicoes[token.i] = manter_capitalizacao(token.text, novo_substantivo)  # Registra a substituição do substantivo
            
            for filho in token.children:  # Percorre os modificadores sintáticos dependentes diretos do substantivo
                filho_chave = filho.text.lower()  # Converte o texto do termo filho para minúsculas

                if filho.dep_ in ("det", "det:poss") and filho_chave in MAPA_DETERMINANTES:  # Caso seja artigo ou determinante
                    novo_artigo = MAPA_DETERMINANTES[filho_chave]  # Obtém o artigo no gênero concordante
                    substituicoes[filho.i] = manter_capitalizacao(filho.text, novo_artigo)  # Registra o artigo concordante
                elif filho.dep_ in ("amod", "appos", "nmod") and filho_chave in MAPA_ADJETIVOS:  # Caso seja adjetivo adjunto
                    novo_adj = MAPA_ADJETIVOS[filho_chave]  # Obtém o adjetivo no gênero concordante
                    substituicoes[filho.i] = manter_capitalizacao(filho.text, novo_adj)  # Registra a substituição do adjetivo

            verbo_pai = token.head  # Identifica o verbo principal do qual o substantivo pode ser sujeito

            if token.dep_ in ("nsubj", "nsubj:pass") and verbo_pai.pos_ in ("AUX", "VERB"):  # Se for sujeito da oração
                for irmao in verbo_pai.children:  # Percorre os complementos e predicativos ligados ao verbo
                    irmao_chave = irmao.text.lower()  # Converte o termo ligado ao verbo para minúsculo
                    if irmao.dep_ in ("xcomp", "acomp", "attr") and irmao_chave in MAPA_ADJETIVOS:  # Se for predicativo do sujeito
                        novo_predicativo = MAPA_ADJETIVOS[irmao_chave]  # Flexiona o adjetivo predicativo em concordância
                        substituicoes[irmao.i] = manter_capitalizacao(irmao.text, novo_predicativo)  # Salva a flexão do predicativo

    tokens_resultado = []  # Lista para armazenar o texto de cada token após a concordância
    for token in doc:  # Percorre todos os tokens do documento original
        if token.i in substituicoes:  # Se o token sofreu flexão de gênero
            tokens_resultado.append(substituicoes[token.i] + token.whitespace_)  # Adiciona a palavra flexionada mantendo o espaçamento
        else:  # Caso o token não tenha sido modificado
            tokens_resultado.append(token.text + token.whitespace_)  # Mantém a forma original com seu espaçamento



    return "".join(tokens_resultado)  # Junta os tokens preservando exatamente os espaços e pontuações



# Função que substitui palavras com critério morfossintático
def mutacao_troca_palavras_inteligente(doc, probabilidade=0.3):  

    tokens_resultado = []  # Lista para acumular o texto processado
    for token in doc:  # Percorre cada token do texto analisado
        chave = (token.text.lower(), token.pos_)  # Cria tupla de busca unindo o lema e a classe gramatical
        candidatos = []  # Lista para armazenar sinônimos e antônimos válidos para a classe
        if chave in SINONIMOS_POS:  # Verifica se existem sinônimos cadastrados para aquela classe gramatical
            candidatos.extend(SINONIMOS_POS[chave])  # Adiciona os sinônimos compatíveis
        if chave in ANTONIMOS_POS:  # Verifica se existem antônimos cadastrados compatíveis
            candidatos.extend(ANTONIMOS_POS[chave])  # Adiciona os antônimos compatíveis

        if candidatos and random.random() < probabilidade:  # Sorteia a substituição com taxa equilibrada
            escolhida = random.choice(candidatos)  # Escolhe aleatoriamente uma opção compatível
            termo_final = manter_capitalizacao(token.text, escolhida)  # Preserva o formato de maiúsculas
            tokens_resultado.append(termo_final + token.whitespace_)  # Adiciona a palavra substituída com seu espaço
        else:  # Caso a palavra não seja sorteada ou não esteja cadastrada
            tokens_resultado.append(token.text + token.whitespace_)  # Mantém o vocábulo original com seu espaço


    return "".join(tokens_resultado)  # Reconstrói a string do texto com precisão


# Função orquestradora que aplica todas as mutações coerentes em sequência
def gerar_mutacoes_texto(texto_origem):  

    if not texto_origem or not str(texto_origem).strip():  # Valida se o texto de entrada não está em branco
        return ""  # Retorna string vazia se não houver texto
    doc_inicial = nlp(str(texto_origem).strip())  # Processa o texto original no spaCy para análise de concordância
    texto_com_genero = mutacao_troca_genero_inteligente(doc_inicial)  # Aplica troca de gênero em bloco com concordância
    doc_intermediario = nlp(texto_com_genero)  # Processa o texto resultante para validar classes gramaticais
    
    texto_final = mutacao_troca_palavras_inteligente(doc_intermediario)  # Aplica troca de palavras respeitando POS tags


    return texto_final.strip()  # Devolve o texto final mutado e coerente


# Ponto de entrada integrado com a interface Streamlit do app.py
def main(texto_usuario=None):  

    if texto_usuario is not None:  # Se receber texto vindo da interface web do Streamlit
        return gerar_mutacoes_texto(texto_usuario)  # Executa o pipeline inteligente e devolve o texto



if __name__ == "__main__":  # Garante execução apenas se invocado como script principal
    main()  # Dispara a execução principal
