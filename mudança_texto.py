import random  
import spacy  



nlp = spacy.load("pt_core_news_sm", disable=["ner"])  


# Mapeamento bidirecional de gênero para substantivos e pronomes (masculino <-> feminino)
MAPA_SUBSTANTIVOS = {  
    "cantor": "cantora",  
    "cantora": "cantor", 
    "aluno": "aluna", 
    "aluna": "aluno",  
    "professor": "professora",  
    "professora": "professor", 
    "escritor": "escritora",  
    "escritora": "escritor",  
    "menino": "menina",  
    "menina": "menino",  
    "amigo": "amiga", 
    "amiga": "amigo",  
    "médico": "médica",  
    "médica": "médico",  
    "ator": "atriz", 
    "atriz": "ator",  
    "homem": "mulher",  
    "mulher": "homem",  
    "ele": "ela",  
    "ela": "ele",  
    "eles": "elas",  
    "elas": "eles", 
    "filho": "filha",  
    "filha": "filho", 
    "irmão": "irmã",  
    "irmã": "irmão",  
}  

# Mapeamento bidirecional de determinantes, artigos e pronomes possessivos/demonstrativos
MAPA_DETERMINANTES = { 
    "o": "a",  
    "a": "o",  
    "os": "as",  
    "as": "os", 
    "um": "uma",  
    "uma": "um",  
    "uns": "umas",  
    "umas": "uns",  
    "este": "esta",  
    "esta": "este",  
    "esse": "essa",  
    "essa": "esse",  
    "aquele": "aquela",  
    "aquela": "aquele",  
    "meu": "minha",  
    "minha": "meu",  
    "seu": "sua",  
    "sua": "seu",  
    "nosso": "nossa",  
    "nossa": "nosso",  
    "pelo": "pela",  
    "pela": "pelo",  
    "pelos": "pelas",  
    "pelas": "pelos",  
    "dele": "dela",  
    "dela": "dele", 
    "no": "na",  
    "na": "no", 
    "nos": "nas",  
    "nas": "nos",  
    "ao": "à",  
    "à": "ao",  
    "aos": "às", 
    "às": "aos", 
} 

# Mapeamento bidirecional para flexão de gênero em adjetivos e particípios
MAPA_ADJETIVOS = {  
    "famoso": "famosa",  
    "famosa": "famoso",  
    "famosos": "famosas", 
    "famosas": "famosos", 
    "bonito": "bonita", 
    "bonita": "bonito", 
    "alto": "alta",  
    "alta": "alto", 
    "baixo": "baixa", 
    "baixa": "baixo", 
    "novo": "nova", 
    "nova": "novo",
    "novos": "novas", 
    "novas": "novos", 
    "antigo": "antiga", 
    "antiga": "antigo", 
    "dedicado": "dedicada", 
    "dedicada": "dedicado", 
    "cansado": "cansada", 
    "cansada": "cansado", 
    "animado": "animada",
    "animada": "animado", 
    "conhecido": "conhecida", 
    "conhecida": "conhecido", 
    "aplaudido": "aplaudida", 
    "aplaudida": "aplaudido", 
    "lido": "lida", 
    "lida": "lido", 
}  


# Dicionário de sinônimos contextualizados por classe gramatical (POS)
SINONIMOS_POS = {  
    ("famoso", "ADJ"): ["conhecido", "notável", "célebre", "ilustre"],
    ("famosa", "ADJ"): ["conhecida", "notável", "célebre", "ilustre"], 
    ("aluno", "NOUN"): ["estudante", "discípulo", "aprendiz"], 
    ("aluna", "NOUN"): ["estudante", "discípula", "aprendiz"], 
    ("cantor", "NOUN"): ["vocalista", "músico", "intérprete"], 
    ("cantora", "NOUN"): ["vocalista", "música", "intérprete"], 
    ("texto", "NOUN"): ["artigo", "escrito", "documento"], 
    ("livro", "NOUN"): ["obra", "volume", "exemplar"],
    ("rapidamente", "ADV"): ["depressa", "velozmente", "ligeiramente"],  
    ("grande", "ADJ"): ["amplo", "extenso", "imenso"],  
    ("pequeno", "ADJ"): ["reduzido", "curto", "modesto"], 
    ("pequena", "ADJ"): ["reduzida", "curta", "modesta"], 
    ("feliz", "ADJ"): ["contente", "alegre", "satisfeito"], 
    ("bonito", "ADJ"): ["belo", "lindo", "atraente"], 
    ("bonita", "ADJ"): ["bela", "linda", "atraente"],  
}  

# Dicionário de antônimos moderados contextualizados por classe gramatical
ANTONIMOS_POS = {  
    ("famoso", "ADJ"): ["desconhecido", "anônimo"], 
    ("famosa", "ADJ"): ["desconhecida", "anônima"], 
    ("grande", "ADJ"): ["pequeno", "reduzido"], 
    ("pequeno", "ADJ"): ["grande", "amplo"], 
    ("feliz", "ADJ"): ["triste", "descontente"], 
    ("bonito", "ADJ"): ["feio", "desagradável"], 
    ("bonita", "ADJ"): ["feia", "desagradável"], 
    ("novo", "ADJ"): ["antigo", "velho"], 
    ("antigo", "ADJ"): ["recente", "moderno"], 
} 


# Função auxiliar para espelhar a caixa alta/baixa
def manter_capitalizacao(palavra_original, nova_palavra):  

    if palavra_original.isupper():  
        return nova_palavra.upper()  
    elif palavra_original.istitle():  
        return nova_palavra.capitalize()  


    return nova_palavra.lower() 


# Função que realiza a troca de gênero com concordância sintática em bloco
def mutacao_troca_genero_inteligente(doc):  

    substituicoes = {}  

    for token in doc:  
        chave_token = token.text.lower()  
        if token.pos_ in ("NOUN", "PRON") and chave_token in MAPA_SUBSTANTIVOS: 
            novo_substantivo = MAPA_SUBSTANTIVOS[chave_token]  
            substituicoes[token.i] = manter_capitalizacao(token.text, novo_substantivo)  
            
            for filho in token.children: 
                filho_chave = filho.text.lower() 

                if filho.dep_ in ("det", "det:poss") and filho_chave in MAPA_DETERMINANTES:  
                    novo_artigo = MAPA_DETERMINANTES[filho_chave]  
                    substituicoes[filho.i] = manter_capitalizacao(filho.text, novo_artigo)  
                elif filho.dep_ in ("amod", "appos", "nmod") and filho_chave in MAPA_ADJETIVOS:  
                    novo_adj = MAPA_ADJETIVOS[filho_chave]  
                    substituicoes[filho.i] = manter_capitalizacao(filho.text, novo_adj)  

            verbo_pai = token.head  

            if token.dep_ in ("nsubj", "nsubj:pass") and verbo_pai.pos_ in ("AUX", "VERB"):  
                for irmao in verbo_pai.children: 
                    irmao_chave = irmao.text.lower() 
                    if irmao.dep_ in ("xcomp", "acomp", "attr") and irmao_chave in MAPA_ADJETIVOS: 
                        novo_predicativo = MAPA_ADJETIVOS[irmao_chave]  
                        substituicoes[irmao.i] = manter_capitalizacao(irmao.text, novo_predicativo) 

    tokens_resultado = []  
    for token in doc:  
        if token.i in substituicoes:  
            tokens_resultado.append(substituicoes[token.i] + token.whitespace_)  
        else:  
            tokens_resultado.append(token.text + token.whitespace_)  



    return "".join(tokens_resultado)  



# Função que substitui palavras com critério morfossintático
def mutacao_troca_palavras_inteligente(doc, probabilidade=0.3):  

    tokens_resultado = [] 
    for token in doc:  
        chave = (token.text.lower(), token.pos_)  
        candidatos = []  
        if chave in SINONIMOS_POS:  
            candidatos.extend(SINONIMOS_POS[chave])  
        if chave in ANTONIMOS_POS:  
            candidatos.extend(ANTONIMOS_POS[chave])  

        if candidatos and random.random() < probabilidade: 
            escolhida = random.choice(candidatos) 
            termo_final = manter_capitalizacao(token.text, escolhida) 
            tokens_resultado.append(termo_final + token.whitespace_) 
        else:  
            tokens_resultado.append(token.text + token.whitespace_)  


    return "".join(tokens_resultado) 


# Função orquestradora que aplica todas as mutações coerentes em sequência
def gerar_mutacoes_texto(texto_origem):  

    if not texto_origem or not str(texto_origem).strip():  
        return "" 
    doc_inicial = nlp(str(texto_origem).strip())  
    texto_com_genero = mutacao_troca_genero_inteligente(doc_inicial)  
    doc_intermediario = nlp(texto_com_genero)  
    
    texto_final = mutacao_troca_palavras_inteligente(doc_intermediario)  


    return texto_final.strip()  


# Ponto de entrada integrado com a interface Streamlit do app.py
def main(texto_usuario=None):  

    if texto_usuario is not None:  
        return gerar_mutacoes_texto(texto_usuario) 



if __name__ == "__main__":  
     main()  


