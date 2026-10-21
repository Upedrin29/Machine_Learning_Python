# Classificação de complexidade textual em português

Modelo de Machine Learning que classifica textos em português por **nível de complexidade** (Ensino Fundamental I, Fundamental II, Médio e Superior), com uma **interface web em Streamlit** e um **algoritmo de mutação de texto** para observar como a classificação reage a alterações no texto.

O modelo final é um **SVC** sobre **30 features linguísticas** (pontuação, tamanho de sentenças, classes gramaticais, métricas lexicais e canonicidade silábica). No conjunto de teste, usado uma única vez, atingiu **F1 macro de 0,778 e acurácia de 0,790**.

## Sumário

- [Resultados](#resultados)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Instalação](#instalação)
- [Executando a interface](#executando-a-interface)
- [Usando o modelo em código](#usando-o-modelo-em-código)
- [Dados](#dados)
- [Features](#features)
- [Metodologia](#metodologia)
- [Experimentos](#experimentos)
- [Algoritmo de mutação de texto](#algoritmo-de-mutação-de-texto)
- [Limitações](#limitações)
- [Trabalhos futuros](#trabalhos-futuros)

## Resultados

**Comparação dos algoritmos** (validação cruzada estratificada de 5 dobras, treino + validação, 30 features):

| Modelo | F1 macro | Desvio | Acurácia |
|---|---|---|---|
| **SVC** | **0,785** | 0,015 | 0,790 |
| Random Forest | 0,771 | 0,016 | 0,778 |
| Regressão Logística | 0,742 | 0,019 | 0,741 |
| MultinomialNB | 0,342 | 0,017 | 0,487 |

SVC e Random Forest empatam tecnicamente (a diferença é menor que o desvio). O MultinomialNB falha porque assume contagens, e as features do projeto são proporções contínuas.

**Avaliação final no teste** (309 textos, usado uma única vez, modelo treinado em treino + validação):

| Classe | Precision | Recall | F1 | Suporte |
|---|---|---|---|---|
| Fundamental I | 0,80 | 0,91 | 0,85 | 44 |
| Fundamental II | 0,68 | 0,65 | 0,67 | 49 |
| Médio | 0,79 | 0,72 | 0,76 | 94 |
| Superior | 0,83 | 0,85 | 0,84 | 122 |

Matriz de confusão no teste (linhas = classe real, colunas = classe prevista):

| | Fund. I | Fund. II | Médio | Superior |
|---|---|---|---|---|
| **Fund. I** | 40 | 3 | 0 | 1 |
| **Fund. II** | 10 | 32 | 3 | 4 |
| **Médio** | 0 | 9 | 68 | 17 |
| **Superior** | 0 | 3 | 15 | 104 |

Dos 65 erros, 57 ocorrem entre classes adjacentes: o modelo aprendeu uma ordem de complexidade. A maior confusão é entre **Médio e Superior**.

![Resultado final no teste](figuras_relatorio/fig12_teste_final.png)

## Estrutura do repositório

```
.
├── construindo_ML.ipynb             # pesquisa: dados, features, experimentos, validação cruzada, teste e salvamento do modelo
├── Features.py                      # extração das 30 features de um texto
├── mudança_texto.py                 # algoritmo de mutação de texto
├── app.py                           # interface web (Streamlit)
├── modelo_svc_complexidade.joblib   # pipeline treinado (StandardScaler + SVC)
├── requirements.txt                 # versões das dependências
├── figuras_relatorio/               # figuras usadas no relatório e neste README
└── Relatório                        # relatório técnico completo do projeto
```

> Os nomes dos arquivos e os `import` precisam coincidir exatamente em maiúsculas e minúsculas (`Features.py` e `from Features import ...`) para funcionar em sistemas sensíveis à caixa, como Linux.

## Instalação

Requisitos: Python (o projeto foi desenvolvido com 3.14.6, conforme os metadados do notebook) e acesso à internet na primeira execução.

```bash
# 1. (opcional) criar e ativar um ambiente virtual
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
.venv\Scripts\activate             # Windows

# 2. instalar as dependências
pip install -r requirements.txt
```

O `requirements.txt` já inclui o modelo de português do spaCy (`pt_core_news_sm` 3.8.0). Se preferir instalá-lo à parte:

```bash
python -m spacy download pt_core_news_sm
```

Na primeira execução, `Features.py` baixa os recursos `punkt` e `punkt_tab` do NLTK, o que exige conexão com a internet.

**Versões principais** (do `requirements.txt`): pandas 3.0.6, numpy 2.5.3, scikit-learn 1.9.1, spaCy 3.8.16, NLTK 3.10.3, joblib 1.6.0 e Streamlit 1.66.0. O arquivo `.joblib` depende da versão do scikit-learn usada no treino; use a mesma versão para carregá-lo.

## Executando a interface

Na pasta do projeto:

```bash
streamlit run app.py
```

Como usar:

1. Escolha como enviar o texto: **digitar/colar** ou **enviar um arquivo `.txt`** (a leitura tenta `utf-8-sig`, `cp1252` e `latin-1`).
2. Clique em **Classificar texto** para ver o nível previsto.
3. (Opcional) Clique em **Gerar mutação do texto** para criar uma versão modificada. Enquanto existir uma mutação ativa, **Classificar texto** classifica o texto modificado. Use **Voltar ao original** para descartá-la.

A interface avisa quando o texto tem menos de 300 palavras (o modelo foi treinado com textos de ~300 palavras ou mais, então o resultado pode ser menos confiável) e recusa textos acima de 500.000 caracteres.

## Usando o modelo em código

```python
import joblib
import pandas as pd
from Features import extrair_features_nlp, NOMES_FEATURES

modelo = joblib.load("modelo_svc_complexidade.joblib")

texto = "Cole aqui um texto de pelo menos ~300 palavras..."
vetor = pd.DataFrame([extrair_features_nlp(texto)])[NOMES_FEATURES]

print(modelo.predict(vetor)[0])   # ex.: '3_Ensino_Medio'
```

As colunas precisam estar **na mesma ordem e quantidade** do treino (30 features); por isso o `[NOMES_FEATURES]`.

## Dados

O corpus não acompanha este repositório. O notebook espera uma pasta principal com uma subpasta por categoria, cada uma com arquivos `.txt` (um texto por arquivo):

```
dataset_caed/
├── 1_Ensino_Fundamental_I/*.txt
├── 2_Ensino_Fundamental_II/*.txt
├── 3_Ensino_Medio/*.txt
└── 4_Ensino_Superior/*.txt
```

Ajuste a variável `caminho_base_dataset` no notebook para o caminho local.

| Classe | Total | Treino | Validação | Teste |
|---|---|---|---|---|
| Fundamental I | 294 | 206 | 44 | 44 |
| Fundamental II | 324 | 226 | 49 | 49 |
| Médio | 628 | 440 | 94 | 94 |
| Superior | 814 | 570 | 122 | 122 |
| **Total** | **2.060** | **1.442** | **309** | **309** |

Tratamentos: arquivos vazios descartados, arquivos fora de UTF-8 registrados e ignorados, e **16 textos duplicados removidos** antes da divisão, para evitar vazamento de dados entre treino e teste. A divisão é estratificada (70/15/15) com `random_state=42`. O corpus é desbalanceado: Superior tem cerca de 2,8 vezes mais textos que Fundamental I.

## Features

A função `extrair_features_nlp(texto)` devolve um dicionário com 30 features, sempre na mesma ordem (`NOMES_FEATURES`). Todas são **proporções ou médias**, e não contagens brutas, para que o tamanho do texto não seja codificado.

| Grupo | Nº | Features | Ferramenta |
|---|---|---|---|
| Pontuação | 10 | vírgula, ponto final, reticências, interrogação, exclamação, ponto e vírgula, dois pontos, aspas, parênteses, travessão | Regex |
| Sentenças | 2 | tamanho médio e desvio-padrão (em palavras) | spaCy |
| Classes gramaticais | 13 | NOUN, VERB, ADJ, ADV, PRON, DET, ADP, CCONJ, SCONJ, AUX, NUM, PROPN, INTJ | spaCy |
| Lexicais | 4 | comprimento médio e desvio da palavra, proporção de palavras longas (> 6 letras), TTR nas primeiras 300 palavras | spaCy |
| Canonicidade | 1 | proporção de palavras formadas só por sílabas consoante + vogal | NLTK + regex |

**Canonicidade** (feature obrigatória do projeto): uma palavra é canônica quando só tem sílabas CV (*ja-ne-la*, *ga-to*). A implementação normaliza o Unicode (NFC), tokeniza com o NLTK, substitui dígrafos (`ch`, `lh`, `nh`, `rr`, `ss`, `qu` e `gu` antes de `e`/`i`) por uma consoante marcadora e exige tamanho par com alternância consoante/vogal. Teve baixo poder discriminativo (médias de 0,22 a 0,23 nas quatro classes), registrado como achado no relatório.

**TTR em janela fixa de 300 palavras:** o TTR clássico cai conforme o texto cresce, e as classes altas têm textos mais longos. Medir todos os textos nas primeiras 300 palavras evita esse viés. Foi a feature mais importante do modelo.

## Metodologia

- **Divisão:** treino (1.442), validação (309) e teste (309), estratificada.
- **Métrica de decisão:** **F1 macro**, por causa do desbalanceamento (a acurácia favorece as classes grandes).
- **Validação cruzada:** estratificada de 5 dobras sobre treino + validação (1.751 textos), com as **mesmas dobras** para todos os modelos, o que torna a comparação pareada.
- **Regra de decisão:** uma mudança só conta como melhora se a média da validação cruzada subir mais do que o desvio (≈ 0,016).
- **Prevenção de vazamento:** escaladores dentro de `Pipeline` (ajustados só no treino de cada dobra) e remoção de duplicatas. A extração de features é uma regra fixa, sem parâmetros aprendidos.
- **Teste:** usado **uma única vez**, ao final, depois de todas as decisões.

| Algoritmo | Pré-processamento | Observação |
|---|---|---|
| Random Forest | nenhum | `class_weight='balanced'`, 300 árvores |
| SVC | `StandardScaler` | `class_weight='balanced'`, kernel RBF padrão |
| Regressão Logística | `StandardScaler` | `class_weight='balanced'` |
| MultinomialNB | `MinMaxScaler` | exige valores não negativos; sem `class_weight` |

Os hiperparâmetros ficaram nos valores padrão (exceto o balanceamento).

## Experimentos

| # | Experimento | Resultado (F1 macro) | Decisão |
|---|---|---|---|
| 1 | Random Forest, 26 features | 0,697 (validação); 0,740 ± 0,029 (CV) | Ponto de partida |
| 2 | `class_weight='balanced'` | 0,701; recall de Fund. II sobe de 0,45 para 0,61 | Mantido |
| 3 | + 4 features lexicais (30) | 0,766 (validação); **0,771 ± 0,016** (CV) | Mantidas |
| 4 | + 4 features de riqueza lexical (34) | 0,745 (validação); 0,761 ± 0,019 (CV) | Descartadas (redundantes) |
| 5 | Comparação dos 4 algoritmos | SVC 0,785; RF 0,771; RL 0,742; NB 0,342 | SVC escolhido |

Pontos de aprendizado documentados:

- A validação simples sugeriu um ganho de **+0,065** com as features lexicais; a validação cruzada mostrou **+0,031**. A comparação pareada evitou reportar um ganho inflado.
- O balanceamento de classes redistribuiu os acertos (Fund. II melhorou, Médio e Superior perderam) sem mudar o F1 geral, indicando que o gargalo era a falta de informação nas features.
- Quatro features de riqueza lexical (MATTR, proporção de hapax, TTR de lemas e entropia) não melhoraram o modelo e foram descartadas.

![Validação simples contra validação cruzada](figuras_relatorio/fig09_validacao_vs_cv.png)

## Algoritmo de mutação de texto

`mudança_texto.py` gera uma versão modificada do texto de entrada, para que ela possa ser reclassificada na interface. O fluxo é:

1. análise do texto com o spaCy (classes gramaticais e dependências sintáticas);
2. **troca de gênero com concordância**: substantivos e pronomes dos dicionários (*aluno/aluna*, *ele/ela* etc.) trocam de gênero, e determinantes e adjetivos dependentes são flexionados pela árvore de dependências;
3. nova análise do texto já alterado;
4. **substituição lexical**: palavras presentes nos dicionários de sinônimos e antônimos são trocadas com probabilidade de 0,3;
5. reconstrução do texto preservando espaços e capitalização.

**Resultado:** o algoritmo executa as alterações previstas, mas **não produz textos coesos e coerentes nem altera a complexidade de forma controlada**. Os dicionários são pequenos (24 entradas de sinônimos e antônimos), os antônimos invertem o sentido, a concordância só cobre as palavras mapeadas e o sorteio não usa semente. A melhoria proposta é gerar um texto novo a partir do conteúdo do original (assunto central, temas e palavras mais usadas), com a complexidade-alvo como entrada, em vez de perturbar um texto existente. O efeito sobre a classificação não foi quantificado; o relatório descreve um protocolo de avaliação.

## Limitações

- **Médio ↔ Superior:** as features medem a **forma** do texto (estrutura, classes gramaticais, comprimento e diversidade de palavras), e não o **conteúdo** (tema, vocabulário técnico).
- **Risco de aprendizado por atalho:** as amostras sugerem fontes editoriais diferentes por nível (cabeçalhos, marcadores de fórmula). Se o modelo reconhece a fonte e não a complexidade, pode não generalizar para textos de outra origem. Esse risco não foi testado.
- **Classes pequenas:** 44 e 49 textos de teste em Fund. I e II tornam as métricas por classe imprecisas (um erro vale cerca de 2 pontos de recall).
- **Hiperparâmetros** não otimizados; SVC e Random Forest empatam tecnicamente.
- **Textos curtos ou de outro domínio:** o modelo foi treinado com textos de ~300 palavras ou mais.
- **Mutação:** efeito não quantificado.
- **Interface:** a comparação entre texto original e modificado não é simultânea (é preciso voltar ao original e classificar de novo), e o campo do texto modificado é somente leitura na prática.

## Trabalhos futuros

1. Avaliar quantitativamente o algoritmo de mutação (taxa de mudança de classe, variação das features, controles e várias sementes).
2. Testar o aprendizado por atalho com ablação de `pos_PROPN`, `punct_aspas` e `punct_parenteses` e avaliação em uma fonte não vista.
3. Combinar as 30 features com **TF-IDF** para atacar a confusão Médio ↔ Superior (contexto em que o MultinomialNB poderia funcionar).
4. **Canonicidade por sílabas**, com separação silábica real.
5. Ajuste de hiperparâmetros (`GridSearchCV`) e comparação pareada entre SVC e Random Forest.
6. Substituir a mutação por geração de texto guiada pelo conteúdo.
7. Melhorar a interface: exibir as previsões do original e do modificado lado a lado.

## Relatório

O relatório técnico completo (dados, metodologia, todos os experimentos, erros encontrados e limitações) acompanha o repositório.

## Autor

[Seu nome] · [LinkedIn / contato]

## Licença

[Definir a licença do repositório]
