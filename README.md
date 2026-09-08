# Reprodução dos experimentos da dissertação

Este repositório reproduz os números do capítulo de métodos e execuções da
dissertação **Decision Predicate Graphs para explicabilidade de detecção de
ataques em redes**.

São **sete comandos**. Nenhum deles precisa de GPU, de credencial ou de baixar
nada além deste repositório.

---

## Instalação

```bash
git clone https://github.com/Eronponce/dpg-ids-reproducao.git
cd dpg-ids-reproducao
pip install -r requirements.txt
```

Python 3.10 ou mais novo.

---

## Os sete comandos

| # | comando | o que reproduz | tempo |
|---|---|---|---|
| 1 | `python 1_random_forest.py` | desempenho da floresta e resultado por classe | ~5 s |
| 2 | `python 2_correctness.py` | as duas medidas de *Correctness* e o controle negativo por sorteio | ~1 min |
| 3 | `python 3_isolation_forest.py` | as comunidades dos três grafos não supervisionados e o *backbone* | ~1 min |
| 4 | `python 4_grafos_sem_filtro.py` | as leituras dos três grafos não supervisionados, com autoconferência | ~2 min |
| 5 | `python 5_metricas_completas.py` | a tabela por nó com *IOP-Score*, alcance e *betweenness* lado a lado | ~2 min |
| 6 | `python 6_class_bounds_rf.py` | os *class bounds* do cenário supervisionado, reexecutando o DPG | lento |
| 7 | `python 7_rankings_rf.py` | as ordenações por classe do supervisionado, por alcance e por *betweenness* | ~5 s |

Rode na ordem que quiser, são independentes. O sexto é o único que constrói um
grafo do zero, e por isso é o único demorado.

### Três comandos conferem sozinhos

O **primeiro** imprime a acurácia com quinze casas decimais e a compara com o
valor do capítulo:

```
  acuracia conferida contra o valor do capitulo
    obtido   0.905852231163131
    esperado 0.905852231163131
    CONFERE
```

O **quarto** compara nós, arestas, *betweenness*, alcance e a correlação de
posto contra os valores publicados, nos três cenários, e sai com código 1 se
algum divergir.

O **quinto** confere as tabelas que gera contra o quarto, e o **sexto** exige
que o grafo reconstruído tenha os mesmos nós do já publicado, senão os *bounds*
seriam de outro grafo.

---

## Sobre o filtro de features

**Não há filtro.** Os três grafos não supervisionados são construídos sobre os
splits crus de 39 features, e têm 76 nós cada.

Uma versão anterior deste repositório trazia também grafos filtrados por ANOVA
com corte de correlação de 0,9, de 62, 60 e 56 nós. Eles foram **removidos em
08/09/2026** e não voltam. A correlação de posto entre centralidade e polaridade
invertia entre os dois conjuntos, o que tornava qualquer número ambíguo quanto à
sua origem.

---

## O que está em `dados/`

| arquivo | o que é | origem |
|---|---|---|
| `dataset_balanceado_label.csv` | 68.347 linhas, 39 features e o rótulo em quatro macroclasses | derivado do `Merged01.csv` do CICIoT2023 |
| `rf_grafo_nos.csv` | métricas por nó do grafo supervisionado, 479 nós, com alcance e *betweenness* | saída da biblioteca DPG |
| `rf_grafo_arestas.csv` | as 758 arestas do mesmo grafo, com o peso de co-ocorrência | saída da biblioteca DPG |
| `rf_grafo_comunidades.json` | os agrupamentos por classe do mesmo grafo | saída da biblioteca DPG |
| `rf_rankings_por_classe.csv` | cada predicado com sua classe e sua posição por alcance e por *betweenness* | gerado pelo comando 7 |
| `rf_cobertura_seletividade.csv` e demais `rf_controle_*`, `rf_importancia_*`, `rf_sensibilidade_*` | as tabelas de *Correctness* e seus controles | gerados pelo comando 2 |
| `if_{single,dupla,trio}_nos.csv` | métricas por nó dos três grafos não supervisionados, 76 nós, com o *IOP-Score* | saída da DPG-iForest, execuções de 04/09/2026 |
| `if_{single,dupla,trio}_arestas.csv` | as arestas dos mesmos grafos | saída da DPG-iForest |
| `if_{single,dupla,trio}_comunidades.csv` | os agrupamentos dos mesmos grafos | saída da DPG-iForest |
| `if_{single,dupla,trio}_metricas.csv` | os mesmos nós com *IOP-Score*, alcance e *betweenness* juntos | gerado pelo comando 5 |

O `dataset_balanceado_label.csv` é o arquivo exato de onde saiu o grafo do
capítulo, e não uma reconstrução. É o que permite que a acurácia bata na décima
quinta casa.

---

## O que este repositório NÃO faz, e por quê

**Não reconstrói os grafos não supervisionados.** A biblioteca `DPG-iForest` não
está no PyPI. Os três grafos vêm prontos em `dados/` e os comandos leem deles.
Quem quiser refazer a construção encontra os parâmetros abaixo. O grafo
supervisionado, esse sim, é reconstruído pelo comando 6, porque os *class
bounds* só existem com o objeto de explicação em mãos.

**Não calcula nenhuma métrica de plausibilidade.** O capítulo confronta cada
afirmação do grafo com fontes escritas de segurança e relata o que cada fonte
respondeu, sem voto, sem média e sem nota. Não há número a automatizar.

**A centralidade do cenário não supervisionado é calculada aqui, não lida.** A
`DPG-iForest` não computa centralidade. Os comandos 4 e 5 reconstroem cada grafo
a partir das arestas salvas e aplicam as **mesmas chamadas do `networkx`** que a
implementação supervisionada usa, que é o que torna os dois cenários
comparáveis:

```python
nx.betweenness_centrality(G, k=len(G.nodes), normalized=True,
                          weight='weight', endpoints=False)
nx.local_reaching_centrality(G, n, weight='weight')
```

---

## Configuração do modelo

Idêntica em todos os comandos do cenário supervisionado, e é a do capítulo.

```python
RandomForestClassifier(n_estimators=20, max_depth=16, min_samples_split=2,
                       min_samples_leaf=2, class_weight="balanced",
                       random_state=42)
train_test_split(X, y, test_size=0.2, random_state=42)   # sem estratificar
```

Parâmetros da extração do DPG supervisionado: `perc_var = 0,001`,
`decimal_threshold = 2`, modo `aggregated_transitions` e limiar de agrupamento
`0,2`.

O cenário não supervisionado usa cinquenta árvores de isolamento, contaminação
de 0,01, semente 42, partição 70/30 e predicados de dois elementos, sobre as 39
features cruas.

---

## Base de dados

CICIoT2023, do Canadian Institute for Cybersecurity da University of New
Brunswick.

> Neto, E. C. P. et al. **CICIoT2023: A Real-Time Dataset and Benchmark for
> Large-Scale Attacks in IoT Environment**. Sensors, v. 23, n. 13, 2023.

Uma observação registrada no capítulo e verificável aqui: o artigo do conjunto
de dados descreve 46 features utilizáveis e o arquivo distribuído carrega 39.
O cabeçalho de `dados/dataset_balanceado_label.csv` mostra as 39.

## Método

> Arrighi, L.; Pennella, L.; Tavares, G. M.; Barbon Junior, S. **Decision
> Predicate Graphs: Enhancing Interpretability in Tree Ensembles**. World
> Conference on Explainable Artificial Intelligence, 2024, p. 311-332.

> Ceschin, M.; Arrighi, L.; Longo, L.; Barbon Junior, S. **Extending Decision
> Predicate Graphs for Comprehensive Explanation of Isolation Forest**.
> xAI 2025, CCIS 2577, p. 271-293.
