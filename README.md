# Reprodução dos experimentos da dissertação

Este repositório reproduz os números do capítulo de métodos e execuções da
dissertação **Decision Predicate Graphs para explicabilidade de detecção de
ataques em redes**.

São **quatro comandos**. Nenhum deles precisa de GPU, de credencial ou de baixar
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

## Os quatro comandos

| # | comando | o que reproduz | tempo |
|---|---|---|---|
| 1 | `python 1_random_forest.py` | desempenho da floresta e resultado por classe | ~5 s |
| 2 | `python 2_correctness.py` | as duas medidas de *Correctness* e o controle negativo por sorteio | ~1 min |
| 3 | `python 3_isolation_forest.py` | os três grafos não supervisionados **com filtro ANOVA**, de 62/60/56 nós | ~30 s |
| 4 | `python 4_grafos_sem_filtro.py` | os três grafos **sem filtro**, de 76 nós, e as leituras que saem deles | ~30 s |

Rode na ordem que quiser, são independentes.

> ⚠️ **Os comandos 3 e 4 leem grafos diferentes, e a conclusão muda entre eles.**
> O 3 usa os grafos que passaram pelo filtro ANOVA+CON, de 62, 60 e 56 nós. O 4 usa
> os construídos sobre os splits crus de 39 features, de 76 nós cada.
>
> A correlação de posto entre centralidade e polaridade dá `−0,053`, `−0,131` e
> `+0,204`, nenhuma significativa, nos grafos com filtro; e `+0,364`, `+0,188` e
> `+0,378` nos sem filtro. **Não citar um número de um conjunto como se fosse do
> outro.** O que vale nos dois é que o líder de uma leitura nunca é o da outra.

### O quarto comando também confere sozinho

Ele compara nós, arestas, *betweenness*, alcance e a correlação de posto contra os
valores publicados, nos três cenários, e sai com código 1 se algum divergir.

### O primeiro comando confere sozinho

O script 1 imprime a acurácia com quinze casas decimais e a compara com o valor
do capítulo. Se ele imprimir `CONFERE`, o ambiente reproduz o experimento.

```
  acuracia conferida contra o valor do capitulo
    obtido   0.905852231163131
    esperado 0.905852231163131
    CONFERE
```

---

## O que está em `dados/`

| arquivo | o que é | origem |
|---|---|---|
| `dataset_balanceado_label.csv` | 68.347 linhas, 39 features e o rótulo em quatro macroclasses | derivado do `Merged01.csv` do CICIoT2023 |
| `rf_grafo_nos.csv` | as métricas por nó do grafo supervisionado, 479 nós | saída da biblioteca DPG |
| `rf_grafo_arestas.csv` | as 758 arestas do mesmo grafo, com o peso de co-ocorrência | saída da biblioteca DPG |
| `rf_grafo_comunidades.json` | os agrupamentos por classe do mesmo grafo | saída da biblioteca DPG |
| `if_{single,dupla,trio}_arestas.csv` | as arestas dos três grafos não supervisionados **com filtro ANOVA**, 62/60/56 nós | saída da biblioteca DPG-iForest |
| `if_{single,dupla,trio}_nos.csv` | as métricas por nó dos mesmos grafos, com o *propagation score* | saída da biblioteca DPG-iForest |
| `if_{single,dupla,trio}_{nos,arestas,comunidades}_sem_filtro.csv` | os três grafos **sem filtro**, de 76 nós, sobre os splits crus de 39 features | saída da biblioteca DPG-iForest, execuções de 04/09/2026 |

O `dataset_balanceado_label.csv` é o arquivo exato de onde saiu o grafo do
capítulo, e não uma reconstrução. É o que permite que a acurácia bata na
décima quinta casa.

---

## O que este repositório NÃO faz, e por quê

**Não reconstrói os grafos.** Construir o grafo supervisionado leva cerca de
cinco minutos e exige a biblioteca `dpg`, e o não supervisionado exige a
`DPG-iForest`, que não está no PyPI. Os grafos vêm prontos em `dados/`, e os
scripts leem deles. Quem quiser refazer a construção encontra os parâmetros
declarados no capítulo, `perc_var = 0,001`, `decimal_threshold = 2`, modo
`aggregated_transitions` e limiar de agrupamento `0,2`.

**Não calcula as métricas de plausibilidade.** O protocolo de corroboração
contra as taxonomias de segurança exige dois avaliadores humanos independentes
e um gabarito congelado, e nenhuma dessas duas coisas é automatizável. O
capítulo declara o protocolo e registra que os seus números ainda não existem.

**A centralidade do cenário não supervisionado é calculada aqui, não lida.**
A biblioteca `DPG-iForest` não computa centralidade. O script 3 reconstrói cada
grafo a partir das arestas salvas e aplica as **mesmas chamadas do `networkx`**
que a implementação supervisionada usa, que é o que torna os dois cenários
comparáveis:

```python
nx.betweenness_centrality(G, k=len(G.nodes), normalized=True,
                          weight='weight', endpoints=False)
nx.local_reaching_centrality(G, n, weight='weight')
```

---

## Configuração do modelo

Idêntica nos três scripts, e é a do capítulo.

```python
RandomForestClassifier(n_estimators=20, max_depth=16, min_samples_split=2,
                       min_samples_leaf=2, class_weight="balanced",
                       random_state=42)
train_test_split(X, y, test_size=0.2, random_state=42)   # sem estratificar
```

O cenário não supervisionado usa cinquenta árvores de isolamento, contaminação
de 0,01, semente 42, partição 70/30, predicados de dois elementos e seleção de
features por ANOVA com corte de correlação de 0,9.

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
