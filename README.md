# Reprodução dos experimentos da dissertação

Este repositório reproduz os números do capítulo de métodos e execuções da
dissertação **Decision Predicate Graphs para explicabilidade de detecção de
ataques em redes**.

São **oito comandos**. Nenhum deles precisa de GPU, de credencial ou de baixar
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

## Os oito comandos

| # | comando | o que reproduz | tempo |
|---|---|---|---|
| 1 | `python 1_random_forest.py` | desempenho da floresta e resultado por classe | ~5 s |
| 2 | `python 2_correctness.py` | as duas medidas de *Correctness* e o controle negativo por sorteio | ~1 min |
| 3 | `python 3_isolation_forest.py` | as comunidades dos três grafos não supervisionados e o *backbone* | ~1 min |
| 4 | `python 4_grafos_sem_filtro.py` | as leituras dos três grafos não supervisionados, com autoconferência | ~2 min |
| 5 | `python 5_metricas_completas.py` | a tabela por nó com *IOP-Score*, alcance e *betweenness* lado a lado | ~2 min |
| 6 | `python 6_class_bounds_rf.py` | os *class bounds* do cenário supervisionado, reexecutando o DPG | lento |
| 7 | `python 7_rankings_rf.py` | as ordenações por classe do supervisionado, por alcance e por *betweenness* | ~5 s |
| 8 | `python 8_reconstroi_grafos_if.py` | reconstrói os três grafos não supervisionados dos splits e confere contra os publicados | ~50 min por mistura |

Rode na ordem que quiser, são independentes. O sexto e o oitavo constroem
grafos do zero, e por isso são os demorados.

### Quatro comandos conferem sozinhos

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

O **quinto** confere as tabelas que gera contra o quarto. O **sexto** exige que
o grafo reconstruído tenha os mesmos nós do já publicado, senão os *bounds*
seriam de outro grafo. E o **oitavo** compara os grafos que reconstrói contra os
publicados valor por valor, *IOP-Score* em ponto flutuante incluso.

---

## Os valores que as execuções produzem

O diretório `saidas/` guarda a captura integral do que cada comando imprimiu, na
execução de 08/09/2026. O que segue é o resumo.

### Cenário supervisionado, Random Forest

Dados: 68.347 linhas, 39 features, quatro macroclasses. Partição 80/20 sem
estratificar, semente 42.

| medida | valor |
|---|---|
| acurácia | `0.905852231163131` |
| macro F1 | 0,8646 |
| balanced accuracy | 0,8605 |
| coeficiente de Matthews | 0,8571 |
| PR-AUC macro um-contra-resto | 0,9157 |

| classe | precisão | revocação | F1 | amostras |
|---|---|---|---|---|
| Benign | 0,8220 | 0,8240 | 0,8230 | 3.306 |
| DoS/DDoS | 1,0000 | 1,0000 | **1,0000** | 6.794 |
| Reconnaissance | 0,7367 | 0,7766 | 0,7561 | 2.140 |
| Spoofing | 0,9211 | 0,8413 | 0,8794 | 1.430 |

DoS/DDoS separa sem um erro e Reconnaissance fica vinte e quatro pontos abaixo.
A classe perfeita é a maior, então a acurácia de 0,9059 é sobretudo uma
afirmação sobre aquelas 6.794 amostras.

**Grafo:** 479 nós e 758 arestas, quatro comunidades, uma por classe e nenhuma
ambígua.

**Correctness 1**, correlação entre as ordenações do grafo e as do modelo, sobre
38 features presentes nos dois, todas com p < 0,001:

| ordenações comparadas | Spearman | Kendall |
|---|---|---|
| permutação × maior centralidade | 0,582 | 0,413 |
| permutação × centralidade somada | 0,645 | 0,450 |
| permutação × maior betweenness | 0,624 | 0,433 |
| impureza × maior centralidade | **0,799** | 0,619 |

A estrutura do grafo concorda mais com a medida **enviesada**, a impureza, do
que com a não enviesada, a permutação.

**Correctness 2**, cobertura e seletividade contra um piso sorteado:

| classe | predicados | cobertura | razão | cobertura do piso | razão do piso |
|---|---|---|---|---|---|
| DoS/DDoS | 395 | 0,653 | 1,03 | 0,684 | 1,03 |
| Benign | 52 | 0,700 | 1,01 | 0,624 | 1,01 |
| Spoofing | 18 | 0,625 | 0,97 | 0,611 | 0,98 |
| Reconnaissance | 10 | 0,513 | 0,81 | 0,580 | 0,91 |

Como conjunto, os predicados de uma classe são indistinguíveis do sorteio. O que
discrimina é a **ordenação**.

**Class bounds:** 77 linhas, 36 features em DoS/DDoS, 20 em Benign, 12 em
Spoofing e 9 em Reconnaissance. 36 das 77 com intervalo fechado dos dois lados.

**Ordenações por classe:** 479 predicados. Em **2 das 4 classes** o predicado
mais central por alcance e por betweenness é o mesmo; nas outras duas as
medidas apontam para predicados diferentes.

### Cenário não supervisionado, Isolation Forest

Três misturas, cinquenta árvores de isolamento, contaminação 0,01, semente 42.
Os três grafos têm 76 nós, que são 37 features com as duas direções mais os dois
nós de classe.

| | simples | dupla | tripla |
|---|---|---|---|
| arestas | 4.914 | 4.882 | 4.853 |
| densidade | 0,8621 | 0,8565 | 0,8514 |
| betweenness zero em | 27 de 76 | 20 de 76 | 31 de 76 |
| maior betweenness | 0,3220 | 0,3688 | 0,3347 |
| maior alcance | 13,9241 | 13,3610 | 14,3616 |
| onde | `syn_count >` | `rst_count >` | `rst_count >` |
| comunidades no grafo cheio | 1 | 1 | 1 |
| maior comunidade no backbone | 64 de 76 | 60 de 76 | 61 de 76 |
| Spearman alcance × IOP-Score | +0,364 (p 0,001) | +0,188 (p 0,104) | +0,378 (p 0,001) |
| menor IOP-Score | `cwr_flag_number >` −1,8891 | `syn_count >` −0,7143 | `cwr_flag_number >` −3,8610 |

A fórmula do IOP-Score, `(To Inliers − To Outliers) / In Weight`, reproduz a
coluna publicada com desvio máximo de `4,4e-16`.

Seis predicados aparecem no top-8 por IOP-Score das **três** misturas:
`cwr_flag_number >`, `fin_count >`, `fin_flag_number >`, `rst_count >`,
`rst_flag_number >` e `syn_count >`.

**Class bounds:** 74 linhas por mistura, 37 features de cada lado. Features cujo
intervalo difere entre *inlier* e *outlier*: 34 de 37 na simples, 36 de 37 na
dupla, 37 de 37 na tripla.

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
| `rf_class_bounds.csv` | o intervalo de cada feature por classe, 77 linhas em quatro comunidades | gerado pelo comando 6 |
| `rf_predicados_por_classe.csv` | quantos predicados cada classe usa de cada feature | gerado pelo comando 6 |
| `rf_cobertura_seletividade.csv` e demais `rf_controle_*`, `rf_importancia_*`, `rf_sensibilidade_*` | as tabelas de *Correctness* e seus controles | gerados pelo comando 2 |
| `if_{single,dupla,trio}_nos.csv` | métricas por nó dos três grafos não supervisionados, 76 nós, com o *IOP-Score* | saída da DPG-iForest, execuções de 04/09/2026 |
| `if_{single,dupla,trio}_arestas.csv` | as arestas dos mesmos grafos | saída da DPG-iForest |
| `if_{single,dupla,trio}_comunidades.csv` | os agrupamentos dos mesmos grafos | saída da DPG-iForest |
| `if_{single,dupla,trio}_metricas.csv` | os mesmos nós com *IOP-Score*, alcance e *betweenness* juntos | gerado pelo comando 5 |
| `if_{single,dupla,trio}_class_bounds.csv` | o intervalo de cada feature do lado *inlier* e do lado *outlier* | saída da DPG-iForest, execuções de 08/09/2026 |
| `splits_if/{single,dupla,trio}_train_BEN05_cont001.csv` | os splits de treino das três misturas, 37.170 fluxos cada | partição 70/30 com semente 42 |
| `splits_if/{single,dupla,trio}_test_fixed_BEN05.csv` | os splits de teste, 24.051 na simples e 31.542 na dupla e na tripla | idem |

Em `saidas/` está o stdout integral de cada um dos sete comandos, na
execução de 08/09/2026.

O `dataset_balanceado_label.csv` é o arquivo exato de onde saiu o grafo do
capítulo, e não uma reconstrução. É o que permite que a acurácia bata na décima
quinta casa.

---

## Sobre os *class bounds*

As cinco leituras do método existem agora nos dois cenários.

No não supervisionado elas vinham vazias até 08/09/2026, por dois defeitos
independentes no driver da `DPG-iForest`.

O primeiro classificava os caminhos comparando uma chave composta,
`sample{N}_dt{i}`, montada em `dpg/core.py:187`, contra um conjunto de índices de
amostra. A comparação nunca casava e o log registrava `inliers=0 | outliers=0`.
O segundo é que, mesmo classificados, os *bounds* só viviam em memória: nada os
escrevia em disco. Corrigidos os dois, a mistura simples passa a registrar
`inliers=1839900 | outliers=11188` e os arquivos saem.

O predicado do grafo não carregar limiar não impedia nada. Os *bounds* são
calculados sobre os **caminhos de decisão**, que preservam o valor porque vêm das
árvores, e não sobre os rótulos dos nós, que de fato só têm nome e operador.

Cada mistura foi reexecutada com os parâmetros do relatório consolidado de
04/09/2026, e só foi aceita depois de o grafo reconstruído bater rótulo a rótulo
com o publicado. Features cujo intervalo difere entre *inlier* e *outlier*:
34 de 37 na simples, 36 de 37 na dupla, 37 de 37 na tripla.

Algumas regras do lado *outlier* têm limite de um lado só. Isso é o dado, não um
defeito: há cerca de onze mil caminhos de *outlier* contra 1,8 milhão de
*inlier*, então há feature que naquele lado só aparece com um operador.

---

## O que este repositório NÃO faz, e por quê

**Reconstrói os dois grafos, e é aí que a reprodução se fecha.** O comando 6
reconstrói o grafo supervisionado com a biblioteca `dpg`, fixada em `0.1.6` no
`requirements.txt` porque a `0.2.0` existe e muda a saída. O comando 8
reconstrói os três grafos não supervisionados a partir dos splits em
`dados/splits_if/`.

A `DPG-iForest` **não é distribuída pelo PyPI**, então está embarcada em
`vendor/dpg_iforest/`, sob licença MIT. O `NOTICE.md` de lá nomeia a origem e
descreve as duas modificações feitas no driver, ambas posteriores à construção do
grafo e nenhuma alterando-o. Sem embarcá-la, metade deste repositório não seria
reproduzível: traria os grafos prontos e nenhuma forma de refazê-los.

**Não calcula nenhuma métrica de plausibilidade.** O capítulo confronta cada
afirmação do grafo com fontes escritas de segurança e relata o que cada fonte
respondeu, sem voto, sem média e sem nota. Não há número a automatizar.

**A centralidade do cenário não supervisionado é calculada aqui, não lida.** A
`DPG-iForest` devolve graus, pesos e o *IOP-Score*, e nenhuma centralidade. Os
comandos 3, 4 e 5 reconstroem cada grafo a partir das arestas salvas e aplicam as
**mesmas chamadas do `networkx`** que a implementação supervisionada usa em
`metrics/nodes.py`, parâmetro por parâmetro:

```python
nx.betweenness_centrality(G, k=len(G.nodes), normalized=True,
                          weight='weight', endpoints=False)
nx.local_reaching_centrality(G, n, weight='weight')
```

Chamadas idênticas tornam a grandeza **a mesma grandeza** nos dois cenários. Não
tornam os valores comparáveis um a um: os grafos têm tamanhos diferentes, 479 nós
contra 76, e a *betweenness* normalizada depende do tamanho. O que atravessa
entre os cenários é a **ordenação**, não o valor bruto.

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
