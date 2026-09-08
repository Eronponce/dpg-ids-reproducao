# Aviso de terceiros

Este diretório contém uma cópia da **DPG-iForest**, a implementação da extensão
do Decision Predicate Graph para Isolation Forest.

- Origem: `github.com/Math0097/DPG-iForest`
- Licença: MIT, Copyright (c) 2025 mceschin. O texto integral está em `LICENSE`.

A cópia está aqui porque a biblioteca **não é distribuída pelo PyPI**. Sem ela o
cenário não supervisionado deste trabalho não seria reproduzível: o repositório
traria os grafos prontos e nenhuma forma de refazê-los.

## O que foi modificado

A cópia não é idêntica ao upstream. As alterações estão todas em
`dpg_custom.py`, que é o driver, e nenhuma toca no cálculo do grafo.

**1. Classificação dos caminhos em inlier e outlier.** A chave do log de eventos
é composta, `sample{N}_dt{i}`, montada em `dpg/core.py:187`, com o índice da
amostra e o da árvore. Ela era comparada contra um conjunto de índices de
amostra, e a comparação nunca casava:

```python
paths_inliers = [(cid, steps) for cid, steps in paths_all if cid in inlier_ids]
```

O log registrava `[BOUNDS] paths_all=1851088 | inliers=0 | outliers=0` e os
*class bounds* saíam vazios. A correção extrai o índice da amostra da chave
antes de comparar. A mesma execução passa a registrar
`inliers=1839900 | outliers=11188`.

**2. Gravação dos bounds.** Mesmo classificados, os *bounds* só existiam em
memória: entravam no dicionário devolvido por `get_dpg_metrics` e nada os
escrevia em disco. Foi acrescentada a gravação de `dpg_class_bounds.csv` junto
dos outros CSVs de saída.

## Por que isso não altera o grafo

As duas alterações ficam **depois** da construção. O `get_dpg()` monta o grafo,
`get_dpg_node_metrics` e `get_dpg_edge_metrics` extraem as métricas, e só então
vem o bloco dos *bounds*. Não há caminho de volta.

Verificado empiricamente: os arquivos de nós, arestas e comunidades das três
misturas saem **idênticos bit a bit** entre a execução original, de 04/09/2026,
e a reexecução com as correções, de 08/09/2026, incluindo o *IOP-Score* em ponto
flutuante. O comando `8_reconstroi_grafos_if.py` refaz essa verificação.
