# -*- coding: utf-8 -*-
"""9. Correctness que sai da propria DPG, sem medida inventada por este trabalho.

O Capitulo 2 nao deixa reportar Coherence sozinha: ela vai sempre ao lado de uma
medida aplicavel de Correctness.

A `DPGExplainer.evaluate_faithfulness` ja existe na biblioteca. Ela compara a
explicacao LOCAL que a DPG produz para uma amostra contra o que o modelo de fato
fez com aquela amostra. Isto e Correctness no sentido do Nauta: a explicacao e
fiel ao modelo? A medida vem do metodo, nao de nos.

A propria docstring da biblioteca avisa, e o aviso vai para o texto:

    `It does not measure ground-truth correctness unless y_true is provided, and
    the returned composite score is a heuristic summary, not a calibrated
    probability.`

Entao o escore composto NAO e reportado como nota. O que se reporta sao as partes
nomeadas, fidelidade de saida e recall e precisao de no e de aresta.

Para esta medida a DPG e construida sem poda, com perc_var = 0, de modo que
nenhum caminho de decisao da floresta e descartado. E o que o texto declara.

Um atalho, e a conferencia dele. `evaluate_faithfulness` pede a tabela de
metricas por no, e a biblioteca calcula nela o alcance e a betweenness. Nesta
DPG, que passa de 21 mil nos, isso leva horas. Essas duas colunas so entram numa
media descritiva da explicacao local. O voto, os recalls e as precisoes dependem
apenas de QUAIS nos e arestas existem no grafo. Por isso o comando entrega a
tabela com as duas colunas em zero. Antes de medir, ele confere o atalho numa
floresta pequena, onde as metricas da biblioteca saem em segundos: as cinco
partes tem de ser identicas com a tabela da biblioteca e com a tabela em zero.

O script se autoconfere contra os valores do texto e sai com codigo 1 se algo
divergir.

    python 9_faithfulness_rf.py

Leva cerca de dez minutos, quase todos na construcao do grafo.
"""
import io
import json
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import dpg

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace",
                              line_buffering=True)

ESPERADO = 0.905852231163131
N_AMOSTRAS = 300
PARTES = ["output_fidelity", "mean_node_recall", "mean_node_precision",
          "mean_edge_recall", "mean_edge_precision"]
# os valores do texto, com quatro casas
TEXTO = {"output_fidelity": 0.9767, "mean_node_recall": 0.9999,
         "mean_node_precision": 1.0, "mean_edge_recall": 0.9290,
         "mean_edge_precision": 1.0}
CONFIG = {
    "dpg": {
        "default": {"perc_var": 0.0, "decimal_threshold": 2, "n_jobs": -1},
        "graph_construction": {"mode": "aggregated_transitions"},
    }
}


def metricas_sem_centralidade(expl):
    """A tabela por no que a biblioteca montaria, com alcance e betweenness em zero.

    Mesmas colunas e mesma juncao de `NodeMetrics.extract_node_metrics`: so os nos
    que estao no grafo e na lista de rotulos.
    """
    G = expl._graph
    rotulo = dict(expl._nodes)
    return pd.DataFrame([{
        "Node": n,
        "Degree": G.in_degree(n) + G.out_degree(n),
        "In degree nodes": G.in_degree(n),
        "Out degree nodes": G.out_degree(n),
        "Betweenness centrality": 0.0,
        "Local reaching centrality": 0.0,
        "Label": rotulo[n],
    } for n in G.nodes() if n in rotulo])


def mede(rf, feats, X_tr, X_te, y_te, atalho):
    """Constroi a DPG da floresta e devolve as cinco partes da medida."""
    alvos = [str(c) for c in rf.classes_]
    expl = dpg.DPGExplainer(rf, feature_names=feats, target_names=alvos,
                            dpg_config=CONFIG)
    expl.fit(X_tr)
    if atalho:
        expl._node_metrics = metricas_sem_centralidade(expl)
        expl._node_metrics_lookup = None
    # `return_details=True` e obrigatorio. Sem ele a funcao devolve UM float, o
    # escore composto, que e exatamente o numero que a docstring manda nao
    # reportar sozinho. As partes nomeadas so vem com os detalhes.
    r = expl.evaluate_faithfulness(X_te, y_true=y_te, return_details=True)
    d = {}
    for parte in (r if isinstance(r, tuple) else [r]):
        if isinstance(parte, dict):
            d.update(parte)
    res = {k: float(d[k]) for k in PARTES}
    return res, expl._graph.number_of_nodes(), expl._graph.number_of_edges()


def main():
    print("lendo dados/dataset_balanceado_label.csv")
    df = pd.read_csv("dados/dataset_balanceado_label.csv")
    y = df["label"].values
    feats = [c for c in df.columns if c != "label"]
    X = df[feats].values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=20, max_depth=16, min_samples_split=2,
                                min_samples_leaf=2, class_weight="balanced",
                                random_state=42)
    rf.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, rf.predict(X_te))
    if abs(acc - ESPERADO) > 1e-12:
        sys.exit("floresta diferente, %.15f, esperado %.15f" % (acc, ESPERADO))
    print("  floresta conferida, acuracia %.15f" % acc)

    # ---------- autoconferencia do atalho, numa floresta pequena ----------
    print("\nconferindo o atalho numa floresta de 3 arvores e 5000 amostras")
    pequena = RandomForestClassifier(n_estimators=3, max_depth=6, random_state=42)
    pequena.fit(X_tr[:5000], y_tr[:5000])
    com_bib, nos_p, _ = mede(pequena, feats, X_tr[:5000], X_te[:100], y_te[:100], atalho=False)
    com_zero, _, _ = mede(pequena, feats, X_tr[:5000], X_te[:100], y_te[:100], atalho=True)
    if com_bib != com_zero:
        print("  as cinco partes DIVERGEM entre a tabela da biblioteca e a tabela em zero")
        sys.exit(1)
    print("  grafo de %d nos: as cinco partes sao identicas com a tabela da biblioteca"
          % nos_p)
    print("  e com a tabela em zero")

    # ---------- a medida ----------
    print("\nconstruindo o DPG sem poda, perc_var=0 e decimal_threshold=2")
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_te), size=min(N_AMOSTRAS, len(X_te)), replace=False)
    res, nos, arestas = mede(rf, feats, X_tr, X_te[idx], y_te[idx], atalho=True)

    print()
    print("=" * 72)
    print("CORRECTNESS QUE SAI DA PROPRIA DPG, em %d amostras de teste" % len(idx))
    print("=" * 72)
    print("  grafo de %d nos e %d arestas" % (nos, arestas))
    diverge = False
    for k in PARTES:
        ok = abs(res[k] - TEXTO[k]) <= 5e-5
        diverge = diverge or not ok
        print("  %-22s %.4f   texto %.4f   %s" % (k, res[k], TEXTO[k],
                                                 "CONFERE" if ok else "DIVERGE"))

    saida = {"perc_var": 0.0, "n_samples": int(len(idx)), "nos": nos, "arestas": arestas}
    saida.update(res)
    io.open("dados/rf_faithfulness.json", "w", encoding="utf-8").write(
        json.dumps(saida, ensure_ascii=False, indent=2))
    print("\n  gravado dados/rf_faithfulness.json")
    print("\n  AVISO DA PROPRIA BIBLIOTECA, e ele vai para o texto: o escore composto")
    print("  e um resumo heuristico e nao uma probabilidade calibrada. Reporte as")
    print("  partes nomeadas, nao o composto sozinho.")
    if diverge:
        sys.exit(1)


if __name__ == "__main__":
    main()
