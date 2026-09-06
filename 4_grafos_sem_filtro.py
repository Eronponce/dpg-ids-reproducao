# -*- coding: utf-8 -*-
"""Reproduz as leituras do cenario nao supervisionado nos grafos SEM filtro.

Estes sao os grafos de 76 nos, construidos sobre os splits crus de 39 features.
NAO confundir com os de 62/60/56 nos, que passaram pelo filtro ANOVA+CON e estao
em if_{single,dupla,trio}_{nos,arestas}.csv. Sao conjuntos diferentes e a
conclusao sobre centralidade contra polaridade INVERTE entre eles.

Fonte: Isolation-forest-benchmark, results_dpg_raw/*_cont001/, execucoes de
04/09/2026. Isolation Forest com 50 estimadores, contaminacao 0,01, semente 42,
profundidade maxima 8, classe normal BENIGN.

O script confere sozinho: imprime CONFERE se reproduzir os valores publicados.
"""
import io
import sys

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import spearmanr, pearsonr

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

IOPS = "Inlier-Outlier Propagation Score"
CEN = ["single", "dupla", "trio"]

# valores publicados, para a autoconferencia
ESPERADO = {
    "single": dict(nos=76, arestas=4914, bt_zero=27, bt_max=0.3220, lrc_max=13.9241, rho=+0.364),
    "dupla":  dict(nos=76, arestas=4882, bt_zero=20, bt_max=0.3688, lrc_max=13.3610, rho=+0.188),
    "trio":   dict(nos=76, arestas=4853, bt_zero=31, bt_max=0.3347, lrc_max=14.3616, rho=+0.378),
}


def carrega(c):
    n = pd.read_csv("dados/if_%s_nos_sem_filtro.csv" % c)
    e = pd.read_csv("dados/if_%s_arestas_sem_filtro.csv" % c)
    G = nx.DiGraph()
    for _, r in e.iterrows():
        G.add_edge(r["Source_id"], r["Target_id"], weight=float(r["Weighted frequency"]))
    return n, e, G, dict(zip(n["Node"], n["Label"].astype(str)))


def metricas(G):
    """As chamadas SAO as do cenario supervisionado, metrics/nodes.py, ponderadas.
    Sem isso os dois cenarios nao sao comparaveis."""
    bt = nx.betweenness_centrality(G, k=len(G.nodes), normalized=True,
                                   weight="weight", endpoints=False)
    lrc = {v: nx.local_reaching_centrality(G, v, weight="weight") for v in G.nodes()}
    return bt, lrc


def main():
    tops = {}
    falhas = []

    for c in CEN:
        n, e, G, rot = carrega(c)
        bt, lrc = metricas(G)
        iops = dict(zip(n["Node"], n[IOPS]))
        exp = ESPERADO[c]

        print("=" * 74)
        print("### %s ###" % c.upper())

        # item 8
        dens = nx.density(G)
        print("item  8  nos %d | arestas %d | densidade %.4f"
              % (G.number_of_nodes(), G.number_of_edges(), dens))

        # item 9
        zeros = sum(1 for v in bt.values() if v == 0.0)
        print("item  9  betweenness zero em %d de %d, maior %.4f"
              % (zeros, G.number_of_nodes(), max(bt.values())))

        # item 10
        vl = max(lrc, key=lambda v: lrc[v])
        print("item 10  maior alcance %.4f  em %s" % (lrc[vl], rot[vl]))

        # item 11
        cm = list(nx.algorithms.community.asyn_lpa_communities(G.to_undirected(),
                                                              weight="weight", seed=42))
        ws = sorted((d["weight"] for _, _, d in G.edges(data=True)), reverse=True)
        corte = ws[max(0, int(0.10 * len(ws)) - 1)]
        B = nx.DiGraph((u, v, d) for u, v, d in G.edges(data=True) if d["weight"] >= corte)
        cb = sorted(nx.algorithms.community.asyn_lpa_communities(B.to_undirected(),
                                                                weight="weight", seed=42),
                    key=len, reverse=True)
        print("item 11  grafo cheio: %d comunidade(s), maior com %d nos"
              % (len(cm), max(len(x) for x in cm)))
        print("         backbone 10%%: %d arestas, %d comunidade(s), maior com %d de %d nos"
              % (B.number_of_edges(), len(cb), len(cb[0]), G.number_of_nodes()))

        # item 12
        com = [v for v in G.nodes() if v in iops]
        rho, p = spearmanr([lrc[v] for v in com], [iops[v] for v in com])
        print("item 12  Spearman LRC x IOPS  rho %+.3f  p %.3f  sobre %d nos" % (rho, p, len(com)))

        # item 13, parcial: o top-8 mais negativo desta mistura
        t8 = n.nsmallest(8, IOPS)
        tops[c] = list(t8["Label"].astype(str))
        vi = min(iops, key=lambda v: iops[v])
        print("item 13  menor IOPS  %-22s %+.4f" % (rot[vi], iops[vi]))

        # item 14
        calc = (n["To Inliers"] - n["To Outliers"]) / n["In Weight"]
        desvio = (calc - n[IOPS]).abs().max()
        s = n[n["In Weight"] > 0]
        r_pearson = pearsonr(np.log(s["In Weight"]), s[IOPS].abs())[0]
        r_spearman = spearmanr(np.log(s["In Weight"]), s[IOPS].abs())[0]
        print("item 14  formula (To Inliers - To Outliers)/In Weight, desvio maximo %.1e" % desvio)
        print("         log(In Weight) x |IOPS|:  PEARSON %+.3f   spearman %+.3f" % (r_pearson, r_spearman))

        # item 15
        top = sorted(G.edges(data=True), key=lambda x: -x[2]["weight"])[:3]
        print("item 15  transicoes mais pesadas:")
        for u, v, d in top:
            print("         %-22s -> %-16s %.0f" % (rot[u][:22], rot[v][:16], d["weight"]))

        # autoconferencia
        checks = [
            ("nos", G.number_of_nodes(), exp["nos"], 0),
            ("arestas", G.number_of_edges(), exp["arestas"], 0),
            ("betweenness zero", zeros, exp["bt_zero"], 0),
            ("maior betweenness", max(bt.values()), exp["bt_max"], 5e-5),
            ("maior alcance", lrc[vl], exp["lrc_max"], 5e-5),
            ("Spearman LRC x IOPS", rho, exp["rho"], 5e-4),
        ]
        ruins = [(k, o, x) for k, o, x, tol in checks if abs(o - x) > tol]
        if ruins:
            falhas.append((c, ruins))
            print("  >> DIVERGE em: %s" % ", ".join(k for k, _, _ in ruins))
        else:
            print("  >> CONFERE contra os valores publicados")
        print()

    # item 13, o fecho: quantos predicados aparecem nos tres top-8
    print("=" * 74)
    comuns = sorted(set(tops["single"]) & set(tops["dupla"]) & set(tops["trio"]))
    print("item 13  predicados presentes nos TRES top-8 por IOPS: %d" % len(comuns))
    for x in comuns:
        print("         %s" % x)
    if len(comuns) == 6:
        print("  >> CONFERE, sao seis")
    else:
        falhas.append(("vocabulario", [("predicados comuns", len(comuns), 6)]))
        print("  >> DIVERGE, esperado 6")

    print()
    if falhas:
        print("RESULTADO: divergencias encontradas, o ambiente NAO reproduz.")
        for c, ruins in falhas:
            for k, obtido, esperado in ruins:
                print("   %-10s %-22s obtido %s  esperado %s" % (c, k, obtido, esperado))
        return 1
    print("RESULTADO: o ambiente reproduz os itens 8 a 15.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
