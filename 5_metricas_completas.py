# -*- coding: utf-8 -*-
"""Gera a tabela completa de metricas por no dos tres grafos nao supervisionados.

Junta num arquivo so o que estava separado: o IOP-Score, que ja vinha guardado, e
a betweenness e o alcance local, que ate aqui eram recalculados a cada execucao e
descartados. Sem os tres lado a lado nao da para discutir centralidade contra
polaridade no no.

Os nomes das colunas de centralidade sao os MESMOS do arquivo do cenario
supervisionado, `Betweenness centrality` e `Local reaching centrality`, para que
as duas tabelas possam ser lidas uma contra a outra sem renomear nada.

As chamadas do networkx sao identicas as do quarto comando, ponderadas e
normalizadas, entao os valores conferem com os que aquele script publica.
"""
import io
import sys

import networkx as nx
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

IOPS = "Inlier-Outlier Propagation Score"
BT = "Betweenness centrality"
LRC = "Local reaching centrality"
CEN = ["single", "dupla", "trio"]

# do quarto comando, para conferir que nada mudou
ESPERADO = {
    "single": dict(nos=76, arestas=4914, bt_zero=27, bt_max=0.3220, lrc_max=13.9241),
    "dupla":  dict(nos=76, arestas=4882, bt_zero=20, bt_max=0.3688, lrc_max=13.3610),
    "trio":   dict(nos=76, arestas=4853, bt_zero=31, bt_max=0.3347, lrc_max=14.3616),
}


def main():
    falhas = []
    for c in CEN:
        n = pd.read_csv("dados/if_%s_nos.csv" % c)
        e = pd.read_csv("dados/if_%s_arestas.csv" % c)

        G = nx.DiGraph()
        for _, r in e.iterrows():
            G.add_edge(r["Source_id"], r["Target_id"], weight=float(r["Weighted frequency"]))

        bt = nx.betweenness_centrality(G, k=len(G.nodes), normalized=True,
                                       weight="weight", endpoints=False)
        lrc = {v: nx.local_reaching_centrality(G, v, weight="weight") for v in G.nodes()}

        n[BT] = n["Node"].map(bt)
        n[LRC] = n["Node"].map(lrc)

        ordem = ["Node", "Label", IOPS, LRC, BT] + \
                [x for x in n.columns if x not in ("Node", "Label", IOPS, LRC, BT)]
        n = n[ordem].sort_values(IOPS)
        saida = "dados/if_%s_metricas.csv" % c
        n.to_csv(saida, index=False)

        exp = ESPERADO[c]
        obt = dict(nos=G.number_of_nodes(), arestas=G.number_of_edges(),
                   bt_zero=int((n[BT] == 0).sum()), bt_max=round(n[BT].max(), 4),
                   lrc_max=round(n[LRC].max(), 4))
        ok = all(abs(obt[k] - exp[k]) < 1e-4 for k in exp)

        print("=" * 74)
        print("### %s ###  ->  %s" % (c.upper(), saida))
        print("  nos %d | arestas %d | betweenness zero em %d | maior bt %.4f | maior lrc %.4f"
              % (obt["nos"], obt["arestas"], obt["bt_zero"], obt["bt_max"], obt["lrc_max"]))
        print("  %s" % (">> CONFERE com o quarto comando" if ok else ">> DIVERGE do quarto comando"))
        if not ok:
            falhas.append(c)
            print("     esperado %s" % exp)
            print("     obtido   %s" % obt)

        print("  os cinco de IOPS mais negativo, com as centralidades ao lado:")
        for _, r in n.head(5).iterrows():
            print("     %-24s IOPS %+.4f   LRC %7.4f   BT %.4f"
                  % (r["Label"], r[IOPS], r[LRC], r[BT]))

    print()
    if falhas:
        print("DIVERGENCIA em: %s" % ", ".join(falhas))
        sys.exit(1)
    print("RESULTADO: as tres tabelas completas conferem com o quarto comando.")


if __name__ == "__main__":
    main()
