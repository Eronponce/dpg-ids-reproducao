# -*- coding: utf-8 -*-
"""3. O cenario nao supervisionado, os tres grafos e as suas leituras.

Reconstroi cada grafo a partir das arestas salvas, calcula a centralidade de
alcance e a betweenness com as MESMAS chamadas do networkx que o cenario
supervisionado usa, e reproduz as tabelas dos tres grafos, da betweenness, das
comunidades e da relacao entre centralidade e polaridade.

Nao treina nem re-executa o Isolation Forest. O rastreamento dos caminhos, que
e a parte cara, ja esta pago e vive nas arestas. Leva cerca de trinta segundos.

    python 3_isolation_forest.py
"""
import io
import sys

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
IOPS = "Inlier-Outlier Propagation Score"
CEN = ["single", "dupla", "trio"]


def grafo(cen):
    ar = pd.read_csv("dados/if_%s_arestas.csv" % cen)
    G = nx.DiGraph()
    for _, r in ar.iterrows():
        u, v, w = str(r["Source_id"]), str(r["Target_id"]), float(r["Weighted frequency"])
        if G.has_edge(u, v):
            G[u][v]["weight"] += w
        else:
            G.add_edge(u, v, weight=w)
    return G, pd.read_csv("dados/if_%s_nos.csv" % cen)


def backbone(G, q):
    w = np.array([d["weight"] for _, _, d in G.edges(data=True)])
    corte = np.quantile(w, q)
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes)
    for u, v, d in G.edges(data=True):
        if d["weight"] >= corte:
            H.add_edge(u, v, weight=d["weight"])
    return H


tabelas, polar = [], []
print("=" * 76)
print("OS TRES GRAFOS NAO SUPERVISIONADOS")
print("=" * 76)
print("%-8s %6s %8s %9s %14s %10s %10s %10s"
      % ("mistura", "nos", "arestas", "densidade", "betweenness=0", "bc maxima",
         "lrc maxima", "rho L-I"))

for cen in CEN:
    G, no = grafo(cen)
    n = G.number_of_nodes()
    # as MESMAS chamadas da implementacao supervisionada
    bc = nx.betweenness_centrality(G, k=n, normalized=True, weight="weight",
                                   endpoints=False)
    lrc = {x: nx.local_reaching_centrality(G, x, weight="weight") for x in G.nodes}
    no["_id"] = no["Node"].astype(str)
    no["bc"] = no["_id"].map(bc)
    no["lrc"] = no["_id"].map(lrc)
    no = no.dropna(subset=["lrc"])
    dens = G.number_of_edges() / (n * (n - 1))
    rho, p = spearmanr(no["lrc"], no[IOPS])
    print("%-8s %6d %8d %9.3f %8d de %2d %10.4f %10.4f %6.3f p=%.3f"
          % (cen, n, G.number_of_edges(), dens, int((no["bc"] == 0).sum()), len(no),
             no["bc"].max(), no["lrc"].max(), rho, p))
    tabelas.append((cen, G, no))
    a = no.loc[no["lrc"].idxmax()]
    b = no.loc[no[IOPS].idxmin()]
    polar.append((cen, a, b))

print("\n  Somar familias de ataque encolhe e adensa o grafo, de 62 nos a 0,842")
print("  ate 56 nos a 0,901. A betweenness zera em quase metade dos nos, o que")
print("  aciona a regra pre-registrada de nao reivindicar predicado de ponte.")

print("\n" + "=" * 76)
print("CENTRALIDADE E POLARIDADE NAO SAO A MESMA LEITURA")
print("=" * 76)
print("%-8s %-24s %8s %10s   %-24s %8s %10s"
      % ("mistura", "maior centralidade", "lrc", "iops", "iops mais negativo", "lrc", "iops"))
for cen, a, b in polar:
    print("%-8s %-24s %8.4f %10.4f   %-24s %8.4f %10.4f"
          % (cen, a["Label"][:24], a["lrc"], a[IOPS],
             b["Label"][:24], b["lrc"], b[IOPS]))
print("\n  Em nenhuma das tres o predicado mais central e o mais polarizado.")

print("\n" + "=" * 76)
print("AS COMUNIDADES COLAPSAM, E O BACKBONE NAO RESGATA")
print("=" * 76)
for cen, G, no in tabelas:
    U = G.to_undirected()
    p = [set(c) for c in nx.community.asyn_lpa_communities(U, weight="weight", seed=42)]
    maior = max(len(c) for c in p)
    print("  %-8s grafo cheio, %d comunidade(s), a maior com %d de %d nos"
          % (cen, len(p), maior, G.number_of_nodes()))
    H = backbone(G, 0.90)
    ph = [set(c) for c in nx.community.asyn_lpa_communities(H.to_undirected(),
                                                            weight="weight", seed=42)]
    ph = sorted(ph, key=len, reverse=True)
    ids = {str(r["Node"]): r for _, r in no.iterrows()}
    print("           backbone com os 10%% de arestas mais fortes, %d comunidades,"
          " a maior com %d" % (len(ph), len(ph[0])))
    for k, c in enumerate(ph[:2]):
        vals = [ids[x][IOPS] for x in c if x in ids]
        if vals and len(c) > 1:
            mm = float(np.mean(vals))
            print("             c%d, %2d nos, IOPS medio %+7.4f  ->  %s"
                  % (k + 1, len(c), mm,
                     "outlier-centric" if mm < 0 else "benign-centric"))
print("\n  Em single e dupla sobrevive uma comunidade outlier-centric. No trio, nao.")
