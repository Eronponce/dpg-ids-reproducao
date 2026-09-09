# -*- coding: utf-8 -*-
"""11. Por que a betweenness do ack_flag_number <= 0.59 e quase zero.

Duas explicacoes concorrem e so uma se sustenta.

  A INVERSAO DO PESO. O alcance entrega o peso ao networkx como
  total_weight / peso, entao aresta pesada vira distancia curta; a betweenness
  entrega o peso direto, entao aresta pesada vira distancia longa. Um corredor
  muito percorrido seria, para a betweenness, um caminho caro, e ela fugiria dele.

  A TOPOLOGIA. A betweenness conta caminhos minimos entre PARES de nos. Se so uma
  origem alcanca o no, quase nao existem pares para contar, e a conta cai por
  falta de pares e nao por causa do peso.

As duas sao testaveis, e o teste e direto: recalcular a betweenness SEM PESO.

  se a causa for a inversao, tirar o peso deve subir o no no ranking
  se for a topologia, tirar o peso quase nao muda

Resultado, e ele decide: o no sobe de 0,000075 para 0,001504, vinte vezes em
valor, e sai da posicao 356 para a 292 de 475. Continua no fundo. A causa
principal e a topologia. A inversao existe e e o efeito menor.

    python 11_betweenness_topologia.py

Leva menos de um minuto.
"""
import io
import json
import sys

import networkx as nx
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ALVOS = ["Tot sum > 5999.0", "ack_flag_number <= 0.59", "HTTP <= 0.05", "Number <= 10.5"]


def main():
    nos = pd.read_csv("dados/rf_grafo_nos.csv")
    ar = pd.read_csv("dados/rf_grafo_arestas.csv")
    G = nx.DiGraph()
    for _, e in ar.iterrows():
        G.add_edge(str(e["Node_u_label"]), str(e["Node_v_label"]), weight=float(e["Weight"]))
    print("grafo com %d nos e %d arestas" % (G.number_of_nodes(), G.number_of_edges()))

    pred = nos[~nos["Label"].astype(str).str.startswith("Class ")]
    bt_com = dict(zip(pred["Label"].astype(str), pred["Betweenness centrality"]))
    zeros = int((pred["Betweenness centrality"] == 0).sum())
    print("%d dos %d predicados tem betweenness exatamente zero\n" % (zeros, len(pred)))

    print("=" * 84)
    print("1. QUANTAS ORIGENS ALCANCAM CADA NO, E QUANTOS PARES ISSO DA")
    print("=" * 84)
    print("  %-26s %9s %10s %11s" % ("predicado", "origens", "destinos", "pares"))
    pares = {}
    for a in ALVOS:
        o, d = len(nx.ancestors(G, a)), len(nx.descendants(G, a))
        pares[a] = o * d
        print("  %-26s %9d %10d %11d" % (a, o, d, o * d))

    print()
    print("=" * 84)
    print("2. O TESTE: A BETWEENNESS SEM PESO")
    print("=" * 84)
    bt_sem = nx.betweenness_centrality(G, normalized=True, weight=None)
    ord_com = pred.sort_values("Betweenness centrality", ascending=False)["Label"].astype(str)
    pos_com = {x: i + 1 for i, x in enumerate(ord_com)}
    so_pred = [x for x in bt_sem if not x.startswith("Class ")]
    pos_sem = {x: i + 1 for i, x in enumerate(sorted(so_pred, key=lambda z: -bt_sem[z]))}

    print("  %-26s %11s %11s %9s %9s"
          % ("predicado", "com peso", "sem peso", "pos com", "pos sem"))
    for a in ALVOS:
        print("  %-26s %11.6f %11.6f %9d %9d"
              % (a, bt_com.get(a, float("nan")), bt_sem[a], pos_com[a], pos_sem[a]))

    print()
    print("=" * 84)
    print("3. CONCLUSAO")
    print("=" * 84)
    a = "ack_flag_number <= 0.59"
    razao = bt_sem[a] / bt_com[a] if bt_com[a] else float("inf")
    print("  %s" % a)
    print("     sem peso o valor sobe %.0f vezes, mas a posicao so vai de %d para %d de %d"
          % (razao, pos_com[a], pos_sem[a], len(pred)))
    print("     tem %d pares contra os %d do HTTP <= 0.05, %.0f vezes menos"
          % (pares[a], pares["HTTP <= 0.05"], pares["HTTP <= 0.05"] / pares[a]))
    print()
    print("  A causa principal e a TOPOLOGIA, nao a inversao do peso.")


if __name__ == "__main__":
    main()
