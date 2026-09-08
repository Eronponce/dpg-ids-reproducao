# -*- coding: utf-8 -*-
"""Ordenacoes por classe do cenario supervisionado.

O grafo do supervisionado ja trazia a centralidade de alcance e a betweenness no
arquivo de nos, e as comunidades no arquivo de clusters, mas as duas coisas nunca
tinham sido cruzadas. Uma centralidade so vira leitura quando se sabe DE QUE
CLASSE aquele predicado e o mais central. E o cruzamento que este comando faz.

Grava uma tabela com classe, predicado, alcance, betweenness e as duas posicoes
dentro da classe, e imprime o topo de cada uma.

Nada e reexecutado. Le os arquivos ja publicados.
"""
import io
import json
import sys

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

LRC = "Local reaching centrality"
BT = "Betweenness centrality"
TOPO = 5


def main():
    nos = pd.read_csv("dados/rf_grafo_nos.csv")
    com = json.load(io.open("dados/rf_grafo_comunidades.json", encoding="utf-8"))
    clusters = com["Clusters"]

    print("nos %d | clusters %s" % (len(nos), list(clusters.keys())))
    metrica = {str(r["Label"]): (r[LRC], r[BT]) for _, r in nos.iterrows()}

    linhas = []
    sem_metrica = 0
    for classe, preds in clusters.items():
        preds = list(preds) if not isinstance(preds, dict) else list(preds.keys())
        for p in preds:
            p = str(p)
            if p not in metrica:
                sem_metrica += 1
                continue
            lrc, bt = metrica[p]
            linhas.append({"classe": classe, "predicado": p,
                           "alcance": lrc, "betweenness": bt})

    df = pd.DataFrame(linhas)
    if sem_metrica:
        print("  %d predicados do cluster sem metrica no arquivo de nos" % sem_metrica)

    df["posicao_alcance"] = df.groupby("classe")["alcance"].rank(ascending=False,
                                                                method="min").astype(int)
    df["posicao_betweenness"] = df.groupby("classe")["betweenness"].rank(ascending=False,
                                                                        method="min").astype(int)
    df = df.sort_values(["classe", "posicao_alcance"])
    df.to_csv("dados/rf_rankings_por_classe.csv", index=False)
    print("\ndados/rf_rankings_por_classe.csv  %d linhas, %d classes"
          % (len(df), df["classe"].nunique()))

    for classe, g in df.groupby("classe"):
        print()
        print("=" * 74)
        print("### %s ###  %d predicados" % (classe, len(g)))
        print("  os %d de maior ALCANCE" % TOPO)
        for _, r in g.nsmallest(TOPO, "posicao_alcance").iterrows():
            print("     %2d  alcance %8.4f  bt %.4f   %s"
                  % (r["posicao_alcance"], r["alcance"], r["betweenness"], r["predicado"]))
        print("  os %d de maior BETWEENNESS" % TOPO)
        for _, r in g.nsmallest(TOPO, "posicao_betweenness").iterrows():
            print("     %2d  bt %.4f  alcance %8.4f   %s"
                  % (r["posicao_betweenness"], r["betweenness"], r["alcance"], r["predicado"]))

    coincide = (df["posicao_alcance"] == 1) & (df["posicao_betweenness"] == 1)
    print()
    print("classes em que o predicado mais central pelas duas medidas e o MESMO: %d de %d"
          % (int(coincide.sum()), df["classe"].nunique()))


if __name__ == "__main__":
    main()
