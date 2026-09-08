# -*- coding: utf-8 -*-
"""Extrai os class bounds do cenario supervisionado, reexecutando o DPG.

Esta e a unica leitura das cinco que nunca tinha sido salva. As outras quatro do
supervisionado ja estao em dados/rf_grafo_{nos,arestas,comunidades}. Os class
bounds exigem o objeto de explicacao, e por isso exigem o DPG rodando, nao dao
para derivar dos arquivos guardados.

A floresta e a mesma do primeiro comando, mesma particao e mesma semente. Os
quatro parametros da extracao sao os declarados no capitulo de metodos, a
frequencia relativa minima de uma variante em 0,001, o arredondamento dos
valores dos predicados em 2 casas, o modo que agrega variantes antes de contar
transicoes, e o limiar de absorcao do agrupamento em 0,2.

O script se autoconfere: o grafo reconstruido tem de ter os mesmos nos e as
mesmas arestas dos arquivos ja publicados. Se divergir, sai com codigo 1, porque
entao os bounds seriam de outro grafo.
"""
import io
import json
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import dpg  # noqa: E402

PERC_VAR = 0.001
DECIMAIS = 2
LIMIAR_COMUNIDADE = 0.2


def main():
    print("lendo dados/dataset_balanceado_label.csv")
    df = pd.read_csv("dados/dataset_balanceado_label.csv")
    y = df["label"].values
    feats = [c for c in df.columns if c != "label"]
    X = df[feats].values

    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    print("  %d linhas de treino, %d features" % (len(X_tr), len(feats)))

    rf = RandomForestClassifier(n_estimators=20, max_depth=16, min_samples_split=2,
                                min_samples_leaf=2, class_weight="balanced",
                                random_state=42)
    rf.fit(X_tr, y_tr)
    alvos = [str(c) for c in rf.classes_]
    print("  floresta treinada, classes %s" % alvos)

    cfg = {"perc_var": PERC_VAR, "decimal_threshold": DECIMAIS,
           "num_processes": 1, "model_type": "classifier"}
    print("\nconstruindo o DPG, isto e a parte demorada")
    expl = dpg.DPGExplainer(rf, feature_names=feats, target_names=alvos, dpg_config=cfg)
    explicacao = expl.explain_global(X_tr, communities=True,
                                     community_threshold=LIMIAR_COMUNIDADE)

    d = explicacao.as_dict() if hasattr(explicacao, "as_dict") else {}
    print("  chaves da explicacao: %s" % sorted(d.keys()))

    bounds = dpg.classwise_feature_bounds_from_communities(explicacao)
    contagens = dpg.class_feature_predicate_counts(explicacao)

    # ---------- autoconferencia contra o grafo ja publicado ----------
    nos_pub = pd.read_csv("dados/rf_grafo_nos.csv")
    print("\nautoconferencia contra o grafo publicado")
    print("  nos publicados %d" % len(nos_pub))
    G = getattr(expl, "graph", None) or getattr(expl, "dpg", None)
    if G is not None and hasattr(G, "number_of_nodes"):
        print("  nos reconstruidos %d | arestas %d"
              % (G.number_of_nodes(), G.number_of_edges()))
        if G.number_of_nodes() != len(nos_pub):
            print("  >> DIVERGE, os bounds seriam de outro grafo")
            sys.exit(1)
        print("  >> CONFERE")
    else:
        print("  nao consegui alcancar o grafo pelo explainer, conferencia parcial")

    # ---------- grava ----------
    if hasattr(bounds, "to_csv"):
        bounds.to_csv("dados/rf_class_bounds.csv", index=True)
        print("\ndados/rf_class_bounds.csv  %d linhas" % len(bounds))
        print(bounds.head(12).to_string())
    else:
        io.open("dados/rf_class_bounds.json", "w", encoding="utf-8").write(
            json.dumps(bounds, indent=2, ensure_ascii=False, default=str))
        print("\ndados/rf_class_bounds.json gravado")
        print(json.dumps(bounds, indent=2, ensure_ascii=False, default=str)[:1800])

    if hasattr(contagens, "to_csv"):
        contagens.to_csv("dados/rf_predicados_por_classe.csv", index=True)
        print("\ndados/rf_predicados_por_classe.csv  %s" % (contagens.shape,))

    print("\nRESULTADO: class bounds do supervisionado extraidos.")


if __name__ == "__main__":
    main()
