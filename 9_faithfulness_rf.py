# -*- coding: utf-8 -*-
"""9. Correctness que sai da propria DPG, sem medida inventada por este trabalho.

O Capitulo 2 nao deixa reportar Coherence sozinha: ela vai sempre ao lado de uma
medida aplicavel de Correctness. Ate aqui a medida usada era a razao de
seletividade, que este trabalho calcula contando fluxos. Ela funciona, mas e
escolha nossa e nao saida do metodo.

A `DPGExplainer.evaluate_faithfulness` ja existe na biblioteca e nunca foi usada.
Ela compara a explicacao LOCAL que a DPG produz para uma amostra contra o que o
modelo de fato fez com aquela amostra. Isto e Correctness no sentido do Nauta: a
explicacao e fiel ao modelo? A medida vem do metodo, nao de nos.

A propria docstring da biblioteca avisa, e o aviso vai para o texto:

    `It does not measure ground-truth correctness unless y_true is provided, and
    the returned composite score is a heuristic summary, not a calibrated
    probability.`

Entao o escore composto NAO e reportado como nota. O que se reporta sao as partes
nomeadas, fidelidade de saida e recall e precisao de no e de aresta.

O grafo e reconstruido com a MESMA configuracao dos outros comandos e o script se
autoconfere contra o grafo publicado antes de medir. Se os rotulos divergirem ele
sai com codigo 1, porque medir fidelidade de um grafo diferente do que o Capitulo
4 discute nao serve para nada.

    python 9_faithfulness_rf.py

Leva alguns minutos, quase todos na construcao do grafo.
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

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ESPERADO = 0.905852231163131
LIMIAR_COMUNIDADE = 0.2
N_AMOSTRAS = 300
CONFIG = {
    "dpg": {
        "default": {"perc_var": 0.001, "decimal_threshold": 2, "n_jobs": -1},
        "graph_construction": {"mode": "aggregated_transitions"},
    }
}


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

    alvos = [str(c) for c in rf.classes_]
    print("\nconstruindo o DPG, perc_var=0.001 e decimal_threshold=2")
    expl = dpg.DPGExplainer(rf, feature_names=feats, target_names=alvos,
                            dpg_config=CONFIG)
    explicacao = expl.explain_global(X_tr, communities=True,
                                     community_threshold=LIMIAR_COMUNIDADE)

    # ---------- autoconferencia: e o mesmo grafo do Capitulo 4? ----------
    print("\nautoconferencia contra dados/rf_grafo_nos.csv")
    pub = sorted(str(x) for x in pd.read_csv("dados/rf_grafo_nos.csv")["Label"])
    nm = explicacao.node_metrics
    novo = sorted(str(x) for x in (nm["Label"] if isinstance(nm, pd.DataFrame)
                                   else pd.DataFrame(nm)["Label"]))
    if novo != pub:
        print("  os rotulos DIVERGEM: %d reconstruidos, %d publicados" % (len(novo), len(pub)))
        sys.exit(1)
    print("  %d rotulos, identicos ao grafo publicado" % len(novo))

    # ---------- a medida ----------
    print("\nmedindo fidelidade em %d amostras de teste" % N_AMOSTRAS)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_te), size=min(N_AMOSTRAS, len(X_te)), replace=False)
    # `return_details=True` e obrigatorio. Sem ele a funcao devolve UM float, o
    # escore composto, que e exatamente o numero que a docstring manda nao
    # reportar sozinho. As partes nomeadas so vem com os detalhes.
    r = expl.evaluate_faithfulness(X_te[idx], y_true=y_te[idx], return_details=True)

    print()
    print("=" * 72)
    print("CORRECTNESS QUE SAI DA PROPRIA DPG")
    print("=" * 72)
    if isinstance(r, tuple):
        print("  a funcao devolveu uma tupla de %d itens" % len(r))
        d = {}
        for i, parte in enumerate(r):
            if isinstance(parte, dict):
                d.update(parte)
            else:
                d["item_%d" % i] = parte
    elif isinstance(r, dict):
        d = r
    else:
        d = getattr(r, "__dict__", {"resultado": r})
    for k in sorted(d):
        v = d[k]
        if isinstance(v, (int, float, np.floating, np.integer)):
            print("  %-42s %s" % (k, "%.4f" % float(v) if v is not None else "-"))
        elif v is None:
            print("  %-42s -" % k)
        else:
            print("  %-42s %s" % (k, str(v)[:44]))

    io.open("dados/rf_faithfulness.json", "w", encoding="utf-8").write(
        json.dumps({k: (float(v) if isinstance(v, (int, float, np.floating, np.integer))
                        else str(v)) for k, v in d.items()},
                   ensure_ascii=False, indent=2))
    print("\n  gravado dados/rf_faithfulness.json")
    print("\n  AVISO DA PROPRIA BIBLIOTECA, e ele vai para o texto: o escore composto")
    print("  e um resumo heuristico e nao uma probabilidade calibrada. Reporte as")
    print("  partes nomeadas, nao o composto sozinho.")


if __name__ == "__main__":
    main()
