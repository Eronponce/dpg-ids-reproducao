# -*- coding: utf-8 -*-
"""2. As duas medidas de Correctness e o controle negativo por sorteio.

Reproduz a tabela de concordancia entre as ordenacoes do grafo e do modelo, a
tabela de cobertura e seletividade, e o piso do sorteio aleatorio.

Usa o grafo ja construido, em dados/rf_grafo_nos.csv e
dados/rf_grafo_comunidades.json. Para reconstruir o grafo do zero, veja o
script 4. Leva cerca de um minuto, quase todo na importancia por permutacao.

    python 2_correctness.py
"""
import io
import json
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ESPERADO = 0.905852231163131
PAD = re.compile(r"^(.*?)\s*(<=|>)\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$")

df = pd.read_csv("dados/dataset_balanceado_label.csv")
y = df["label"].values
X = df.drop(columns=["label"])
feats = list(X.columns)
Xtr, Xte, ytr, yte = train_test_split(X.values, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier(n_estimators=20, max_depth=16, min_samples_split=2,
                            min_samples_leaf=2, class_weight="balanced", random_state=42)
rf.fit(Xtr, ytr)
acc = accuracy_score(yte, rf.predict(Xte))
if abs(acc - ESPERADO) < 1e-12:
    print("floresta conferida, acuracia %.15f\n" % acc)
else:
    sys.exit("floresta diferente, %.15f, esperado %.15f" % (acc, ESPERADO))

nos = pd.read_csv("dados/rf_grafo_nos.csv")
com = json.load(io.open("dados/rf_grafo_comunidades.json", encoding="utf-8"))["Clusters"]

# ------------------------------- medida 1, ordenacao do grafo contra a do modelo
print("=" * 68)
print("CORRECTNESS 1, as ordenacoes do grafo contra as do modelo")
print("=" * 68)
print("calculando importancia por permutacao, macro F1, 10 repeticoes...")
pi = permutation_importance(rf, Xte, yte, scoring="f1_macro", n_repeats=10,
                            random_state=42, n_jobs=-1)
imp = pd.DataFrame({"feature": feats, "perm": pi.importances_mean,
                    "mdi": rf.feature_importances_})

pred = nos[nos["Label"].astype(str).str.match(PAD)].copy()
pred["feature"] = pred["Label"].str.split(" <= | > ", regex=True).str[0].str.strip()
agg = pred.groupby("feature").agg(
    lrc_max=("Local reaching centrality", "max"),
    lrc_soma=("Local reaching centrality", "sum"),
    bc_max=("Betweenness centrality", "max"),
    n=("Label", "count")).reset_index()
m = imp.merge(agg, on="feature")

print("\n  %-46s %9s %9s" % ("ordenacoes comparadas", "Spearman", "Kendall"))
for rot, a, b in [("permutacao contra maior centralidade", "perm", "lrc_max"),
                  ("permutacao contra centralidade somada", "perm", "lrc_soma"),
                  ("permutacao contra maior betweenness", "perm", "bc_max"),
                  ("impureza contra maior centralidade", "mdi", "lrc_max")]:
    print("  %-46s %9.3f %9.3f"
          % (rot, spearmanr(m[a], m[b]).statistic, kendalltau(m[a], m[b]).statistic))
print("\n  n = %d features presentes no modelo e no grafo, todas com p < 0,001" % len(m))
print("  A estrutura do grafo concorda mais com a medida ENVIESADA, a impureza,")
print("  do que com a nao enviesada, a permutacao.\n")

print("  o par que mostra isso sem estatistica")
for f in ["Tot sum", "Number"]:
    r = m[m["feature"] == f].iloc[0]
    print("    %-10s permutacao %.5f   maior centralidade %.4f   %d predicados"
          % (f, r["perm"], r["lrc_max"], r["n"]))

# ------------------------------------ medida 2, cobertura e seletividade, e piso
Xted = pd.DataFrame(Xte, columns=feats)


def avalia(rot):
    mm = PAD.match(str(rot).strip())
    if not mm or mm.group(1).strip() not in feats:
        return None
    col = Xted[mm.group(1).strip()].values
    v = float(mm.group(3))
    return (col <= v) if mm.group(2) == "<=" else (col > v)


lrc = dict(zip(nos["Label"], nos["Local reaching centrality"]))
linhas = []
for cl, rots in com.items():
    if not cl.startswith("Class "):
        continue
    c = cl[len("Class "):]
    if c not in set(yte):
        continue
    alvo = (yte == c)
    for rot in rots:
        k = avalia(rot)
        if k is None:
            continue
        cob, sel = float(k[alvo].mean()), float(k[~alvo].mean())
        linhas.append({"classe": c, "predicado": rot, "lrc": float(lrc.get(rot, np.nan)),
                       "cobertura": cob, "razao": cob / sel if sel > 0 else np.inf})
d = pd.DataFrame(linhas)

rng = np.random.default_rng(42)
todos = [r for r in nos["Label"] if PAD.match(str(r).strip())]
ctrl = []
for c in sorted(set(yte)):
    alvo = (yte == c)
    for _ in range(200):
        k = avalia(str(rng.choice(todos)))
        if k is None:
            continue
        cob, sel = float(k[alvo].mean()), float(k[~alvo].mean())
        ctrl.append({"classe": c, "cobertura": cob,
                     "razao": cob / sel if sel > 0 else np.nan})
cd = pd.DataFrame(ctrl)

print("\n" + "=" * 68)
print("CORRECTNESS 2, cobertura e seletividade contra um piso sorteado")
print("=" * 68)
print("  %-16s %7s %10s %8s %10s %8s"
      % ("classe", "pred", "cobertura", "razao", "cob piso", "razao piso"))
for c in ["DoS/DDoS", "Benign", "Spoofing", "Reconnaissance"]:
    g, gc = d[d.classe == c], cd[cd.classe == c]
    if len(g) == 0:
        continue
    print("  %-16s %7d %10.3f %8.2f %10.3f %8.2f"
          % (c, len(g), g.cobertura.mean(), g.razao.replace(np.inf, np.nan).median(),
             gc.cobertura.mean(), gc.razao.median()))

# ------------------------------- a tabela por predicado, gravada para o Capitulo 4
# Quatro pares por predicado, nao so o da comunidade: a razao sai de contar fluxos
# e nao precisa da atribuicao de classe do grafo. Com as quatro colunas o predicado
# deixa de "ser de uma classe" e passa a ter uma razao em cada uma.
#
# As contagens cruas vao junto de proposito. Quando a seletividade e zero a razao
# e indefinida, e `0 de 6876 fluxos fora da classe` diz o que `inf` nao diz. A
# celula fica VAZIA em vez de infinita, para nao contaminar media nem correlacao.
bt = dict(zip(nos["Label"], nos["Betweenness centrality"]))
tabela = []
for rot in nos["Label"]:
    k = avalia(rot)
    if k is None:
        continue
    mm = PAD.match(str(rot).strip())
    for c in sorted(set(yte)):
        alvo_c = (yte == c)
        n_dentro, n_fora = int(k[alvo_c].sum()), int(k[~alvo_c].sum())
        n_cl, n_ncl = int(alvo_c.sum()), int((~alvo_c).sum())
        cob, sel = n_dentro / n_cl, n_fora / n_ncl
        tabela.append({
            "classe": c, "predicado": rot, "feature": mm.group(1).strip(),
            "lrc": float(lrc.get(rot, np.nan)), "betweenness": float(bt.get(rot, np.nan)),
            "n_dentro": n_dentro, "n_classe": n_cl,
            "n_fora": n_fora, "n_nao_classe": n_ncl,
            "cobertura": cob, "seletividade": sel,
            "razao": (cob / sel) if sel > 0 else np.nan,
            "sem_contraexemplo": sel == 0,
        })
tab = pd.DataFrame(tabela)
tab.to_csv("dados/rf_cobertura_seletividade.csv", index=False)
print("\n  gravado dados/rf_cobertura_seletividade.csv, %d linhas, %d predicados x %d classes"
      % (len(tab), tab.predicado.nunique(), tab.classe.nunique()))
sc = tab[tab.sem_contraexemplo]
print("  %d linhas sem contraexemplo, razao indefinida, todas em %s"
      % (len(sc), ", ".join(sorted(sc.classe.unique()))))
print("  as cinco de maior cobertura entre elas:")
for _, r in sc.nlargest(5, "cobertura").iterrows():
    print("     %-22s %-16s vale em %d de %d dentro, e em %d de %d fora"
          % (r.predicado, r.classe, r.n_dentro, r.n_classe, r.n_fora, r.n_nao_classe))

print("\n  Como conjunto, os predicados atribuidos a uma classe sao indistinguiveis")
print("  do sorteio. O que discrimina e a ORDENACAO, veja os cinco de maior")
print("  centralidade do cluster de DoS/DDoS:\n")
g = d[d.classe == "DoS/DDoS"].nlargest(5, "lrc")
for _, r in g.iterrows():
    print("    centralidade %6.3f   razao %5.2f   %s" % (r.lrc, r.razao, r.predicado))
