# -*- coding: utf-8 -*-
"""1. A floresta explicada e o seu desempenho.

Uma execucao, a mesma do capitulo. Reproduz a tabela de desempenho e a tabela
por classe, e confere a acuracia contra o valor declarado. Leva cerca de cinco
segundos.

    python 1_random_forest.py
"""
import io
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, classification_report,
                             f1_score, matthews_corrcoef)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ESPERADO = 0.905852231163131

print("lendo dados/dataset_balanceado_label.csv")
df = pd.read_csv("dados/dataset_balanceado_label.csv")
y = df["label"].values
X = df.drop(columns=["label"]).values
print("  %d linhas, %d features, classes %s\n" % (len(df), X.shape[1], sorted(set(y))))

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier(n_estimators=20, max_depth=16, min_samples_split=2,
                            min_samples_leaf=2, class_weight="balanced", random_state=42)
rf.fit(X_tr, y_tr)
pred = rf.predict(X_te)
acc = accuracy_score(y_te, pred)

print("=" * 64)
print("A FLORESTA EXPLICADA, 80/20 sem estratificar, semente 42")
print("=" * 64)
print("  acuracia conferida contra o valor do capitulo")
print("    obtido   %.15f" % acc)
print("    esperado %.15f" % ESPERADO)
print("    %s\n" % ("CONFERE" if abs(acc - ESPERADO) < 1e-12 else "DIVERGE"))

classes = list(rf.classes_)
prauc = average_precision_score(label_binarize(y_te, classes=classes),
                                rf.predict_proba(X_te), average="macro")

print("  Tabela de desempenho")
for nome, v in [("acuracia", acc),
                ("macro F1", f1_score(y_te, pred, average="macro")),
                ("balanced accuracy", balanced_accuracy_score(y_te, pred)),
                ("coeficiente de Matthews", matthews_corrcoef(y_te, pred)),
                ("PR-AUC macro um-contra-resto", prauc)]:
    print("    %-30s %.4f" % (nome, v))

print("\n  Tabela por classe")
rep = classification_report(y_te, pred, output_dict=True, zero_division=0)
print("    %-16s %9s %9s %9s %8s" % ("classe", "precisao", "revocacao", "F1", "amostras"))
for c in classes:
    r = rep[str(c)]
    print("    %-16s %9.4f %9.4f %9.4f %8d"
          % (c, r["precision"], r["recall"], r["f1-score"], int(r["support"])))

print("\n  DoS/DDoS separa sem um erro e Reconnaissance fica em 0,7561, vinte e")
print("  quatro pontos de diferenca, e a classe perfeita e a maior. A acuracia")
print("  de 0,9059 e sobretudo uma afirmacao sobre aquelas 6.794 amostras.")
