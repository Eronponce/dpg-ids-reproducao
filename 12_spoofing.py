# -*- coding: utf-8 -*-
"""12. A evidencia da subsecao de Spoofing.

A versao anterior daquela subsecao culpava o CORPUS: o benchmark nao teria como
expressar o ataque, porque o ATT&CK detecta ARP poisoning por ligacao entre
endereco e MAC e nao existe feature dessas.

Medido, isso nao se sustenta em duas frentes, e este comando produz as duas.

  1. OS DOIS PREDICADOS QUE O CORTE ESCOLHE SAO QUASE UNIVERSAIS, e a mesma
     comunidade tem um que separa e que o corte nao alcanca porque ele e o sexto
     no ranking do alcance.

  2. O INDICADOR DE ARP EXISTE. O grafo o usa em cinco predicados. Nenhum esta
     nesta comunidade, quatro deles dizem que o fluxo NAO e ARP, e nos dados o
     Spoofing tem a MENOR fracao de fluxos com ARP das quatro classes.

    python 12_spoofing.py
"""
import io
import json
import re
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ESPERADO = 0.905852231163131
PAD = re.compile(r"^(.*?)\s*(<=|>)\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$")
CLASSES = ["DoS/DDoS", "Benign", "Spoofing", "Reconnaissance"]


def main():
    df = pd.read_csv("dados/dataset_balanceado_label.csv")
    y = df["label"].values
    X = df.drop(columns=["label"])
    feats = list(X.columns)
    Xtr, Xte, ytr, yte = train_test_split(X.values, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=20, max_depth=16, min_samples_split=2,
                                min_samples_leaf=2, class_weight="balanced", random_state=42)
    rf.fit(Xtr, ytr)
    acc = accuracy_score(yte, rf.predict(Xte))
    if abs(acc - ESPERADO) > 1e-12:
        sys.exit("floresta diferente, %.15f" % acc)
    print("floresta conferida, acuracia %.15f\n" % acc)
    Xte = pd.DataFrame(Xte, columns=feats)
    alvo = (yte == "Spoofing")

    nos = pd.read_csv("dados/rf_grafo_nos.csv")
    lrc = dict(zip(nos["Label"].astype(str), nos["Local reaching centrality"]))
    j = json.load(io.open("dados/rf_grafo_comunidades.json", encoding="utf-8"))
    clusters = j["Clusters"]

    def masc(rot):
        m = PAD.match(rot)
        c, op, v = m.group(1).strip(), m.group(2), float(m.group(3))
        return (Xte[c] <= v).values if op == "<=" else (Xte[c] > v).values

    def razao(k, cl):
        a = (yte == cl)
        sel = float(k[~a].mean())
        return float(k[a].mean()) / sel if sel > 0 else float("inf")

    it = [str(x) for x in clusters["Class Spoofing"] if not str(x).startswith("Class ")]
    it = sorted(it, key=lambda q: -float(lrc[q]))

    print("=" * 92)
    print("1. OS 18 PREDICADOS DA COMUNIDADE, por ALCANCE, com a razao ao lado")
    print("=" * 92)
    print("  %2s %-26s %9s %10s %9s" % ("", "predicado", "alcance", "cobertura", "razao"))
    for i, x in enumerate(it, 1):
        k = masc(x)
        print("  %2d %-26s %9.4f %10.3f %9.2f"
              % (i, x, float(lrc[x]), float(k[alvo].mean()), razao(k, "Spoofing")))

    print()
    print("  os dois que o corte escolhe, em todas as classes:")
    for x in it[:2]:
        k = masc(x)
        print("     %-26s %s" % (x, "  ".join("%s %.3f" % (c[:6], float(k[(yte == c)].mean()))
                                              for c in CLASSES)))

    print()
    print("=" * 92)
    print("2. O QUE O `Min <= 59.0` SEPARA, e o corte nao alcanca")
    print("=" * 92)
    print("  Min e `Minimum packet length in the flow`; 60 bytes e o quadro Ethernet minimo")
    print()
    print("  %-16s %8s %9s %8s %12s" % ("classe", "min", "mediana", "max", "<= 59.0"))
    for c in CLASSES:
        s = Xte["Min"][yte == c]
        print("  %-16s %8.1f %9.1f %8.1f %11.1f%%"
              % (c, s.min(), s.median(), s.max(), 100.0 * (s <= 59.0).mean()))
    print()
    k = (Xte["Min"] <= 59.0).values
    dentro, fora = alvo & k, alvo & ~k
    print("  nos fluxos de Spoofing, a composicao com e sem o predicado:")
    print("  %-12s %14s %14s %8s" % ("feature", "com Min<=59", "sem", "razao"))
    for f in ["ARP", "ICMP", "DNS", "TCP", "HTTPS", "UDP"]:
        a, b = Xte[f][dentro].mean(), Xte[f][fora].mean()
        print("  %-12s %14.4f %14.4f %8.2f" % (f, a, b, (a / b) if b else float("inf")))

    print()
    print("=" * 92)
    print("3. O INDICADOR DE ARP EXISTE, E NAO MARCA A CLASSE")
    print("=" * 92)
    dono = {}
    for kk, v in clusters.items():
        for x in v:
            dono[str(x)] = kk.replace("Class ", "")
    arp = [x for x in lrc if PAD.match(x) and PAD.match(x).group(1).strip() == "ARP"]
    print("  %d predicados de ARP no grafo:" % len(arp))
    for x in sorted(arp, key=lambda q: -float(lrc[q])):
        print("     %-18s alcance %8.4f   comunidade %s" % (x, float(lrc[x]), dono.get(x, "?")))
    print("     dos %d, %d dizem que o fluxo NAO e ARP"
          % (len(arp), sum(1 for x in arp if " <= " in x)))
    print("     na comunidade Spoofing: %d" % sum(1 for x in arp if dono.get(x) == "Spoofing"))
    print()
    print("  %-16s %14s %16s" % ("classe", "media de ARP", "% com ARP > 0"))
    for c in CLASSES:
        s = Xte["ARP"][yte == c]
        print("  %-16s %14.4f %15.1f%%" % (c, s.mean(), 100.0 * (s > 0).mean()))
    k = (Xte["ARP"] > 0.0).values
    print()
    print("  `ARP > 0.0` como seletor do Spoofing: razao %.2f" % razao(k, "Spoofing"))


if __name__ == "__main__":
    main()
