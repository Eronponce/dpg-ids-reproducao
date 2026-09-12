# -*- coding: utf-8 -*-
"""16. Recalcula do dataset TODO numero medido que entrou no Capitulo 4.

Este script nao produz figura nem tabela. Ele refaz a conta e COMPARA com o
valor que esta publicado, imprimindo OK ou DIVERGE em cada linha. Serve de
verificacao: se a dissertacao e o repositorio discordarem, aparece aqui.

O que ele cobre, por secao da dissertacao:

  A  4.1  o valor do benigno nos nove predicados confrontados com as fontes
  B  4.2  os cinco predicados comuns, e as medias por classe de fluxo
  C  4.2  a mistura single, a identidade do LLC e o HTTP
  D  4.2  a mistura dupla, Rate e IAT
  E  4.2  a mistura trio, Number e ICMP

Le apenas de dados/, que esta versionado por inteiro. Nao treina modelo e nao
refaz grafo, porque essas duas coisas ja tem script proprio, o 8 e o 1.

    python 16_medidas_dissertacao.py
"""
import io
import sys

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

IOPS = "Inlier-Outlier Propagation Score"
CEN = ["single", "dupla", "trio"]
PISO = 0.001
COMUM = ["rst_count >", "syn_count >", "fin_count >",
         "rst_flag_number >", "fin_flag_number >"]

ok = falhas = 0


def ver(rotulo, obtido, esperado, tol=0.006, fmt="%.4f"):
    """Compara o recalculo com o valor publicado."""
    global ok, falhas
    bate = abs(float(obtido) - float(esperado)) <= tol
    ok, falhas = ok + bate, falhas + (not bate)
    print("   %-52s %s  calculado %s   texto %s"
          % (rotulo, "OK     " if bate else "DIVERGE",
             fmt % obtido, fmt % esperado))


def splits(cen):
    d = pd.read_csv("dados/splits_if/%s_train_BEN05_cont001.csv" % cen,
                    low_memory=False)
    return d, d[d["Label"] == "BENIGN"], d[d["Label"] != "BENIGN"]


def nos(cen):
    n = pd.read_csv("dados/if_%s_nos.csv" % cen)
    p = n[~n["Label"].astype(str).str.startswith("Class ")].copy()
    p["sh"] = p["In Weight"] / p["In Weight"].sum()
    return p[p["sh"] >= PISO].sort_values(IOPS).reset_index(drop=True)


# ===================================================================== A, 4.1
print("=" * 92)
print("A) SECAO 4.1, o valor do benigno nos nove predicados confrontados")
print("   fonte: dados/dataset_balanceado_label.csv, linhas de rotulo Benign")
d = pd.read_csv("dados/dataset_balanceado_label.csv", low_memory=False)
b = d[d["label"] == "Benign"]


def m(col):
    return pd.to_numeric(b[col], errors="coerce").replace(
        [np.inf, -np.inf], np.nan).mean()


ver("Tot sum, fracao acima de 5999.0 (DoS)",
    100 * (b["Tot sum"] > 5999.0).mean(), 39.0, tol=0.1, fmt="%.1f%%")
ver("ack_flag_number, media", m("ack_flag_number"), 0.8058)
ver("Number, media (DoS e Benign)", m("Number"), 9.9984)
ver("ack_count, media (Spoofing)", m("ack_count"), 8.0570)
ver("SSH, media (Spoofing)", m("SSH"), 0.0033)
ver("Min, media (Spoofing)", m("Min"), 130.15, tol=0.01, fmt="%.2f")
ver("syn_count, media (Reconnaissance)", m("syn_count"), 0.1405)
ver("HTTPS, media (Reconnaissance)", m("HTTPS"), 0.6950)
ver("Tot sum, media (Reconnaissance)", m("Tot sum"), 6155.66,
    tol=0.01, fmt="%.2f")
ver("HTTP, fracao em 0.05 ou abaixo (Benign)",
    100 * (b["HTTP"] <= 0.05).mean(), 84.9, tol=0.1, fmt="%.1f%%")

# =================================================================== B, 4.2.1
print()
print("=" * 92)
print("B) SECAO 4.2.1, os cinco predicados comuns")
K = {c: nos(c).set_index("Label") for c in CEN}
PUB = {
    "rst_count >":       (-0.9027, -0.6026, -0.8795),
    "syn_count >":       (-0.6562, -0.7143, -0.2789),
    "fin_count >":       (-0.5446, -0.4330, -0.2786),
    "rst_flag_number >": (-0.4034, -0.4240, -0.5122),
    "fin_flag_number >": (-0.2814, -0.3580, -0.2172),
}
for lab, esp in PUB.items():
    for c, e in zip(CEN, esp):
        ver("IOPS de %-18s em %-6s" % (lab, c), K[c].loc[lab][IOPS], e,
            tol=0.0001)
print()
print("   o conjunto dos cinco menores e o mesmo nas tres misturas:")
conj = [set(nos(c).head(5)["Label"]) for c in CEN]
print("      %s" % ("OK      identico"
                    if conj[0] == conj[1] == conj[2] == set(COMUM)
                    else "DIVERGE"))

print()
print("   Tabela das medias das cinco features por classe de fluxo")
FEAT = ["rst_count", "syn_count", "fin_count", "rst_flag_number",
        "fin_flag_number"]
TAB = {
    "BENIGN":           (0.02, 0.14, 0.11, 0.002, 0.011),
    "DDOS-HTTP_FLOOD":  (15.59, 31.33, 8.63, 0.156, 0.088),
    "DOS-HTTP_FLOOD":   (25.74, 28.65, 2.22, 0.257, 0.022),
    "DDOS-RSTFINFLOOD": (99.76, 0.03, 99.76, 0.998, 0.998),
    "DDOS-ICMP_FLOOD":  (0.01, 0.03, 0.00, 0.000, 0.000),
}
pool = {}
for c in CEN:
    dd, bb, aa = splits(c)
    pool.setdefault("BENIGN", bb)
    for nm, g in aa.groupby("Label"):
        pool.setdefault(nm, g)
for nm, esp in TAB.items():
    g = pool[nm]
    for f, e in zip(FEAT, esp):
        ver("%-18s %-16s media" % (nm, f), g[f].mean(), e, tol=0.006)

print()
print("   a identidade _count / _flag_number = Number")
dd, _, _ = splits("trio")
for a, bb_ in (("rst_count", "rst_flag_number"), ("fin_count", "fin_flag_number"),
               ("syn_count", "syn_flag_number")):
    m_ = dd[dd[bb_] > 0]
    r = np.corrcoef(m_[a] / m_[bb_], m_["Number"])[0, 1]
    ver("corr(%s / %s, Number)" % (a, bb_), r, 1.0, tol=1e-9, fmt="%.6f")

# =================================================================== C, 4.2.2
print()
print("=" * 92)
print("C) SECAO 4.2.2, a mistura single")
dd, bb, aa = splits("single")
for cen in CEN:
    x, _, _ = splits(cen)
    ver("%-6s  LLC == IPv, fracao das linhas" % cen,
        100 * np.isclose(x["LLC"], x["IPv"]).mean(), 100.0, tol=1e-9,
        fmt="%.2f%%")
    ver("%-6s  LLC + ARP == 1, fracao das linhas" % cen,
        100 * np.isclose(x["LLC"] + x["ARP"], 1.0).mean(), 100.0, tol=1e-9,
        fmt="%.2f%%")
ver("LLC, media benigno", bb["LLC"].mean(), 0.9759)
ver("LLC, media DDoS-HTTP_Flood", aa["LLC"].mean(), 0.9943)
ver("ARP, media benigno", bb["ARP"].mean(), 0.0241)
ver("ARP, media DDoS-HTTP_Flood", aa["ARP"].mean(), 0.0057)
ver("ARP, razao benigno sobre ataque, por pacote",
    bb["ARP"].mean() / aa["ARP"].mean(), 4.2, tol=0.05, fmt="%.1f")
ver("fluxos com ARP, benigno", 100 * (bb["ARP"] > 0).mean(), 16.85,
    tol=0.01, fmt="%.2f%%")
ver("fluxos com ARP, ataque", 100 * (aa["ARP"] > 0).mean(), 35.48,
    tol=0.01, fmt="%.2f%%")
ver("HTTP, media benigno", bb["HTTP"].mean(), 0.0581)
ver("HTTP, media DDoS-HTTP_Flood", aa["HTTP"].mean(), 0.7858)
ver("fluxos com HTTP, ataque", 100 * (aa["HTTP"] > 0).mean(), 86.83,
    tol=0.01, fmt="%.2f%%")
ver("fluxos com HTTP, benigno", 100 * (bb["HTTP"] > 0).mean(), 14.79,
    tol=0.01, fmt="%.2f%%")
seis = ["LLC <=", "LLC >", "ARP >", "ARP <=", "IPv <=", "IPv >"]
n1 = nos("single").set_index("Label")
print("   os seis nos que o grafo abre para esse um bit:")
for s in seis:
    print("      %-10s IOPS %+.4f" % (s, n1.loc[s][IOPS]))

# =================================================================== D, 4.2.3
print()
print("=" * 92)
print("D) SECAO 4.2.3, a mistura dupla")
dd, bb, aa = splits("dupla")
fam = {nm: g for nm, g in aa.groupby("Label")}


def med(g, f):
    v = pd.to_numeric(g[f], errors="coerce")
    return v[np.isfinite(v)].median()


ver("Rate, mediana benigno", med(bb, "Rate"), 170.67, tol=0.01, fmt="%.2f")
ver("Rate, mediana DDoS-HTTP_Flood", med(fam["DDOS-HTTP_FLOOD"], "Rate"),
    730.65, tol=0.01, fmt="%.2f")
ver("Rate, mediana DDoS-RSTFINFlood", med(fam["DDOS-RSTFINFLOOD"], "Rate"),
    30824.68, tol=0.01, fmt="%.2f")
p90 = pd.to_numeric(bb["Rate"], errors="coerce")
p90 = p90[np.isfinite(p90)].quantile(.9)
ver("Rate, ataque acima do p90 benigno",
    100 * (aa["Rate"] > p90).mean(), 59.0, tol=0.6, fmt="%.1f%%")
ver("IAT, mediana benigno", med(bb, "IAT"), 0.0067)
ver("IAT, mediana DDoS-HTTP_Flood", med(fam["DDOS-HTTP_FLOOD"], "IAT"), 0.0014)
ver("IAT, mediana DDoS-RSTFINFlood", med(fam["DDOS-RSTFINFLOOD"], "IAT"),
    0.0000)

# =================================================================== E, 4.2.4
print()
print("=" * 92)
print("E) SECAO 4.2.4, a mistura trio")
dd, bb, aa = splits("trio")
fam = {nm: g for nm, g in aa.groupby("Label")}
ver("Number, fluxos benignos em exatamente 10",
    (bb["Number"] == 10).sum(), 36771, tol=0, fmt="%.0f")
ver("Number, total de fluxos benignos", len(bb), 36798, tol=0, fmt="%.0f")
ver("Number, fluxos de ataque em exatamente 100",
    (aa["Number"] == 100).sum(), 369, tol=0, fmt="%.0f")
ver("Number, erros da regra > 50, em 37.170 linhas",
    (bb["Number"] > 50).sum() + (aa["Number"] <= 50).sum(), 2, tol=0, fmt="%.0f")
ver("Number, ataque acima de qualquer limiar de 10 a 100",
    100 * (aa["Number"] > 50).mean(), 99.46, tol=0.01, fmt="%.2f%%")
ver("Number, benigno acima de 10", 100 * (bb["Number"] > 10).mean(), 0.0,
    tol=1e-9, fmt="%.2f%%")
cb = pd.read_csv("dados/if_trio_class_bounds.csv")
lim = float(cb[(cb["feature"] == "Number") &
               (cb["lado"] == "outlier")]["limite_inferior"].iloc[0])
ver("Number, class bound do lado outlier", lim, 45.75, tol=0.01, fmt="%.2f")

# O grau do `Number >` nao e mais conferido: o paragrafo que o publicava foi
# retirado do texto. Fica so como informacao, porque explica o predicado.
n3 = nos("trio")
r = n3[n3["Label"] == "Number >"].iloc[0]
g = n3["In degree nodes"] + n3["Out degree nodes"]
print("   informativo, fora do texto: Number > tem grau %d mais %d, soma %d, "
      "contra mediana %.1f dos 68"
      % (r["In degree nodes"], r["Out degree nodes"],
         r["In degree nodes"] + r["Out degree nodes"], g.median()))

ver("ICMP, media benigno", bb["ICMP"].mean(), 0.0081)
ver("ICMP, media DDoS-ICMP_Flood", fam["DDOS-ICMP_FLOOD"]["ICMP"].mean(),
    0.9898)
ver("ICMP > 0.5, fluxos do DDoS-ICMP_Flood",
    (fam["DDOS-ICMP_FLOOD"]["ICMP"] > 0.5).sum(), 123, tol=0, fmt="%.0f")
ver("ICMP > 0.5, fluxos benignos", (bb["ICMP"] > 0.5).sum(), 0, tol=0,
    fmt="%.0f")
ver("ICMP > 0.5, fluxos das outras duas familias",
    sum((g_["ICMP"] > 0.5).sum() for nm, g_ in fam.items()
        if nm != "DDOS-ICMP_FLOOD"), 0, tol=0, fmt="%.0f")
for c, e in zip(CEN, (-0.0530, -0.0198, -0.1603)):
    ver("IOPS de ICMP > em %-6s" % c, K[c].loc["ICMP >"][IOPS], e, tol=0.0001)

print()
print("   o movimento do ICMP contra o de todos os predicados")
labs = sorted(set(K["single"].index) & set(K["dupla"].index)
              & set(K["trio"].index))
# ATENCAO AO SENTIDO. Nas duas primeiras o texto diz `larger than in N per
# cent`, que e a fracao de predicados que o ICMP SUPERA. Na terceira ele diz
# `moves it less than in N per cent of them`, que e a fracao que SUPERA o ICMP,
# ou seja o complemento. Os tres numeros do texto estao certos; o que muda e a
# direcao da comparacao, e e por isso que a terceira e conferida ao contrario.
for a, bq, e, maior in (("single", "trio", 90, True),
                        ("dupla", "trio", 87, True),
                        ("single", "dupla", 38, False)):
    dif = np.array([abs(K[bq].loc[l][IOPS] - K[a].loc[l][IOPS]) for l in labs])
    di = abs(K[bq].loc["ICMP >"][IOPS] - K[a].loc["ICMP >"][IOPS])
    val = 100 * ((dif < di).mean() if maior else (dif > di).mean())
    ver("ICMP >, %s de %s para %s"
        % ("supera" if maior else "e superado, de", a, bq), val, e,
        tol=1.5, fmt="%.0f%%")

print()
print("=" * 92)
print("%d conferencias OK, %d divergentes" % (ok, falhas))
sys.exit(1 if falhas else 0)
