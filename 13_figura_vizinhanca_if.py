# -*- coding: utf-8 -*-
"""13. A vizinhanca dos dois predicados mais negativos de cada mistura.

UMA FIGURA POR MISTURA, tres ao todo. Cada uma mostra os DOIS predicados de menor
propagation score acima do piso, os TRES vizinhos mais pesados de cada um, e o
terminal `Class -1`, que e o unico no que os dois compartilham.

POR QUE NAO DIRIGIDA, e isto e o ponto da figura. No grafo bruto 99 por cento
das arestas existem nos dois sentidos, os dois sentidos correlacionam a 0,87 e a
razao mediana entre o maior e o menor e 1,46. Uma relacao de `imediatamente
depois` extraida de caminhos de arvore nao teria como ser reciproca: nao se sobe
de volta. A direcao nao carrega informacao, entao os dois sentidos sao somados e
a figura e desenhada sem seta. O peso diz co-ocorrencia, quantas vezes os dois
predicados apareceram no mesmo caminho de decisao durante o treino.

O QUE A FIGURA MOSTRA. Que os dois predicados que mais empurram para o lado
outlier NAO tem aresta entre si, em nenhuma das tres misturas. Eles nao sao uma
condicao conjunta, sao dois caminhos alternativos para o mesmo desfecho, e so se
encontram no terminal.

TIPOGRAFIA, igual a do 10_figuras_vizinhanca.py e a das figuras do Capitulo 2:
largura fixa 6.30 in, serif Times New Roman, corpos 11, 10.5 e 10 pt, mesma
paleta, caixas com rounding_size 1.0 e linewidth 0.8. A largura de cada caixa e
MEDIDA no renderer, nao estimada.

A figura sai DOS ARTEFATOS, nunca de valores digitados.

    python 13_figura_vizinhanca_if.py
"""
import io
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

W = 6.30
FS_NO, FS_NOTA, FS_MINI = 11.0, 10.5, 10.0
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
                     "mathtext.fontset": "stix", "font.size": FS_NO})

INK, NODE, LEAF, PALE, MUTED, SOFT = ("#1a1a1a", "#ffffff", "#d9d9d9",
                                      "#b8b8b8", "#4a4a4a", "#f2f2f2")

D = "Disserta\u00e7\u00e3o"
DEST = ("C:/Users/eronp/OneDrive/Documentos/GitHub/" + D + "/" + D +
        "/Latex/figuras/")

IOPS = "Inlier-Outlier Propagation Score"
CEN = [("single", "if-vizinhanca-single"),
       ("dupla", "if-vizinhanca-dupla"),
       ("trio", "if-vizinhanca-trio")]
PISO = 0.1
NVIZ = 3


def fmt(v):
    return "{:,}".format(int(round(v))).replace(",", ".")


def medir(fig, ax, texto, fs):
    """Largura do texto em unidades de dados: desenha, mede, apaga."""
    t = ax.text(0, 0, texto, fontsize=fs)
    fig.canvas.draw()
    bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
    inv = ax.transData.inverted()
    x0 = inv.transform((0, 0))[0]
    x1 = inv.transform((bb.width, 0))[0]
    t.remove()
    return abs(x1 - x0)


def dados(cen):
    e = pd.read_csv("dados/if_%s_arestas.csv" % cen)
    n = pd.read_csv("dados/if_%s_nos.csv" % cen)
    tot = float(e["Weighted frequency"].sum())
    sc = dict(zip(n["Label"].astype(str), n[IOPS]))

    p = n[~n["Label"].astype(str).str.startswith("Class ")].copy()
    p["sh"] = 100.0 * p["In Weight"] / tot
    s = p[p["sh"] >= PISO].sort_values(IOPS)
    focos = [str(s.iloc[0]["Label"]), str(s.iloc[1]["Label"])]

    und = {}
    for _, r in e.iterrows():
        u, v = str(r["Node_u_label"]), str(r["Node_v_label"])
        if u == v:
            continue
        k = (u, v) if u < v else (v, u)
        und[k] = und.get(k, 0.0) + float(r["Weighted frequency"])

    entre = und.get(tuple(sorted(focos)), 0.0)
    viz, term = {}, {}
    for f in focos:
        lista = sorted(((w, k[0] if k[1] == f else k[1])
                        for k, w in und.items() if f in k), reverse=True)
        term[f] = next((w for w, o in lista if o == "Class -1"), 0.0)
        viz[f] = [(o, w) for w, o in lista if not o.startswith("Class ")][:NVIZ]
    return focos, sc, viz, term, entre, tot


def caixa(fig, ax, x, y, texto, fs, cor, alt):
    lg = medir(fig, ax, texto, fs) + 3.2
    ax.add_patch(FancyBboxPatch((x - lg / 2.0, y - alt / 2.0), lg, alt,
                                boxstyle="round,pad=0,rounding_size=1.0",
                                linewidth=0.8, edgecolor=INK, facecolor=cor,
                                mutation_aspect=0.28, zorder=3))
    ax.text(x, y, texto, ha="center", va="center", fontsize=fs,
            color=INK, zorder=4)
    return lg


def linha(ax, x1, y1, x2, y2, peso, pmax, rotulo):
    lw = 0.7 + 3.3 * (peso / pmax) ** 0.55
    ax.plot([x1, x2], [y1, y2], color=MUTED, linewidth=lw,
            solid_capstyle="round", zorder=1)
    ax.text((x1 + x2) / 2.0, (y1 + y2) / 2.0 + 1.1, rotulo, ha="center",
            va="bottom", fontsize=FS_MINI, color=MUTED, zorder=2)


def desenha(cen, nome):
    focos, sc, viz, term, entre, tot = dados(cen)
    pmax = max(max(term.values()), max(w for f in focos for _, w in viz[f]))

    fig, ax = plt.subplots(figsize=(W, 4.15))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 66)
    ax.axis("off")

    XF = [27.0, 73.0]
    YF, YV, YT = 46.0, 60.0, 13.0
    ALT = 5.2

    for i, f in enumerate(focos):
        for j, (o, w) in enumerate(viz[f]):
            xv = XF[i] + (j - (NVIZ - 1) / 2.0) * 21.0
            linha(ax, XF[i], YF + ALT / 2.0, xv, YV - ALT / 2.0, w, pmax, fmt(w))
            caixa(fig, ax, xv, YV, o, FS_MINI, SOFT, ALT)
        linha(ax, XF[i], YF - ALT / 2.0, 50.0, YT + ALT / 2.0,
              term[f], pmax, fmt(term[f]))
        caixa(fig, ax, XF[i], YF, focos[i], FS_NO, NODE, ALT + 0.8)
        ax.text(XF[i], YF - ALT - 2.4, "score %+.4f" % sc[focos[i]],
                ha="center", va="top", fontsize=FS_NOTA, color=MUTED)

    caixa(fig, ax, 50.0, YT, "Class -1", FS_NO, LEAF, ALT + 0.8)

    ax.plot([XF[0] + 11.0, XF[1] - 11.0], [YF, YF], color=PALE,
            linewidth=0.9, linestyle=(0, (1.6, 2.6)), zorder=1)
    ax.text(50.0, YF + 1.4, "no edge", ha="center", va="bottom",
            fontsize=FS_MINI, color=PALE, style="italic")

    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(DEST, "%s.%s" % (nome, ext)),
                    dpi=300 if ext == "png" else None)
    plt.close(fig)

    print("  %s.pdf e .png" % nome)
    print("     focos %s (%+.4f) e %s (%+.4f)"
          % (focos[0], sc[focos[0]], focos[1], sc[focos[1]]))
    print("     aresta entre os dois: %s" % ("%.0f" % entre if entre else "NENHUMA"))
    for f in focos:
        print("     %-20s Class -1 %9s   %s" % (
            f, fmt(term[f]),
            ", ".join("%s %s" % (o, fmt(w)) for o, w in viz[f])))


def main():
    if not os.path.isdir(DEST):
        sys.exit("destino nao encontrado: %s" % DEST)
    for cen, nome in CEN:
        desenha(cen, nome)


if __name__ == "__main__":
    main()
