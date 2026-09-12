# -*- coding: utf-8 -*-
"""13. Para onde vai o peso que entra em cada predicado, os dois extremos.

UMA FIGURA POR MISTURA, tres ao todo. Cada figura traz o predicado de MAIOR
propagation score em cima e o de MENOR embaixo, e os dois apontam para os MESMOS
tres destinos, desenhados uma vez na faixa do meio:

    outlier             a esquerda, o terminal `Class -1`
    back to the graph   ao centro, o que segue para outros predicados
    inlier              a direita, o terminal `Class 1`

A LEITURA. A raiz de cada predicado e o seu `Out Weight`, o peso total que SAI
dele. As tres setas repartem esse peso e somam cem por cento. O score fica
escrito ao lado do nome e NAO e a subtracao das duas setas coloridas, porque ele
divide as mesmas duas arestas pelo peso que ENTRA no no, que e outro total.

OS TRES DESTINOS SAO COMPARTILHADOS porque sao os mesmos nos do grafo para os
dois predicados. Um predicado chega neles por cima e o outro por baixo, entao
nenhuma seta cruza outra.

O QUE A FIGURA MOSTRA. Que os dois polos do score nao sao simetricos. O de baixo
e quase uma folha, termina a maior parte do que recebe e termina de um lado so.
O de cima e passagem, devolve a maior parte ao grafo, e o score positivo dele vem
da pureza do pouco que termina.

TIPOGRAFIA, igual a do 10_figuras_vizinhanca.py: largura fixa 6.30 in, serif
Times New Roman, corpos 11, 10.5 e 10 pt. As duas cores tem luminancia diferente
alem do matiz, entao sobrevivem a impressao em cinza, e a posicao das caixas e
constante para que a leitura nao dependa de enxergar cor.

A figura sai DOS ARTEFATOS, nunca de valores digitados.

    python 13_figura_destino_if.py
"""
import io
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

W = 6.30
FS_NO, FS_NOTA, FS_MINI, FS_IOP = 11.0, 10.5, 10.0, 9.0
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
                     "mathtext.fontset": "stix", "font.size": FS_NO})

INK, PALE, MUTED = "#1a1a1a", "#b8b8b8", "#4a4a4a"
VERDE_L, VERDE_E = "#dcece2", "#2e7d55"
VERM_L, VERM_E = "#f3dcda", "#b3352e"
CINZA_L = "#f2f2f2"

D = "Disserta\u00e7\u00e3o"
DEST = ("C:/Users/eronp/OneDrive/Documentos/GitHub/" + D + "/" + D +
        "/Latex/figuras/")

IOPS = "Inlier-Outlier Propagation Score"
CEN = ["single", "dupla", "trio"]
PISO = 0.1

XD = [17.0, 50.0, 83.0]                                   # outlier, back, inlier
COR_L = [VERM_L, CINZA_L, VERDE_L]
COR_E = [VERM_E, PALE, VERDE_E]
NOME = ["outlier", "back to the graph", "inlier"]
YM, YC, YB = 30.0, 54.0, 6.0                              # meio, cima, baixo
ALT = 4.8


def medir(fig, ax, texto, fs):
    t = ax.text(0, 0, texto, fontsize=fs)
    fig.canvas.draw()
    bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
    inv = ax.transData.inverted()
    lg = abs(inv.transform((bb.width, 0))[0] - inv.transform((0, 0))[0])
    t.set_visible(False)
    t.remove()
    return lg


def caixa(fig, ax, x, y, texto, fs, cor, borda, alt, negrito=False, nota=None):
    """Uma caixa. Com `nota`, o score entra numa segunda linha dentro dela."""
    lg = medir(fig, ax, texto, fs) + 3.2
    if nota:
        lg = max(lg, medir(fig, ax, nota, FS_IOP) + 3.2)
        alt = alt + 2.9
    ax.add_patch(FancyBboxPatch((x - lg / 2.0, y - alt / 2.0), lg, alt,
                                boxstyle="round,pad=0,rounding_size=1.0",
                                linewidth=1.0, edgecolor=borda, facecolor=cor,
                                mutation_aspect=0.30, zorder=3))
    dy = 1.5 if nota else 0.0
    ax.text(x, y + dy, texto, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if negrito else "normal", zorder=4)
    if nota:
        ax.text(x, y - 1.7, nota, ha="center", va="center", fontsize=FS_IOP,
                color=MUTED, zorder=4)
    return lg, alt


def fracoes(r):
    """As tres fracoes sao do peso que SAI do no, `Out Weight`, e somam um.

    Nao do `In Weight`. O IOP-Score divide as duas arestas de terminal pelo peso
    que ENTRA, que e definicao do Ceschin 2025, mas `In Weight` e `Out Weight`
    diferem aqui em ate 46 por cento, entao dividir as tres pelo que entra daria
    uma terceira fatia que nao e fluxo nenhum. A figura reparte o que sai, que e
    o que a terceira seta afirma; o score continua escrito ao lado do nome e nao
    e a subtracao das duas setas coloridas.
    """
    ow = float(r["Out Weight"])
    fo = float(r["To Outliers"]) / ow
    fi = float(r["To Inliers"]) / ow
    return [fo, 1.0 - fo - fi, fi]


def corte(cen):
    n = pd.read_csv("dados/if_%s_nos.csv" % cen)
    e = pd.read_csv("dados/if_%s_arestas.csv" % cen)
    tot = float(e["Weighted frequency"].sum())
    p = n[~n["Label"].astype(str).str.startswith("Class ")].copy()
    p["sh"] = 100.0 * p["In Weight"] / tot
    s = p[p["sh"] >= PISO].sort_values(IOPS, ascending=False)
    return (s.iloc[0], s.iloc[1]), (s.iloc[-2], s.iloc[-1])


def raio(fig, ax, r, y, cima):
    fr = fracoes(r)
    lab = str(r["Label"])
    lg, ah = caixa(fig, ax, 50.0, y, lab, FS_NO, "#ffffff", INK, ALT + 0.6,
                   nota="IOP = %+.4f" % float(r[IOPS]))
    y0 = y - ah / 2.0 if cima else y + ah / 2.0
    y1 = YM + ALT / 2.0 + 0.5 if cima else YM - ALT / 2.0 - 0.5
    for j in range(3):
        lw = 0.6 + 4.6 * fr[j] ** 0.62
        ax.add_patch(FancyArrowPatch((50.0, y0), (XD[j], y1), arrowstyle="-|>",
                                     mutation_scale=9, linewidth=lw,
                                     color=COR_E[j], shrinkA=0, shrinkB=0,
                                     zorder=1))
        t = 0.55
        ax.text(50.0 + (XD[j] - 50.0) * t, y0 + (y1 - y0) * t,
                "%.1f%%" % (100 * fr[j]), ha="center", va="center",
                fontsize=FS_MINI, color=COR_E[j], zorder=5,
                bbox=dict(boxstyle="round,pad=0.14", fc="#ffffff", ec="none",
                          alpha=0.95))
    return lab, fr


def desenha(par, nome):
    fig, ax = plt.subplots(figsize=(W, 3.45))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")

    a = raio(fig, ax, par[0], YC, True)
    b = raio(fig, ax, par[1], YB, False)
    for j in range(3):
        caixa(fig, ax, XD[j], YM, NOME[j], FS_NOTA, COR_L[j], COR_E[j], ALT,
              negrito=j != 1)

    fig.subplots_adjust(left=0.004, right=0.996, top=0.996, bottom=0.004)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(DEST, "%s.%s" % (nome, ext)),
                    dpi=300 if ext == "png" else None)
    plt.close(fig)

    print("  %s" % nome)
    for lab, fr in (a, b):
        print("     %-22s outlier %5.1f%%  back %5.1f%%  inlier %5.1f%%  soma %.1f%%"
              % (lab, 100 * fr[0], 100 * fr[1], 100 * fr[2], 100 * sum(fr)))


def main():
    if not os.path.isdir(DEST):
        sys.exit("destino nao encontrado: %s" % DEST)
    for cen in CEN:
        pos, neg = corte(cen)
        desenha(pos, "if-destino-%s-pos" % cen)
        desenha(neg, "if-destino-%s-neg" % cen)


if __name__ == "__main__":
    main()
