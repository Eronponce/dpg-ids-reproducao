# -*- coding: utf-8 -*-
"""14. O esquema que ensina a ler as figuras do cenario nao supervisionado.

UMA FIGURA SO, e ela NAO TEM DADO. E um esquema com valores inventados, redondos
de proposito, para que o leitor aprenda a anatomia antes de ver a primeira
figura com numero real. As figuras com dado saem do 13_figura_destino_if.py.

SEM ANOTACAO NENHUMA, por decisao do autor: o que a figura significa e explicado
no texto, nao em legenda dentro dela. O esquema so mostra a forma.

O QUE ELA MOSTRA:

  1. a forma do predicado neste cenario, uma feature e um operador, sem valor
  2. que a raiz e o peso que SAI do predicado, e que as tres fatias sao dele
  3. os tres destinos possiveis, e que as tres fracoes somam cem por cento
  4. que a espessura da seta acompanha a fracao

O IOP-SCORE APARECE SEM VALOR, como `IOP = ...`, e isso e deliberado. Ele e a
metrica do Ceschin 2025, calculada pela biblioteca, e este esquema nao tem
predicado nenhum, logo nao tem IOP real para mostrar. Escrever um numero daria a
entender que existe um IOP deste trabalho, e nao existe. As reticencias marcam
onde o valor aparece nas figuras com dado, onde ele e sempre lido da coluna
`Inlier-Outlier Propagation Score` que a ferramenta escreve.

VALORES DO ESQUEMA: 38, 52 e 10 por cento, redondos e somando cem. Nao
correspondem a predicado nenhum e nao devem ser citados como resultado.

TIPOGRAFIA, igual a do 13_figura_destino_if.py e a do 10_figuras_vizinhanca.py:
largura fixa 6.30 in, serif Times New Roman, corpos 11, 10.5 e 10 pt, mesma
paleta e as mesmas duas cores.

    python 14_figura_esquema_if.py
"""
import io
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import pandas as pd  # noqa: F401  (mantido para o mesmo ambiente dos demais)

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
NOME = "if-esquema"

FR = [0.38, 0.52, 0.10]                      # inlier, de volta, outlier
XD = [18.0, 50.0, 82.0]
COR_L = [VERDE_L, CINZA_L, VERM_L]
COR_E = [VERDE_E, PALE, VERM_E]
ROT = ["inlier", "back to the graph", "outlier"]
YR, YM = 50.0, 22.0
ALT = 5.0


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
    """A mesma caixa do 13_figura_destino_if.py, com o score na segunda linha."""
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


def chamada(ax, x, y, xt, yt, texto, ha="left"):
    """Anotacao com fio fino ligando o texto ao elemento."""
    ax.plot([x, xt], [y, yt], color=PALE, linewidth=0.7,
            linestyle=(0, (2.2, 2.2)), zorder=0)
    ax.text(xt, yt, texto, ha=ha, va="center", fontsize=FS_MINI,
            color=MUTED, style="italic", zorder=5)


def teia(ax, x, y, r=6.2):
    """Um glifo de teia: nos pequenos muito ligados entre si."""
    import math
    pts = [(x + r * math.cos(math.radians(a)) * 1.45,
            y + r * math.sin(math.radians(a)) * 0.62)
           for a in (90, 162, 234, 306, 18)]
    pts.append((x, y))
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            ax.plot([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]],
                    color=PALE, linewidth=0.55, zorder=3)
    for px, py in pts:
        ax.add_patch(Circle((px, py), 0.85, facecolor="#ffffff",
                            edgecolor=MUTED, linewidth=0.7, zorder=4))


def entradas(ax, x, ytopo, lg):
    """Setas fracas chegando no predicado, de varios lugares nao desenhados."""
    for dx, alt in ((-15.0, 9.0), (-8.0, 11.5), (0.0, 12.5),
                    (8.0, 11.5), (15.0, 9.0)):
        ax.add_patch(FancyArrowPatch((x + dx * 1.25, ytopo + alt),
                                     (x + dx * 0.42, ytopo + 0.6),
                                     arrowstyle="-|>", mutation_scale=7,
                                     linewidth=0.8, color=PALE,
                                     shrinkA=0, shrinkB=0, zorder=1))


def main():
    if not os.path.isdir(DEST):
        sys.exit("destino nao encontrado: %s" % DEST)

    fig, ax = plt.subplots(figsize=(W, 3.00))
    ax.set_xlim(0, 100)
    ax.set_ylim(4, 68)
    ax.axis("off")

    lg, ah = caixa(fig, ax, 50.0, YR, "feature >", FS_NO, "#ffffff", INK,
                   ALT + 0.6, nota="IOP = ...")

    chamada(ax, 50.0 - lg / 2.0 - 0.4, YR, 20.0, YR + 8.0,
            "reached from many predicates", ha="center")
    chamada(ax, 64.0, 36.4, 83.0, 58.0,
            "the share that goes each way", ha="center")
    chamada(ax, XD[1] + 2.0, YM - ALT / 2.0 - 0.4, 66.0, 10.0,
            "returns to the web", ha="center")

    y0 = YR - ah / 2.0
    y1 = YM + (5.4 if False else ALT / 2.0 + 0.5)
    for j in range(3):
        lw = 0.7 + 5.0 * FR[j] ** 0.62
        ax.add_patch(FancyArrowPatch((50.0, y0), (XD[j], y1), arrowstyle="-|>",
                                     mutation_scale=10, linewidth=lw,
                                     color=COR_E[j], shrinkA=0, shrinkB=0,
                                     zorder=1))
        ax.text(50.0 + (XD[j] - 50.0) * 0.54, y0 + (y1 - y0) * 0.54,
                "%.0f%%" % (100 * FR[j]), ha="center", va="center",
                fontsize=FS_NOTA, color=COR_E[j], zorder=5,
                bbox=dict(boxstyle="round,pad=0.16", fc="#ffffff", ec="none",
                          alpha=0.95))
        caixa(fig, ax, XD[j], YM, ROT[j], FS_NOTA, COR_L[j], COR_E[j], ALT,
              negrito=j != 1)

    fig.subplots_adjust(left=0.004, right=0.996, top=0.996, bottom=0.004)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(DEST, "%s.%s" % (NOME, ext)),
                    dpi=300 if ext == "png" else None)
    plt.close(fig)
    print("  %s.pdf e .png  (esquema, sem dado)" % NOME)
    print("     %.0f%% inlier, %.0f%% de volta, %.0f%% outlier, soma %.0f%%"
          % (100 * FR[0], 100 * FR[1], 100 * FR[2], 100 * sum(FR)))


if __name__ == "__main__":
    main()
