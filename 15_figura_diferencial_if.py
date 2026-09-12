# -*- coding: utf-8 -*-
"""15. A camada diferencial: os dois menores IOPS FORA dos cinco comuns.

UMA FIGURA POR MISTURA, tres ao todo, no mesmo formato do 13_figura_destino_if.py:
o predicado em cima, o de baixo, e os tres destinos compartilhados na faixa do
meio.

A REGRA DE SELECAO e a mesma nas tres e nao depende da mistura: entre os 68
predicados que passam o piso, retira-se os CINCO que sao os menores por IOPS nas
tres misturas ao mesmo tempo, e desenha-se os dois menores do que sobra. Os
cinco retirados sao identicos nos tres grafos, o que a Secao da camada comum
reporta, entao retira-los e a mesma operacao nos tres.

    rst_count >   syn_count >   fin_count >   rst_flag_number >   fin_flag_number >

O QUE MUDA em relacao a figura 13. La o par era o maior e o menor por IOPS, e a
seta grossa era a vermelha, do terminal outlier. Aqui a seta grossa e a cinza:
os predicados desta camada devolvem de 61 a 79 por cento ao grafo, contra 26 a
53 dos cinco comuns. A camada comum termina, a diferencial e passagem.

A figura sai DOS ARTEFATOS, nunca de valores digitados.

    python 15_figura_diferencial_if.py
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
PISO = 0.001
COMUM = ["rst_count >", "syn_count >", "fin_count >",
         "rst_flag_number >", "fin_flag_number >"]

XD = [17.0, 50.0, 83.0]
COR_L = [VERM_L, CINZA_L, VERDE_L]
COR_E = [VERM_E, PALE, VERDE_E]
NOME = ["outlier", "back to the graph", "inlier"]
YM, YC, YB = 30.0, 54.0, 6.0
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

    Nao do `In Weight`. O IOPS divide as duas arestas de terminal pelo peso que
    ENTRA, que e a definicao do Ceschin 2025, mas os dois totais diferem, entao
    dividir as tres pelo que entra daria uma terceira fatia que nao e fluxo
    nenhum. O score fica escrito na caixa e nao e a subtracao das duas setas.
    """
    ow = float(r["Out Weight"])
    fo = float(r["To Outliers"]) / ow
    fi = float(r["To Inliers"]) / ow
    return [fo, 1.0 - fo - fi, fi]


def corte(cen):
    """Os dois menores por IOPS entre os que passam o piso e nao sao os cinco."""
    n = pd.read_csv("dados/if_%s_nos.csv" % cen)
    p = n[~n["Label"].astype(str).str.startswith("Class ")].copy()
    p["sh"] = p["In Weight"] / p["In Weight"].sum()
    k = p[p["sh"] >= PISO]
    r = k[~k["Label"].isin(COMUM)].sort_values(IOPS).reset_index(drop=True)
    margem = float(r.iloc[2][IOPS]) - float(r.iloc[1][IOPS])
    return r.iloc[0], r.iloc[1], len(r), str(r.iloc[2]["Label"]), margem


def raio(fig, ax, r, y, cima):
    fr = fracoes(r)
    lab = str(r["Label"])
    lg, ah = caixa(fig, ax, 50.0, y, lab, FS_NO, "#ffffff", INK, ALT + 0.6,
                   nota="IOPS = %+.4f" % float(r[IOPS]))
    y0 = y - ah / 2.0 if cima else y + ah / 2.0
    y1 = YM + ALT / 2.0 + 0.5 if cima else YM - ALT / 2.0 - 0.5
    for j in range(3):
        lw = 0.6 + 4.6 * fr[j] ** 0.62
        ax.add_patch(FancyArrowPatch((50.0, y0), (XD[j], y1), arrowstyle="-|>",
                                     mutation_scale=9, linewidth=lw,
                                     color=COR_E[j], shrinkA=0, shrinkB=0,
                                     zorder=1))
        ax.text(50.0 + (XD[j] - 50.0) * 0.55, y0 + (y1 - y0) * 0.55,
                "%.1f%%" % (100 * fr[j]), ha="center", va="center",
                fontsize=FS_MINI, color=COR_E[j], zorder=5,
                bbox=dict(boxstyle="round,pad=0.14", fc="#ffffff", ec="none",
                          alpha=0.95))
    return lab, fr


def main():
    if not os.path.isdir(DEST):
        sys.exit("destino nao encontrado: %s" % DEST)
    for cen in CEN:
        a, b, n, prox, marg = corte(cen)
        fig, ax = plt.subplots(figsize=(W, 3.45))
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 62)
        ax.axis("off")
        ra = raio(fig, ax, a, YC, True)
        rb = raio(fig, ax, b, YB, False)
        for j in range(3):
            caixa(fig, ax, XD[j], YM, NOME[j], FS_NOTA, COR_L[j], COR_E[j], ALT,
                  negrito=j != 1)
        fig.subplots_adjust(left=0.004, right=0.996, top=0.996, bottom=0.004)
        nome = "if-diferencial-%s" % cen
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(DEST, "%s.%s" % (nome, ext)),
                        dpi=300 if ext == "png" else None)
        plt.close(fig)
        print("  %s   %d candidatos" % (nome, n))
        for lab, fr in (ra, rb):
            print("     %-22s outlier %5.1f%%  back %5.1f%%  inlier %5.1f%%  "
                  "soma %.1f%%"
                  % (lab, 100 * fr[0], 100 * fr[1], 100 * fr[2], 100 * sum(fr)))
        print("     descartado em seguida: %-18s margem %.4f" % (prox, marg))


if __name__ == "__main__":
    main()
