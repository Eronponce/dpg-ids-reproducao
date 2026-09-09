# -*- coding: utf-8 -*-
"""10. As arvores de CO-OCORRENCIA dos predicados discutidos no Capitulo 4.

Uma figura por classe, duas arvores em cada, uma por predicado discutido.

A arvore cresce da esquerda para a direita. A raiz e o predicado discutido; os
ramos sao as arestas que saem dele, com o peso em cima; e a partir do ramo mais
pesado a arvore segue, sempre pela aresta mais pesada, ate um no de classe ou ate
esgotar a profundidade. O caminho mais pesado sai em traco cheio e escuro.

O peso e a CO-OCORRENCIA: quantas vezes os dois predicados foram satisfeitos em
sequencia durante o treino. Nao tem relacao com class bounds, que sao intervalos
por feature e por classe e vivem em outro arquivo.

O que ENTRA aparece como anotacao a esquerda da raiz, porque uma arvore tem uma
raiz so, mas saber se algo entra e o que separa uma RAIZ do grafo de um no de
passagem.

DESENHO EM UNIDADES DE CARACTERE. A largura de uma caixa e o numero de
caracteres do rotulo, e o tamanho da figura e derivado da extensao ocupada. Assim
o texto cabe por construcao, em vez de depender de um fator chutado que muda
quando os limites do eixo mudam.

A figura sai DOS ARTEFATOS, nunca de valores digitados.

    python 10_figuras_vizinhanca.py
"""
import io
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

CLASSES = ["DoS/DDoS", "Benign", "Spoofing", "Reconnaissance"]
SAIDA = "figuras"

FS = 6.4                        # corpo, em pontos
CHAR_IN = 0.6 * FS / 72.0       # largura de um caractere monoespacado, em polegadas
PAD = 3.2                       # folga da caixa, em caracteres
ALT = 2.0                       # altura da caixa, em caracteres
PASSO_Y = 4.4                   # distancia vertical entre irmaos
PASSO_X = 12.0                   # folga horizontal entre niveis
PROF = 2                        # passos extras da espinha, alem do primeiro filho
NOTA_X = 30.0                   # espaco reservado a esquerda para a anotacao

TINTA = "#1a1a1a"
CINZA = "#9a9a96"
FOCO = "#d8e6ec"
BORDA_FOCO = "#2b5d70"
VIZ = "#f5f5f3"
BORDA_VIZ = "#bcbcb8"
CL_FUNDO = "#eae3d2"
CL_BORDA = "#8a7c56"


def fmt(v):
    return "{:,}".format(int(v)).replace(",", ".")


def larg(t):
    return len(t) + PAD


def caixa(ax, x, y, texto, foco=False):
    if texto.startswith("Class "):
        fc, ec, lw = CL_FUNDO, CL_BORDA, 1.0
    elif foco:
        fc, ec, lw = FOCO, BORDA_FOCO, 1.2
    else:
        fc, ec, lw = VIZ, BORDA_VIZ, 0.7
    w = larg(texto)
    ax.add_patch(FancyBboxPatch((x, y - ALT / 2), w, ALT,
                                boxstyle="round,pad=0.1,rounding_size=0.4",
                                linewidth=lw, edgecolor=ec, facecolor=fc, zorder=3))
    ax.text(x + w / 2, y, texto, ha="center", va="center", fontsize=FS,
            family="monospace", color=TINTA, zorder=4)
    return w


def ramo(ax, x1, y1, x2, y2, peso, forte):
    cor = TINTA if forte else CINZA
    lw = 1.3 if forte else 0.7
    xm = x1 + (x2 - x1) * 0.45
    ax.plot([x1, xm, xm, x2], [y1, y1, y2, y2], color=cor, linewidth=lw,
            zorder=2, solid_capstyle="round", solid_joinstyle="round")
    ax.plot([x2 - 0.7, x2, x2 - 0.7], [y2 + 0.45, y2, y2 - 0.45],
            color=cor, linewidth=lw, zorder=2)
    ax.text(x2 - 0.9, y2 + 1.05, fmt(peso), ha="right", va="bottom",
            fontsize=FS - 0.6, color=cor, zorder=5,
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                      edgecolor="none", alpha=0.93))


def arvore(ax, alvo, ar, y0):
    filhos = ar[ar["Node_u_label"] == alvo].sort_values("Weight", ascending=False)
    n = len(filhos)
    wr = caixa(ax, 0.0, y0, alvo, foco=True)

    ent = ar[ar["Node_v_label"] == alvo]
    if len(ent) == 0:
        nota = "raiz do grafo\nnada entra"
    elif len(ent) == 1:
        nota = "entra %s de\n%s" % (fmt(ent.iloc[0]["Weight"]), ent.iloc[0]["Node_u_label"])
    else:
        nota = "entram %d arestas\npeso total %s" % (len(ent), fmt(ent["Weight"].sum()))
    ax.text(-1.6, y0, nota, ha="right", va="center", fontsize=FS - 0.4,
            style="italic", color="#6f6f6f", zorder=4, linespacing=1.6)

    ys = [y0 + (i - (n - 1) / 2.0) * PASSO_Y for i in range(n)]
    x1 = wr + PASSO_X
    maior = filhos["Weight"].max() if n else 0
    y_p = no_p = None
    w_p = 0
    xmax = x1
    for (_, e), yy in zip(filhos.iterrows(), ys):
        rot = str(e["Node_v_label"])
        forte = e["Weight"] == maior
        w = caixa(ax, x1, yy, rot)
        ramo(ax, wr, y0, x1, yy, e["Weight"], forte)
        xmax = max(xmax, x1 + w)
        if forte and y_p is None:
            y_p, no_p, w_p = yy, rot, w

    x, y, atual, w = x1, y_p, no_p, w_p
    vistos = {alvo}
    for _ in range(PROF):
        if atual is None or atual in vistos or str(atual).startswith("Class "):
            break
        vistos.add(atual)
        prox = ar[ar["Node_u_label"] == atual]
        if len(prox) == 0:
            break
        e = prox.nlargest(1, "Weight").iloc[0]
        rot = str(e["Node_v_label"])
        if rot in vistos:            # nao repete no: uma arvore nao volta
            break
        xn = x + w + PASSO_X
        wn = caixa(ax, xn, y, rot)
        ramo(ax, x + w, y, xn, y, e["Weight"], True)
        x, w, atual = xn, wn, rot
        xmax = max(xmax, xn + wn)

    todos = ys + [y0]
    return min(todos) - ALT, max(todos) + ALT, xmax


def main():
    D = "dados"
    ar = pd.read_csv(os.path.join(D, "rf_grafo_arestas.csv"))
    nos = pd.read_csv(os.path.join(D, "rf_grafo_nos.csv"))
    j = json.load(io.open(os.path.join(D, "rf_grafo_comunidades.json"), encoding="utf-8"))
    clusters = j["Clusters"]
    lrc = dict(zip(nos["Label"].astype(str), nos["Local reaching centrality"]))
    if not os.path.isdir(SAIDA):
        os.makedirs(SAIDA)

    for c in CLASSES:
        it = [str(x) for x in clusters["Class " + c]
              if not str(x).startswith("Class ") and str(x) in lrc]
        alvos = sorted(it, key=lambda x: -float(lrc[x]))[:2]

        alturas = [max(1, len(ar[ar["Node_u_label"] == a])) * PASSO_Y + 3.0 for a in alvos]
        fig, ax = plt.subplots()
        y = 0.0
        centros = []
        for h in alturas:
            centros.append(y + h / 2)
            y += h + 4.6
        xmax = 0.0
        for a, yc in zip(alvos, reversed(centros)):
            _, _, xm = arvore(ax, a, ar, yc)
            xmax = max(xmax, xm)

        x0, x1 = -NOTA_X, xmax + 1.5
        y0, y1 = -2.0, y - 2.6
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.axis("off")
        fig.set_size_inches((x1 - x0) * CHAR_IN, (y1 - y0) * CHAR_IN)
        fig.tight_layout(pad=0.1)
        nome = os.path.join(SAIDA, "rf_vizinhanca_%s.pdf" % c.replace("/", "").lower())
        fig.savefig(nome, bbox_inches="tight")
        plt.close(fig)

        print("  %-16s %s   %.1f x %.1f pol"
              % (c, nome, (x1 - x0) * CHAR_IN, (y1 - y0) * CHAR_IN))
        for a in alvos:
            e_ = ar[ar["Node_v_label"] == a]
            s_ = ar[ar["Node_u_label"] == a]
            print("     %-28s entra %d (peso %.0f), sai %d (peso %.0f)"
                  % (a, len(e_), e_["Weight"].sum(), len(s_), s_["Weight"].sum()))


if __name__ == "__main__":
    main()
