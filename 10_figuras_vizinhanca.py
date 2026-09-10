# -*- coding: utf-8 -*-
"""10. As arvores de CO-OCORRENCIA dos predicados discutidos no Capitulo 4.

UMA FIGURA POR PREDICADO, oito ao todo, duas por classe. Cada figura e uma
arvore indentada que DESCE: a raiz no topo, os filhos abaixo e recuados, e o
ramo mais pesado aberto mais um nivel a cada passo, ate um no de classe.

O peso na conexao e a CO-OCORRENCIA: quantas vezes os dois predicados foram
satisfeitos em sequencia durante o treino. Nao tem relacao com class bounds.

O que ENTRA na raiz aparece como anotacao acima dela, porque saber se algo entra
e o que separa uma RAIZ do grafo de um no de passagem.

TIPOGRAFIA, igual a das figuras do Capitulo 2, de gerar_dpg-passos.py:
largura fixa 6.30 in, que e o textwidth do documento; serif Times New Roman;
corpos 11, 10.5 e 10 pt; mesma paleta; caixas com rounding_size 1.0, linewidth
0.8 e mutation_aspect 0.28. Com includegraphics[width=textwidth] a escala fica
1:1 e os corpos saem no PDF com o valor declarado aqui.

A LARGURA DE CADA CAIXA E MEDIDA, nao estimada. Versoes anteriores chutavam a
largura do caractere e o texto vazava das caixas. Aqui o texto e desenhado, o
renderer devolve a extensao e a caixa e dimensionada a partir disso.

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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

W = 6.30                              # = textwidth do documento, 455 pt
FS_NO, FS_NOTA, FS_MINI = 11.0, 10.5, 10.0
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
                     "mathtext.fontset": "stix", "font.size": FS_NO})

INK, NODE, LEAF, PALE, MUTED, SOFT = ("#1a1a1a", "#ffffff", "#d9d9d9",
                                      "#b8b8b8", "#4a4a4a", "#f2f2f2")

DEST = ("C:/Users/eronp/OneDrive/Documentos/GitHub/Dissertação/Dissertação/"
        "Latex/figuras/")
CLASSES = ["DoS/DDoS", "Benign", "Spoofing", "Reconnaissance"]
CURTO = {"DoS/DDoS": "dosddos", "Benign": "benign",
         "Spoofing": "spoofing", "Reconnaissance": "reconnaissance"}

ALT_CX = 5.0                          # altura da caixa, em unidades de dados
LINHA = 7.8                           # altura de uma linha da arvore
RECUO = 7.0                           # recuo por nivel
PROF = 3                              # niveis abertos alem da raiz


def fmt(v):
    return "{:,}".format(int(v)).replace(",", ".")


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


def box(ax, xesq, y, w, t, fill=NODE, fs=FS_NO, lw=0.8):
    ax.add_patch(FancyBboxPatch((xesq, y - ALT_CX / 2.), w, ALT_CX,
                                boxstyle="round,pad=0,rounding_size=1.0",
                                linewidth=lw, edgecolor=INK, facecolor=fill,
                                mutation_aspect=0.28, zorder=4))
    ax.text(xesq + w / 2., y, t, ha="center", va="center", fontsize=fs,
            color=INK, zorder=5)


def cotovelo(ax, x_pai, y_pai, x_filho, y_filho, peso, forte):
    """Conector em L, como se desenha arvore indentada."""
    cor = INK if forte else PALE
    lw = 1.3 if forte else 0.8
    ax.plot([x_pai, x_pai], [y_pai - ALT_CX / 2., y_filho], color=cor,
            linewidth=lw, zorder=3, solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch((x_pai, y_filho), (x_filho - 0.4, y_filho),
                                 arrowstyle="-|>", mutation_scale=8,
                                 linewidth=lw, color=cor, shrinkA=0, shrinkB=0,
                                 zorder=3))
    ax.text((x_pai + x_filho) / 2., y_filho + 1.7, fmt(peso), ha="center",
            va="bottom", fontsize=FS_MINI, color=INK if forte else MUTED,
            zorder=7, bbox=dict(boxstyle="round,pad=0.10", facecolor="white",
                                edgecolor="none", alpha=0.95))


def entrada(ax, x_tronco, y_pai, w_pai, peso):
    """Conector de quem ENTRA, espelhando o dos filhos: o pai fica recuado a
    direita, corre para a esquerda ate o tronco, e o tronco desce ate o no. Todos
    os pais compartilham o mesmo tronco, do mesmo jeito que os filhos, entao as
    duas metades da figura tem a mesma gramatica."""
    ax.plot([x_tronco, x_tronco + RECUO - 0.4], [y_pai, y_pai],
            color=PALE, linewidth=0.8, zorder=3, solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch((x_tronco + RECUO - 0.4, y_pai),
                                 (x_tronco + 0.6, y_pai),
                                 arrowstyle="-|>", mutation_scale=7, linewidth=0.8,
                                 color=PALE, shrinkA=0, shrinkB=0, zorder=3))
    # o peso vai SOBRE a seta horizontal, do mesmo jeito que o dos filhos, e nao
    # solto a direita da caixa
    ax.text(x_tronco + (RECUO - 0.4) / 2., y_pai + 1.2, fmt(peso), ha="center",
            va="bottom", fontsize=FS_MINI, color=MUTED, zorder=7,
            bbox=dict(boxstyle="round,pad=0.10", facecolor="white",
                      edgecolor="none", alpha=0.95))


def linhas_da_arvore(alvo, ar):
    """[(nivel, rotulo, peso da aresta que chega, forte)] em ordem de PROFUNDIDADE.

    Numa arvore indentada a subarvore vem logo abaixo do seu pai, e nao depois de
    todos os irmaos. Entao a ordem e: raiz, ramo mais pesado, a cadeia dele, e so
    entao os demais irmaos. Assim o pai de um no de nivel L e sempre o ultimo no
    desenhado no nivel L-1.
    """
    out = [(0, alvo, None, True)]
    f = ar[ar["Node_u_label"] == alvo].sort_values("Weight", ascending=False)
    if len(f) == 0:
        return out

    # o ramo mais pesado, e a cadeia que desce a partir dele
    e0 = f.iloc[0]
    pesado = str(e0["Node_v_label"])
    out.append((1, pesado, float(e0["Weight"]), True))
    atual, nivel, vistos = pesado, 1, {alvo, pesado}
    for _ in range(PROF - 1):
        if atual.startswith("Class "):
            break
        prox = ar[ar["Node_u_label"] == atual]
        if len(prox) == 0:
            break
        e = prox.nlargest(1, "Weight").iloc[0]
        rot = str(e["Node_v_label"])
        if rot in vistos:
            break
        out.append((nivel + 1, rot, float(e["Weight"]), True))
        vistos.add(rot)
        atual, nivel = rot, nivel + 1

    # os demais irmaos, depois da subarvore do primeiro
    for _, e in f.iloc[1:].iterrows():
        out.append((1, str(e["Node_v_label"]), float(e["Weight"]), False))
    return out


def gera(alvo, arquivo, ar, classe):
    linhas = linhas_da_arvore(alvo, ar)
    ent = ar[ar["Node_v_label"] == alvo]

    n_ent = len(ar[ar['Node_v_label'] == alvo])
    ylim = (len(linhas) + n_ent) * LINHA + 12
    alt_in = W * ylim / 100.0
    fig, ax = plt.subplots(figsize=(W, alt_in))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, ylim)
    ax.axis("off")

    # quem ENTRA na raiz, desenhado ACIMA dela. Sem isso a figura mostra so o
    # que sai e contradiz o texto, que fala de aresta entrando e saindo
    pais = ar[ar["Node_v_label"] == alvo].sort_values("Weight", ascending=False)
    MAX_PAIS = 4
    n_pais = min(len(pais), MAX_PAIS)
    extra = len(pais) - n_pais

    # primeira passada: mede tudo para saber a largura ocupada, e centraliza.
    # a figura tem largura fixa de textwidth, entao sem isso a arvore fica
    # encostada na margem esquerda e sobra um vazio grande a direita
    larguras = [medir(fig, ax, r, FS_NO) + 4.0 for _, r, _, _ in linhas]
    larg_pais = [medir(fig, ax, str(r["Node_u_label"]), FS_NO) + 4.0
                 for _, r in pais.iterrows()]
    ocupado = max([n * RECUO + w for (n, _, _, _), w in zip(linhas, larguras)]
                  + [RECUO + w for w in larg_pais] + [0])
    esq = max(2.0, (100.0 - ocupado) / 2.0)

    y = ylim - 6
    if len(pais):
        # TODOS os pais, sem corte. Sem eles a soma do que entra nao fecha, e o
        # texto cita percentagens cujo denominador ficaria fora da figura
        rot_pais = [str(r["Node_u_label"]) for _, r in pais.iterrows()]
        w_raiz = medir(fig, ax, linhas[0][1], FS_NO) + 4.0
        x_tronco = esq + 1.8
        y_topo_pais = y
        for k2, r2 in enumerate(rot_pais):
            wp = medir(fig, ax, r2, FS_NO) + 4.0
            box(ax, esq + RECUO, y, wp, r2)
            entrada(ax, x_tronco, y, wp, float(pais.iloc[k2]["Weight"]))
            y -= LINHA
        y_raiz = y - 1.2
        # o tronco que junta os pais e desce ate o no
        ax.plot([x_tronco, x_tronco], [y_topo_pais, y_raiz + ALT_CX / 2. + 1.3],
                color=PALE, linewidth=0.8, zorder=3, solid_capstyle="butt")
        ax.add_patch(FancyArrowPatch((x_tronco, y_raiz + ALT_CX / 2. + 1.5),
                                     (x_tronco, y_raiz + ALT_CX / 2. + 0.3),
                                     arrowstyle="-|>", mutation_scale=8, linewidth=0.8,
                                     color=PALE, shrinkA=0, shrinkB=0, zorder=3))
        y = y_raiz
    pos = {}
    for k, ((nivel, rot, peso, forte), w) in enumerate(zip(linhas, larguras)):
        x = esq + nivel * RECUO
        if nivel > 0:
            xp, yp = pos[nivel - 1]
            cotovelo(ax, xp + 1.8, yp, x, y, peso, forte)
        box(ax, x, y, w, rot,
            fill=LEAF if rot.startswith("Class ") else (SOFT if nivel == 0 else NODE),
            lw=1.3 if nivel == 0 else 0.8)
        pos[nivel] = (x, y)
        y -= LINHA

    plt.subplots_adjust(left=0.004, right=0.996, top=0.996, bottom=0.004)
    for d in (DEST, "figuras"):
        if not os.path.isdir(d):
            os.makedirs(d)
        fig.savefig(os.path.join(d, arquivo + ".pdf"), facecolor="white")
    fig.savefig(os.path.join("figuras", arquivo + ".png"), dpi=300, facecolor="white")
    plt.close(fig)
    print("  %-16s %-32s %2d linhas, %.2f x %.2f pol"
          % (classe, arquivo, len(linhas), W, alt_in))


def main():
    D = "dados"
    ar = pd.read_csv(os.path.join(D, "rf_grafo_arestas.csv"))
    nos = pd.read_csv(os.path.join(D, "rf_grafo_nos.csv"))
    j = json.load(io.open(os.path.join(D, "rf_grafo_comunidades.json"), encoding="utf-8"))
    lrc = dict(zip(nos["Label"].astype(str), nos["Local reaching centrality"]))
    for c in CLASSES:
        it = [str(x) for x in j["Clusters"]["Class " + c]
              if not str(x).startswith("Class ") and str(x) in lrc]
        for k, a in enumerate(sorted(it, key=lambda x: -float(lrc[x]))[:2], 1):
            gera(a, "rf-arvore-%s-%d" % (CURTO[c], k), ar, c)


if __name__ == "__main__":
    main()
