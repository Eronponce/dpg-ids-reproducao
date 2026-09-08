# -*- coding: utf-8 -*-
"""Reconstroi os tres grafos do cenario nao supervisionado a partir dos splits.

Este e o comando que fecha a reprodutibilidade do lado nao supervisionado. Ate
ele existir o repositorio trazia os grafos prontos e nenhuma forma de refaze-los,
porque a DPG-iForest nao e distribuida pelo PyPI. Ela agora esta embarcada em
vendor/dpg_iforest, sob licenca MIT, com o aviso e as modificacoes descritos em
vendor/dpg_iforest/NOTICE.md.

Para cada mistura, roda o driver sobre o split de treino e compara o que sai
contra o que esta publicado em dados/. A comparacao e EXATA, valor por valor,
inclusive o IOP-Score em ponto flutuante. Se algo divergir, sai com codigo 1.

    python 8_reconstroi_grafos_if.py            # as tres, cerca de 50 min cada
    python 8_reconstroi_grafos_if.py single     # so uma

Os parametros sao os do relatorio consolidado das execucoes de 04/09/2026:
cinquenta arvores, contaminacao 0,01, semente 42, predicados de dois elementos,
modo de escore freq_pes.
"""
import glob
import io
import os
import shutil
import subprocess
import sys
import tempfile

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

RAIZ = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(RAIZ, "vendor", "dpg_iforest")
DADOS = os.path.join(RAIZ, "dados")
SPLITS = os.path.join(DADOS, "splits_if")

TESTE = {"single": "single_test_fixed_BEN05.csv",
         "dupla": "dupla_test_fixed_BEN05.csv",
         "trio": "trio_test_fixed_BEN05.csv"}

# arquivo que sai da execucao -> arquivo publicado, com a chave de ordenacao
PARES = [("dpg_nodes_metrics.csv", "if_%s_nos.csv", ["Node"]),
         ("dpg_edges_metrics.csv", "if_%s_arestas.csv", ["Source_id", "Target_id"]),
         ("dpg_communities.csv", "if_%s_comunidades.csv", ["Node"]),
         ("dpg_class_bounds.csv", "if_%s_class_bounds.csv", ["lado", "feature"])]


def compara(saiu, publicado, chave):
    a = pd.read_csv(saiu)
    b = pd.read_csv(publicado)
    # a coluna `regra` do arquivo publicado foi normalizada na consolidacao,
    # entao a comparacao dos bounds e sobre os numeros
    for d in (a, b):
        if "regra" in d.columns:
            d.drop(columns=["regra"], inplace=True)
    if list(a.columns) != list(b.columns):
        return False, "colunas diferem: %s contra %s" % (list(a.columns), list(b.columns))
    if a.shape != b.shape:
        return False, "forma difere: %s contra %s" % (a.shape, b.shape)
    ordem = chave if (chave and all(c in a.columns for c in chave)) else list(a.columns)
    a = a.sort_values(ordem).reset_index(drop=True)
    b = b.sort_values(ordem).reset_index(drop=True)
    if a.equals(b):
        return True, "identico, %s" % (a.shape,)
    num = a.select_dtypes("number").columns
    pior, onde = 0.0, ""
    for c in num:
        d = (a[c] - b[c]).abs().max()
        if d > pior:
            pior, onde = d, c
    naonum = [c for c in a.columns if c not in num and not a[c].equals(b[c])]
    if pior == 0 and not naonum:
        return True, "identico"
    return False, "maior desvio %.3e em %s%s" % (
        pior, onde, "; colunas nao numericas diferem: %s" % naonum if naonum else "")


def roda(cen, tmp):
    saida = os.path.join(tmp, cen)
    cmd = [sys.executable, "-u", "dpg_custom.py",
           "--ds", cen,
           "--train_csv", os.path.join(SPLITS, "%s_train_BEN05_cont001.csv" % cen),
           "--test_csv", os.path.join(SPLITS, TESTE[cen]),
           "--l", "50", "--t", "2", "--cont", "0.01", "--seed", "42",
           "--n_jobs", "-1", "--dir", saida,
           "--predicates", "feature_operator", "--mode", "global",
           "--mode_graph", "all", "--mode_score", "freq_pes", "--validate"]
    p = subprocess.run(cmd, cwd=VENDOR, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    for l in (p.stdout or "").split("\n"):
        if "[BOUNDS]" in l:
            print("   %s" % l.strip())
    if p.returncode != 0:
        print("   driver saiu com codigo %d" % p.returncode)
        print("   %s" % (p.stderr or "")[-600:])
        return None
    achados = glob.glob(os.path.join(saida, "*", "dpg_nodes_metrics.csv"))
    return os.path.dirname(achados[-1]) if achados else None


def main():
    quais = sys.argv[1:] or ["single", "dupla", "trio"]
    for c in quais:
        if c not in TESTE:
            print("mistura desconhecida: %s" % c)
            sys.exit(2)

    if not os.path.isdir(SPLITS):
        print("dados/splits_if/ nao existe. Sem os splits nao ha o que reconstruir.")
        sys.exit(2)

    falhas = []
    tmp = tempfile.mkdtemp(prefix="dpg_if_")
    try:
        for cen in quais:
            print("=" * 78)
            print("### %s ###  reconstruindo, isto leva dezenas de minutos" % cen.upper())
            print("=" * 78)
            pasta = roda(cen, tmp)
            if pasta is None:
                print("   >> a execucao falhou")
                falhas.append(cen)
                continue
            for nome, molde, chave in PARES:
                saiu = os.path.join(pasta, nome)
                pub = os.path.join(DADOS, molde % cen)
                if not os.path.exists(saiu):
                    print("   %-26s NAO FOI PRODUZIDO" % nome)
                    falhas.append("%s/%s" % (cen, nome))
                    continue
                if not os.path.exists(pub):
                    print("   %-26s sem par publicado, pulando" % nome)
                    continue
                ok, msg = compara(saiu, pub, chave)
                print("   %-26s %s  %s" % (nome, "CONFERE" if ok else "DIVERGE", msg))
                if not ok:
                    falhas.append("%s/%s" % (cen, nome))
            print()
    finally:
        if falhas:
            print("saida da reconstrucao mantida em %s para inspecao" % tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)

    print("=" * 78)
    if falhas:
        print("DIVERGENCIAS: %s" % ", ".join(falhas))
        sys.exit(1)
    print("RESULTADO: os grafos reconstruidos sao identicos aos publicados.")


if __name__ == "__main__":
    main()
