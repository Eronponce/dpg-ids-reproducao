# -*- coding: utf-8 -*-
"""Extrai os class bounds do cenario supervisionado, reexecutando o DPG.

Esta e a unica das cinco leituras do supervisionado que nunca tinha sido salva.
As outras quatro ja estao em dados/rf_grafo_{nos,arestas,comunidades}. Os bounds
exigem o objeto de explicacao, e por isso exigem o DPG rodando: nao dao para
derivar dos arquivos guardados.

A floresta e a mesma do primeiro comando, mesma particao e mesma semente.

ATENCAO AO FORMATO DA CONFIG. Ela e ANINHADA, config['dpg']['default'] e
config['dpg']['graph_construction']. Um dicionario plano e aceito sem reclamar e
IGNORADO por inteiro, e a execucao segue com os padroes da biblioteca,
perc_var=1e-09 e decimal_threshold=6, que produzem um grafo diferente do
capitulo. Foi o que aconteceu na primeira tentativa, em 08/09/2026.

O script se recusa a gravar se nao conseguir conferir o grafo contra o publicado.
Nao existe conferencia parcial: ou bate, ou sai com codigo 1.
"""
import io
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import dpg  # noqa: E402

# os quatro valores declarados no capitulo de metodos
CONFIG = {
    "dpg": {
        "default": {"perc_var": 0.001, "decimal_threshold": 2, "n_jobs": -1},
        "graph_construction": {"mode": "aggregated_transitions"},
        "visualization": {},
    }
}
LIMIAR_COMUNIDADE = 0.2


def rotulos_do_grafo(d):
    """Rotulos dos nos de predicado e de classe, SEM as arestas.

    A lista de nos da biblioteca traz tambem as arestas, cujo identificador tem
    `->` e cujo rotulo e vazio. Ignora-las e o que o proprio codigo da variante
    Isolation Forest faz. Sem isso a contagem vira 479 + 758 = 1237.

    Devolve None se nao conseguir ler, e quem chama ABORTA. Nunca parcial.
    """
    nm = d.get("node_metrics")
    if nm is not None and hasattr(nm, "columns"):
        for c in ("Label", "label", "name"):
            if c in nm.columns:
                r = [str(x) for x in nm[c] if str(x).strip()]
                if r:
                    print("  rotulos lidos de node_metrics")
                    return r

    nos = d.get("nodes")
    if nos is not None:
        try:
            if hasattr(nos, "columns"):
                for c in ("Label", "label", "name"):
                    if c in nos.columns:
                        r = [str(x) for x in nos[c] if str(x).strip()]
                        if r:
                            print("  rotulos lidos da coluna %s de nodes" % c)
                            return r
            elif isinstance(nos, (list, tuple)) and nos:
                if isinstance(nos[0], (list, tuple)) and len(nos[0]) > 1:
                    r = [str(x[1]) for x in nos
                         if "->" not in str(x[0]) and str(x[1]).strip()]
                    if r:
                        print("  rotulos lidos de nodes, %d entradas de aresta ignoradas"
                              % (len(nos) - len(r)))
                        return r
        except Exception as e:
            print("  falha ao ler 'nodes': %s" % e)

    g = d.get("graph")
    if g is not None and hasattr(g, "nodes"):
        try:
            r = [str(g.nodes[n].get("label", n)) for n in g.nodes()
                 if "->" not in str(n)]
            r = [x for x in r if x.strip()]
            if r:
                print("  rotulos lidos de graph")
                return r
        except Exception as e:
            print("  falha ao ler 'graph': %s" % e)
    return None


def main():
    print("lendo dados/dataset_balanceado_label.csv")
    df = pd.read_csv("dados/dataset_balanceado_label.csv")
    y = df["label"].values
    feats = [c for c in df.columns if c != "label"]
    X = df[feats].values

    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    print("  %d linhas de treino, %d features" % (len(X_tr), len(feats)))

    rf = RandomForestClassifier(n_estimators=20, max_depth=16, min_samples_split=2,
                                min_samples_leaf=2, class_weight="balanced",
                                random_state=42)
    rf.fit(X_tr, y_tr)
    alvos = [str(c) for c in rf.classes_]
    print("  floresta treinada, classes %s" % alvos)

    print("\nconstruindo o DPG. Confira no cabecalho abaixo que saiu")
    print("perc_var=0.001 e decimal_threshold=2, e nao os padroes da biblioteca")
    expl = dpg.DPGExplainer(rf, feature_names=feats, target_names=alvos,
                            dpg_config=CONFIG)
    explicacao = expl.explain_global(X_tr, communities=True,
                                     community_threshold=LIMIAR_COMUNIDADE)
    d = explicacao.as_dict()

    # ---------- autoconferencia, sem meio termo ----------
    print("\nautoconferencia contra o grafo publicado")
    pub = pd.read_csv("dados/rf_grafo_nos.csv")
    rot_pub = sorted(str(x) for x in pub["Label"])
    rot_novo = rotulos_do_grafo(d)

    if rot_novo is None:
        print("  NAO consegui ler os rotulos do grafo reconstruido.")
        print("  chaves disponiveis: %s" % sorted(d.keys()))
        print("  ABORTANDO sem gravar. Bounds nao conferidos nao servem.")
        sys.exit(1)

    print("  publicados %d nos (%d distintos) | reconstruidos %d nos (%d distintos)"
          % (len(rot_pub), len(set(rot_pub)), len(rot_novo), len(set(rot_novo))))
    if set(rot_novo) != set(rot_pub):
        so_novo = set(rot_novo) - set(rot_pub)
        so_pub = set(rot_pub) - set(rot_novo)
        print("  >> DIVERGE. %d rotulos so no novo, %d so no publicado"
              % (len(so_novo), len(so_pub)))
        for x in list(so_novo)[:5]:
            print("       so no novo: %s" % x)
        for x in list(so_pub)[:5]:
            print("       so no publicado: %s" % x)
        print("  ABORTANDO sem gravar. Os bounds seriam de outro grafo.")
        sys.exit(1)
    print("  >> CONFERE, rotulo a rotulo")

    # ---------- so agora grava ----------
    bounds = dpg.classwise_feature_bounds_from_communities(explicacao)
    contagens = dpg.class_feature_predicate_counts(explicacao)

    bounds.to_csv("dados/rf_class_bounds.csv", index=False)
    print("\ndados/rf_class_bounds.csv  %d linhas" % len(bounds))
    print(bounds.head(10).to_string())

    contagens.to_csv("dados/rf_predicados_por_classe.csv", index=True)
    print("\ndados/rf_predicados_por_classe.csv  %s" % (contagens.shape,))

    print("\nRESULTADO: class bounds do supervisionado extraidos e conferidos.")


if __name__ == "__main__":
    main()
