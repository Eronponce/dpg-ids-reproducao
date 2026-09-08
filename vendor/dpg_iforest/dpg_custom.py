import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, f1_score, recall_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time 
import csv 



# Aponte para o diretório raiz do projeto DPG-iForest-main para encontrar os módulos dpg
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import shap
except ImportError:
    print("Aviso: A biblioteca 'shap' não está instalada. A análise de importância de features com SHAP será ignorada. Para habilitá-la, instale com: pip install shap")
    shap = None

import dpg.sklearn_custom_dpg as dpg_runner
from dpg.core import get_dpg, digraph_to_nx, get_dpg_node_metrics, get_dpg_edge_metrics, get_dpg_metrics
from dpg.visualizer import plot_dpg
from dpg.sklearn_custom_dpg import inliers_class_bounds, outlier_class_bounds, verifica_bounds



# --- Funções Auxiliares para o Novo Modo de Validação ---
def clean_numeric_X(X: pd.DataFrame) -> pd.DataFrame:
    # Igual ao benchmark
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)
    return X



def load_and_prepare_data(path, label_col_name=None, normal_class_override=None):
    """Carrega o dataset, identifica a coluna de rótulo e a separa das features numéricas.

    Retorna:
        X (DataFrame): apenas colunas numéricas (features)
        y_binary (Series|None): 1 para classe normal, -1 para anomalia (ou None se não houver rótulo)
        label_col_name (str|None): nome da coluna de rótulo detectada
        normal_class (Any|None): valor considerado "normal"
    """
    df = pd.read_csv(path, sep=',', low_memory=False)


    # 1) Descobrir coluna de rótulo (ou usar override)
    if label_col_name is not None and label_col_name not in df.columns:
        print(f"Aviso: label_col_name='{label_col_name}' não existe em {path}. Tentando autodetecção...")
        label_col_name = None

    if label_col_name is None:
        for col in ['target', 'label', 'Target', 'Label', 'class', 'Class']:
            if col in df.columns:
                label_col_name = col
                break

    # 2) Se não tem rótulo: retorna só features
    if not label_col_name:
        print("Aviso: Nenhuma coluna de rótulo encontrada. A validação supervisionada não será executada.")
        X = df.select_dtypes(include=np.number)
        X = clean_numeric_X(X)
        return X, None, None, None


    # 3) Define lista de features como no benchmark (todas, exceto label e colunas não-feature)
    drop_cols = {label_col_name, "dataset_family"}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].copy()
    X = clean_numeric_X(X)
    y = df[label_col_name]


    # 4) Define classe normal: preferir BENIGN quando existir; senão, majoritária (ou override)
    if normal_class_override is not None:
        normal_class = normal_class_override
    else:
        if 'BENIGN' in set(y.astype(str).unique()):
            normal_class = 'BENIGN'
        else:
            normal_class = y.value_counts().idxmax()

    # 5) Binariza: 1 = normal, -1 = anomalia
    y_binary = y.apply(lambda x: 1 if x == normal_class else -1)

    print(f"Coluna de rótulo encontrada: '{label_col_name}'. Classe normal: '{normal_class}'.")
    return X, y_binary, label_col_name, normal_class


def perform_standard_evaluation(y_true, y_pred, scores):
    """Calcula métricas de classificação e gera a matriz de confusão."""
    print("\n--- Avaliação do Modelo Isolation Forest Padrão (nos 30% de teste) ---")
    
    report = classification_report(y_true, y_pred, target_names=["Anomalia (-1)", "Normal (1)"])
    y_true_binary_for_auc = (y_true == -1).astype(int)
    roc_auc = roc_auc_score(y_true_binary_for_auc, -scores) # Invertemos os scores pois a sklearn espera scores maiores para a classe positiva (anomalia)
    
    print(report)
    print(f"ROC AUC Score: {roc_auc:.4f}")

    # Gera a figura da Matriz de Confusão
    cm = confusion_matrix(y_true, y_pred, labels=[-1, 1])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Anomalia", "Normal"], yticklabels=["Anomalia", "Normal"], ax=ax)
    ax.set_title("Matriz de Confusão")
    ax.set_ylabel("Verdadeiro")
    ax.set_xlabel("Previsto")
    
    metrics_dict = {
        "classification_report": report,
        "roc_auc": roc_auc
    }
    
    return metrics_dict, fig

def plot_anomaly_score_distribution(scores, save_path):
    """Gera um histograma da distribuição dos scores de anomalia."""
    fig, ax = plt.subplots()
    sns.histplot(scores, kde=True, ax=ax)
    ax.set_title("Distribuição dos Scores de Anomalia")
    ax.set_xlabel("Score de Anomalia")
    ax.set_ylabel("Frequência")
    fig.savefig(save_path)
    plt.close(fig) # Fecha a figura para liberar memória
    print(f"Gráfico de distribuição de scores salvo em: {save_path}")

def plot_feature_distributions(X_test, predictions, save_dir):
    """Gera e salva gráficos KDE para cada feature, comparando inliers e outliers."""
    feature_plots_dir = os.path.join(save_dir, "feature_plots")
    os.makedirs(feature_plots_dir, exist_ok=True)
    print(f"Salvando plots de features em: {feature_plots_dir}")

    # Adiciona as predições ao DataFrame de teste para facilitar a plotagem com 'hue'
    plot_df = X_test.copy()
    plot_df['prediction'] = predictions

    for feature in X_test.columns:
        fig, ax = plt.subplots()
        sns.kdeplot(data=plot_df, x=feature, hue='prediction', fill=True, 
                    common_norm=False, palette={1: "blue", -1: "red"}, ax=ax, warn_singular=False)
        ax.set_title(f"Distribuição para Feature: {feature}")
        
        # Sanitize feature name for filename
        sanitized_feature_name = "".join([c if c.isalnum() else "_" for c in feature])
        plot_path = os.path.join(feature_plots_dir, f"dist_{sanitized_feature_name}.png")
        fig.savefig(plot_path)
        plt.close(fig)


def analyze_feature_means(X_test, predictions):
    """
    Replicates the Streamlit logic to find features where the mean value is
    significantly different for anomalies vs. normal instances.
    """
    print("\n--- Análise de Comparação de Médias das Features ---")
    results_df = X_test.copy()
    results_df['prediction'] = predictions

    # Use dict comprehension for a more concise way to get means
    feature_stats = {col: results_df.groupby('prediction')[col].mean() for col in X_test.columns}
    
    anomaly_chars = []
    for feature, stats in feature_stats.items():
        if -1 in stats.index and 1 in stats.index:
            outlier_mean = stats.loc[-1]
            inlier_mean = stats.loc[1]
            
            if not np.isnan(outlier_mean) and not np.isnan(inlier_mean):
                # Check for non-trivial means to avoid flagging features full of zeros
                if abs(outlier_mean) > 1e-6 or abs(inlier_mean) > 1e-6:
                    if outlier_mean > inlier_mean * 1.5:
                        anomaly_chars.append({
                            'Feature': feature,
                            'Observation': 'Higher in Anomalies',
                            'Outlier Mean': f"{outlier_mean:.4f}",
                            'Inlier Mean': f"{inlier_mean:.4f}"
                        })
                    elif outlier_mean < inlier_mean * 0.5:
                        anomaly_chars.append({
                            'Feature': feature,
                            'Observation': 'Lower in Anomalies',
                            'Outlier Mean': f"{outlier_mean:.4f}",
                            'Inlier Mean': f"{inlier_mean:.4f}"
                        })
    
    if not anomaly_chars:
        print("Nenhuma característica de anomalia significativa encontrada na comparação de médias.")
        return None

    char_df = pd.DataFrame(anomaly_chars)
    print("Características de anomalia encontradas:")
    print(char_df.to_string())
    return char_df


def plot_mean_comparison_table(report_df, save_path):
    """
    Saves the feature mean comparison report as a table image.
    """
    if report_df is None or report_df.empty:
        print("DataFrame de comparação de médias está vazio. Gráfico não será gerado.")
        return

    fig, ax = plt.subplots(figsize=(10, len(report_df) * 0.5 + 1)) 
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=report_df.values, colLabels=report_df.columns, loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Gráfico de tabela de comparação de médias salvo em: {save_path}")


def analyze_shap_importance(model, X_test, save_dir):
    """
    Performs SHAP analysis to determine feature importance for the Isolation Forest model.
    """
    if not shap:
        print("Análise SHAP ignorada pois a biblioteca 'shap' não foi encontrada.")
        return None

    print("\n--- Análise de Importância de Features com SHAP ---")
    try:
        # Isolation Forest SHAP values are calculated based on feature contributions to the score
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        # Generate and save the SHAP summary plot (bar plot)
        plot_path = os.path.join(save_dir, "shap_summary_plot.png")
        plt.figure()
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.title("Importância das Features (SHAP)")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"Gráfico de resumo SHAP salvo em: {plot_path}")

        # Create a dataframe with mean absolute SHAP values
        vals = np.abs(shap_values).mean(0)
        feature_importance_df = pd.DataFrame(
            list(zip(X_test.columns, vals)),
            columns=['Feature', 'Mean Absolute SHAP']
        ).sort_values(by='Mean Absolute SHAP', ascending=False)
        
        print("Importância das features (SHAP):")
        print(feature_importance_df.to_string())
        
        return feature_importance_df

    except Exception as e:
        print(f"ERRO: A análise SHAP falhou. Erro: {e}")
        return None


def run_validation_flow(args):
    """
    Executa o fluxo completo de validação: treino/teste, avaliação padrão e análise DPG.
    """
    print("Modo de validação ativado.")
    
    # Passo 1: Carregar e preparar os dados
    if args.train_csv and args.test_csv:
        # Split externo: usa exatamente os CSVs fornecidos (para replicar o benchmark)
        X_train, y_train, label_col, normal_class = load_and_prepare_data(args.train_csv)
        X_test, y_test, _, _ = load_and_prepare_data(
            args.test_csv,
            label_col_name=label_col,
            normal_class_override=normal_class
        )

        if y_train is None or y_test is None:
            print("Encerrando o modo de validação: train_csv/test_csv precisam conter uma coluna de rótulo.")
            return

        # Alinha colunas numéricas do teste às do treino (ordem e conjunto)
        train_cols = X_train.columns.to_list()
        X_test = X_test.reindex(columns=train_cols, fill_value=0)
        X_train = X_train[train_cols]

        features_name = X_train.columns.to_numpy()
        print(f"Usando split externo (train_csv/test_csv). Train={len(X_train)} | Test={len(X_test)}")

    else:
        X, y, label_col, normal_class = load_and_prepare_data(args.ds)
        features_name = X.columns.to_numpy()

        if y is None:
            print("Encerrando o modo de validação, pois nenhum rótulo foi encontrado para avaliação.")
            return

        # Passo 2: Divisão Treino/Teste (70/30)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=args.seed, stratify=y
        )
        print(f"Dados divididos: {len(X_train)} para treino, {len(X_test)} para teste.")

    # Salva o conjunto de treino (útil para auditoria)
    train_df = X_train.copy()
    train_df['target'] = y_train.values
    train_df.to_csv("dataset_treino_70.csv", index=False)
    # Passo 3: Treinar o modelo Isolation Forest
    print("\nTreinando o modelo Isolation Forest...")
    iForest = IsolationForest(
        n_estimators=args.l, 
        contamination=args.cont, 
        random_state=args.seed,
        n_jobs=args.n_jobs
    )
    # depois do reindex/alinhamento (quando usa train_csv/test_csv)
    X_train = clean_numeric_X(X_train)
    X_test  = clean_numeric_X(X_test)

    # e, por segurança, imediatamente antes do fit:
    X_train = clean_numeric_X(X_train).astype(np.float64)
    X_test  = clean_numeric_X(X_test).astype(np.float64)


    print("NaNs train:", X_train.isna().sum().sum(), "| NaNs test:", X_test.isna().sum().sum())
    print("Infs train:", np.isinf(X_train.to_numpy()).sum(), "| Infs test:", np.isinf(X_test.to_numpy()).sum())

    iForest.fit(X_train)

    # Passo 4: Avaliação padrão ("Antes")
    predictions_test = iForest.predict(X_test)
    decision_scores_test = iForest.decision_function(X_test)
    
    eval_metrics, conf_matrix_fig = perform_standard_evaluation(y_test, predictions_test, decision_scores_test)

    # Passo 5: Análise DPG ("Depois")
    print("\nIniciando a análise DPG no conjunto de treino...")
    
    # Prepara dados para DPG (precisa ser numpy array)
    data_train_np = X_train.to_numpy()
    
    # Simula predições no set de treino para obter contagens de inliers/outliers para DPG
    predictions_train = iForest.predict(X_train)
    n_inliers_train = np.sum(predictions_train == 1)
    n_outliers_train = np.sum(predictions_train == -1)
    outliers_df_train = X_train[predictions_train == -1]

    # Chama as funções core do DPG
    dot, event_log, log_base = get_dpg(
        data_train_np, features_name, iForest, args.t, args.predicates,
        args.mode_graph, args.mode_score, data_train_np.shape[0], n_inliers_train, n_outliers_train, args.mode
    )
    
    dpg_model_nx, nodes_list_nx, edges_label_nx = digraph_to_nx(dot)
    
    df_nodes = get_dpg_node_metrics(dpg_model_nx, nodes_list_nx)
    df_edges = get_dpg_edge_metrics(dpg_model_nx, nodes_list_nx)
    
    # =========================
    # FIX: paths e bounds corretos (separa inliers vs outliers)
    # =========================
    from collections import defaultdict

    # 1) Reconstrói paths por case_id
    paths_dict = defaultdict(list)
    for case_id, step in event_log:
        paths_dict[str(case_id)].append(step)


    # 2) Lista ordenada (case_id, steps)
    paths_all = [(cid, steps) for cid, steps in paths_dict.items()]
    paths_all.sort(key=lambda x: x[0])

    # 3) Separa IDs por predição no treino
    # predictions_train está alinhado com X_train (índices 0..n-1)
    inlier_ids = set(np.where(predictions_train == 1)[0].tolist())
    outlier_ids = set(np.where(predictions_train == -1)[0].tolist())

    # CORRIGIDO. A chave do event_log e `sample{N}_dt{i}`, montada em
    # dpg/core.py:187, e nao o indice da amostra. Comparar a chave inteira
    # contra um conjunto de indices nunca casa, e era por isso que os bounds
    # saiam vazios com inliers=0 e outliers=0.
    import re as _re

    def _idx_amostra(cid):
        m = _re.match(r'sample(\d+)_dt\d+$', str(cid))
        return int(m.group(1)) if m else None

    _idx = {cid: _idx_amostra(cid) for cid, _ in paths_all}
    _sem_padrao = sum(1 for v in _idx.values() if v is None)
    if _sem_padrao:
        print(f'[BOUNDS] ATENCAO, {_sem_padrao} chaves fora do padrao sampleN_dtI')

    paths_inliers  = [(cid, steps) for cid, steps in paths_all
                      if _idx[cid] in inlier_ids]
    paths_outliers = [(cid, steps) for cid, steps in paths_all
                      if _idx[cid] in outlier_ids]

    print(f"[BOUNDS] paths_all={len(paths_all)} | inliers={len(paths_inliers)} | outliers={len(paths_outliers)}")

    # 4) Bounds corretos
    print("Calculando bounds (separado)...")
    global_bounds  = inliers_class_bounds(paths_inliers)
    local_bounds   = outlier_class_bounds(paths_outliers)
    anomaly_bounds = verifica_bounds(global_bounds, local_bounds)

    # agora ele deve ser o paths_all
    paths = paths_all


    df_dpg_metrics = get_dpg_metrics(
        dpg_model_nx, nodes_list_nx, outliers_df_train, event_log, edges_label_nx, log_base,
        args.mode, paths, global_bounds=global_bounds, local_bounds=local_bounds, anomaly_bounds=anomaly_bounds
    )
    
    print("Análise DPG concluída.")

    # Passo 6: Consolidar e salvar resultados
    # Criar diretório de resultados com timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = os.path.splitext(os.path.basename(args.train_csv if args.train_csv else args.ds))[0]
    results_dir = os.path.join(args.dir, f"{dataset_name}_validation_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    print(f"\nSalvando resultados em: {results_dir}")
    
    # Salvar as figuras
    conf_matrix_path = os.path.join(results_dir, "matriz_confusao.png")
    conf_matrix_fig.savefig(conf_matrix_path)
    plt.close(conf_matrix_fig)
    print(f"Matriz de confusão salva em: {conf_matrix_path}")

    # Salvar o novo gráfico de distribuição de scores
    score_dist_path = os.path.join(results_dir, "distribuicao_scores.png")
    plot_anomaly_score_distribution(decision_scores_test, score_dist_path)

    # Salvar os gráficos de distribuição por feature
    plot_feature_distributions(X_test, predictions_test, results_dir)

    # Passo 7: Novas Análises de Features
    # 7.1: Análise de Comparação de Médias
    mean_comparison_df = analyze_feature_means(X_test, predictions_test)
    if mean_comparison_df is not None:
        mean_comparison_plot_path = os.path.join(results_dir, "mean_comparison_analysis.png")
        plot_mean_comparison_table(mean_comparison_df, mean_comparison_plot_path)
    
    # 7.2: Análise de Importância com SHAP
    shap_importance_df = analyze_shap_importance(iForest, X_test, results_dir)

    # Salvar os dataframes de métricas do DPG (Moved before plotting to ensure data is saved)
    df_nodes.to_csv(os.path.join(results_dir, "dpg_nodes_metrics.csv"), index=False)
    df_edges.to_csv(os.path.join(results_dir, "dpg_edges_metrics.csv"), index=False)

    # GRAVA OS BOUNDS. Ate 08/09/2026 eles eram calculados e descartados, so
    # viviam no dicionario de get_dpg_metrics. `global_bounds` sao os do INLIER e
    # `local_bounds` os do OUTLIER, apesar do nome do codigo original.
    import math as _math

    def _linhas(d, lado):
        saida = []
        for feat, lim in sorted((d or {}).items()):
            lo, hi = lim.get('>', _math.inf), lim.get('<=', -_math.inf)
            lo = None if lo == _math.inf else lo
            hi = None if hi == -_math.inf else hi
            if lo is None and hi is None:
                continue
            if lo is not None and hi is not None:
                regra = f"{lo} < {feat} <= {hi}"
            elif hi is not None:
                regra = f"{feat} <= {hi}"
            else:
                regra = f"{feat} > {lo}"
            saida.append({"lado": lado, "feature": feat, "limite_inferior": lo,
                          "limite_superior": hi, "regra": regra})
        return saida

    _b = _linhas(global_bounds, "inlier") + _linhas(local_bounds, "outlier")
    if _b:
        pd.DataFrame(_b).to_csv(os.path.join(results_dir, "dpg_class_bounds.csv"),
                                index=False)
        print(f"[BOUNDS] gravadas {len(_b)} linhas em dpg_class_bounds.csv")
    else:
        print("[BOUNDS] NADA a gravar, os dois lados vieram vazios")

    if anomaly_bounds:
        pd.DataFrame([{"feature": k, "detalhe": str(v)}
                      for k, v in sorted(anomaly_bounds.items())]).to_csv(
            os.path.join(results_dir, "dpg_anomaly_bounds.csv"), index=False)
        print(f"[BOUNDS] {len(anomaly_bounds)} features em dpg_anomaly_bounds.csv")
    
    # Save Communities to CSV
    if "Communities" in df_dpg_metrics:
        communities_data = []
        for i, community in enumerate(df_dpg_metrics["Communities"]):
            for node in community:
                communities_data.append({"Node": node, "Community_ID": i})
        
        if communities_data:
            df_communities = pd.DataFrame(communities_data)
            df_communities.to_csv(os.path.join(results_dir, "dpg_communities.csv"), index=False)
            print("Métricas de Comunidades salvas em 'dpg_communities.csv'.")
        else:
             print("Nenhuma comunidade encontrada para salvar.")

    print("Métricas de Nós e Arestas do DPG salvas em arquivos CSV.")

    # Plot DPG
    if args.plot:
        print("Gerando e salvando o gráfico DPG...")
        plot_generated = False
        
        if args.attribute:
            try:
                print(f"Gerando plot com atributo: {args.attribute}")
                plot_dpg(
                    "dpg_validation_plot",
                    dot, df_nodes, df_edges, df_dpg_metrics,
                    save_dir=results_dir, 
                    attribute=args.attribute, 
                    communities=False,
                    class_flag=args.class_flag, 
                    edge_attribute=args.edge_attribute
                )
                plot_generated = True
            except Exception as e:
                print(f"Erro ao gerar plot com atributo: {e}")

        if args.communities:
            try:
                print("Gerando plot de comunidades...")
                plot_dpg(
                    "dpg_validation_plot",
                    dot, df_nodes, df_edges, df_dpg_metrics,
                    save_dir=results_dir, 
                    attribute=None, 
                    communities=True,
                    class_flag=args.class_flag, 
                    edge_attribute=args.edge_attribute
                )
                plot_generated = True
            except Exception as e:
                print(f"Erro ao gerar plot de comunidades: {e}")
        
        if not plot_generated and not args.attribute and not args.communities:
             try:
                print("Gerando plot padrão...")
                plot_dpg(
                    "dpg_validation_plot",
                    dot, df_nodes, df_edges, df_dpg_metrics,
                    save_dir=results_dir, 
                    attribute=None, 
                    communities=False,
                    class_flag=args.class_flag, 
                    edge_attribute=args.edge_attribute
                )
             except Exception as e:
                print(f"Erro ao gerar plot padrão: {e}")
    
    # Salvar o relatório de texto
    report_path = os.path.join(results_dir, "relatorio_consolidado.txt")
    with open(report_path, "w") as f:
        f.write("--- Relatório Consolidado de Análise e Validação ---\n\n")
        f.write("--- Parâmetros de Execução ---\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
        
        f.write("\n--- Métricas de Avaliação (Isolation Forest Padrão no teste) ---\n")
        f.write(f"ROC AUC: {eval_metrics['roc_auc']:.4f}\n\n")
        f.write(eval_metrics['classification_report'])

        f.write("\n\n--- Análise de Comparação de Médias de Features ---\n")
        if mean_comparison_df is not None:
            f.write(mean_comparison_df.to_string())
        else:
            f.write("Nenhuma característica de anomalia significativa encontrada na comparação de médias.")
        f.write("\n")

        f.write("\n\n--- Análise de Importância de Features (SHAP) ---\n")
        if shap_importance_df is not None:
            f.write(shap_importance_df.to_string())
        else:
            f.write("Análise SHAP não executada (biblioteca 'shap' pode estar ausente).")
        f.write("\n")

        f.write("\n\n--- Métricas da Análise DPG (gerado com dados de treino) ---\n")
        for key, value in df_dpg_metrics.items():
            if key in ["Outliers", "Paths"]:
                if isinstance(value, pd.DataFrame):
                    f.write(f"{key}: DataFrame com {value.shape[0]} linhas e {value.shape[1]} colunas\n")
                else:
                    f.write(f"{key}: {type(value).__name__} com {len(value)} elementos\n")
            else:
                f.write(f"{key}: {value}\n")

    print(f"Relatório consolidado salvo em: {report_path}")

    # (Métricas já salvas anteriormente)


def run_original_flow(args):
    """
    Executa o fluxo original do script, chamando a função `test_base_sklearn`.
    """
    print("Executando o fluxo original de análise DPG (sem validação)...")
    
    df_node_list, df_dpg_metrics_list, index_list, df_edge_list = dpg_runner.test_base_sklearn(
        datasets=args.ds,
        n_learners=args.l, 
        decimal_threshold=args.t,
        contamination=args.cont,
        seed=args.seed,
        n_jobs=args.n_jobs, # Pass n_jobs here
        file_name=os.path.join(args.dir, f'custom_l{args.l}_t{args.t}_s{args.seed}_{args.predicates}_{args.mode_graph}_{args.mode_score}_stats.txt'),
        plot=args.plot, 
        save_plot_dir=args.save_plot_dir, 
        attribute=args.attribute, 
        communities=args.communities, 
        class_flag=args.class_flag,
        predicates=args.predicates,
        mode=args.mode,
        mode_graph=args.mode_graph,
        mode_score=args.mode_score,
        edge_attribute=args.edge_attribute
    )

    # Lógica de salvamento original
    nameDataset = os.path.splitext(os.path.basename(args.ds))[0]
    
    if args.mode in ["global", "global_outliers"]:
        df_edge_list.to_csv(os.path.join(args.dir, f'{nameDataset}_l{args.l}_t{args.t}_s{args.seed}_{args.predicates}_{args.mode_graph}_{args.mode_score}_global_edge_metrics.csv'), encoding='utf-8')
        df_node_list.to_csv(os.path.join(args.dir, f'{nameDataset}_l{args.l}_t{args.t}_s{args.seed}_{args.predicates}_{args.mode_graph}_{args.mode_score}_global_node_metrics.csv'), encoding='utf-8')
        with open(os.path.join(args.dir, f'{nameDataset}_l{args.l}_t{args.t}_s{args.seed}_{args.predicates}_{args.mode_graph}_{args.mode_score}_global_dpg_metrics.txt'), 'w') as f:
            for key, value in df_dpg_metrics_list.items():
                if isinstance(value, tuple):
                    f.write(f"{key} (tuple):\n{str(value)}\n")
                else:
                    f.write(f"{key}:\n{value}\n")
    else:
        for i, (df_node, df_dpg_metrics, index, df_edge) in enumerate(zip(df_node_list, df_dpg_metrics_list, index_list, df_edge_list)):
            df_node.to_csv(os.path.join(args.dir, f'{nameDataset}_l{args.l}_t{args.t}_s{args.seed}_{args.predicates}_anomaly{index}_{args.mode_graph}_{args.mode_score}_node_metrics.csv'), encoding='utf-8')    
            df_edge.to_csv(os.path.join(args.dir, f'{nameDataset}_l{args.l}_t{args.t}_s{args.seed}_{args.predicates}_anomaly{index}_{args.mode_graph}_{args.mode_score}_edge_metrics.csv'), encoding='utf-8')
            with open(os.path.join(args.dir, f'{nameDataset}_l{args.l}_t{args.t}_s{args.seed}_{args.predicates}_anomaly{index}_{args.mode_graph}_{args.mode_score}_dpg_metrics.txt'), 'w') as f:
                for key, value in df_dpg_metrics.items():
                    f.write(f"{key}:\n{value}\n")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DPG-iForest Analysis Script with optional validation.")
    
    # Argumentos existentes
    parser.add_argument("--ds", type=str, required=True, help="Path to the dataset CSV file.")
    parser.add_argument("--train_csv", type=str, default=None, help="Optional: path to a pre-split TRAIN CSV (used with --validate to replicate a benchmark split).")
    parser.add_argument("--test_csv", type=str, default=None, help="Optional: path to a pre-split TEST CSV (used with --validate to replicate a benchmark split).")
    parser.add_argument("--l", type=int, default=5, help="Number of learners for the Isolation Forest.")
    parser.add_argument("--t", type=int, default=2, help="Decimal precision of each feature.")
    parser.add_argument("--cont", type=float, default=0.01, help="Rate of outliers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of jobs to run in parallel for IsolationForest.")
    parser.add_argument("--dir", type=str, default="examples/", help="Directory to save results.")
    parser.add_argument("--plot", action='store_true', help="Plot the DPG.")
    parser.add_argument("--save_plot_dir", type=str, default="examples/", help="Directory to save the plot image.")
    parser.add_argument("--attribute", type=str, default=None, help="A specific node attribute to visualize.")
    parser.add_argument("--communities", action='store_true', help="Visualize communities.")
    parser.add_argument("--class_flag", action='store_true', help="Highlight class nodes.")
    parser.add_argument("--predicates", type=str, default="feature_operator", help="Type of predicate.")
    parser.add_argument("--mode", type=str, default="global", help="Analysis mode (e.g., global, local_outliers).")
    parser.add_argument("--mode_graph", type=str, default="all", help="Graph mode (all or last_decisions).")
    parser.add_argument("--mode_score", type=str, default="log2", help="Score mode (e.g., log2, freq_pes).")
    parser.add_argument("--edge_attribute", type=str, default=None, help="A specific edge attribute to visualize.")

    # Novo argumento para validação
    parser.add_argument("--validate", action='store_true', help="Enable validation mode with 70/30 train/test split.")
    
    args = parser.parse_args()

    if args.validate:
        run_validation_flow(args)
    else:
        run_original_flow(args)
