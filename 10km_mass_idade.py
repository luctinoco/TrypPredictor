import pandas as pd
import matplotlib.pyplot as plt
import scikitplot as skplt
import pickle as pk
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, confusion_matrix, precision_score, recall_score, average_precision_score, f1_score, make_scorer
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFECV
import numpy as np
from imblearn.pipeline import Pipeline
import os
import shap

# Caminho para o arquivo Excel
caminho_arquivo = '/projects/pi.samanta.xavier/proj.lucas.leonardo/proj.mestrado/10km_mass_idade/10km_mass_idade_correlacao.xlsx'

# Função para depuração
def debug_params(grid_params, rfe_params, model_params, filename='debug_params.txt'):
    with open(filename, 'a') as f:
        f.write("Comparação de Parâmetros:\n")
        f.write("Parâmetros do GridSearchCV:\n")
        f.write(str(grid_params) + "\n\n")
        f.write("Parâmetros utilizados no RFE:\n")
        f.write(str(rfe_params) + "\n\n")
        f.write("Parâmetros utilizados no modelo final:\n")
        f.write(str(model_params) + "\n\n")
        f.write("=========================================\n\n")

# Definindo função de dataset
def load_dataset(caminho_arquivo):
    col_names = [
        'Animal_infectado', 'Sexo_binary', 'Faixa_etaria', 'Body_mass', 'LAT', 'Menhinick', 'Like.Adjacencies_1',
        'Splitting_index_1', 'variation_fractal_dimension_index_1', 'edge_density_14', 'Splitting_index_14',
        'variation_of_euclidean_nearest.neighbor_distance_22', 'Like.Adjacencies_22', 'Splitting_index_22',
        'variation_fractal_dimension_index_22', 'bio_1', 'bio_17', 'bio_18', 'bio_2', 'bio_3'
    ]
    dataset = pd.read_excel(caminho_arquivo, names=col_names).dropna()
    feature_names = [
        'Sexo_binary', 'Faixa_etaria', 'Body_mass', 'LAT', 'Menhinick', 'Like.Adjacencies_1',
        'Splitting_index_1', 'variation_fractal_dimension_index_1', 'edge_density_14', 'Splitting_index_14',
        'variation_of_euclidean_nearest.neighbor_distance_22', 'Like.Adjacencies_22', 'Splitting_index_22',
        'variation_fractal_dimension_index_22', 'bio_1', 'bio_17', 'bio_18', 'bio_2', 'bio_3'
    ]
    target = 'Animal_infectado'
    return feature_names, target, dataset

# Executando função de dataset
feature_names, target, dataset = load_dataset(caminho_arquivo)

# Definindo função de treino e teste
def data_split(feature_names, target, dataset):
    x_train, x_test, y_train, y_test = train_test_split(dataset.loc[:, feature_names],
                                                        dataset.loc[:, target],
                                                        test_size=0.33, shuffle=True, random_state=50)
    return x_train, x_test, y_train, y_test

# Executando função de treino e teste
x_train, x_test, y_train, y_test = data_split(feature_names, target, dataset)

# Função de scorer personalizado para AUPRC
def custom_auprc_scorer(y_true, y_pred):
    y_true_mapped = np.where(y_true == 'positivo', 1, 0)
    y_pred_numeric = np.where(y_pred == 'positivo', 1, 0)
    return average_precision_score(y_true_mapped, y_pred_numeric)

# Criar o scorer final com make_scorer
custom_auprc_scorer = make_scorer(custom_auprc_scorer, greater_is_better=True)

scorer = {'AUPRC': custom_auprc_scorer, 'ROC_AUC': 'roc_auc'}

# Definindo parametros de CV
kfold = StratifiedKFold(n_splits=5, shuffle=True)

# Inicialize uma lista para armazenar os melhores modelos
best_models = []

# Função para avaliar o algoritmo com GridSearchCV
def evaluate_algorithm(x_train, y_train, kfold, scorer):
    gbm = GradientBoostingClassifier(loss='log_loss')
    smote = SMOTE(sampling_strategy='minority')
    pipeline = Pipeline(steps=[('smote', smote), ('gbm', gbm)])

# SEMPRE CONFIRMAR NO DOCUMENTO DO ARTIGO
    parameters = {
        'smote__k_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'gbm__learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1],
        'gbm__subsample': [0.4, 0.5, 0.6, 0.8, 1],
        'gbm__n_estimators': [50, 100, 200, 300, 400, 600],
        'gbm__max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
        'gbm__min_samples_split': [2, 3, 4, 5],
        'gbm__min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    }

    grid_gbm = GridSearchCV(estimator=pipeline, param_grid=parameters, cv=kfold,
                            verbose=1, n_jobs=-1, refit='AUPRC', scoring=scorer)

    grid_gbm.fit(x_train, y_train)

    auprc_score = grid_gbm.cv_results_['mean_test_AUPRC'][grid_gbm.best_index_]
    roc_score = grid_gbm.cv_results_['mean_test_ROC_AUC'][grid_gbm.best_index_]

    # Melhor modelo do gridsearch
    model_gbm = grid_gbm.best_estimator_
    smote_params_best = model_gbm.named_steps['smote'].get_params()
    gbm_params_best = model_gbm.named_steps['gbm'].get_params()

    # Depuração
    debug_params(grid_gbm.best_params_, gbm_params_best, gbm_params_best)

    return smote_params_best, gbm_params_best, model_gbm, roc_score, auprc_score

def run_multiple_gbm_iterations(x_train, y_train, kfold, scorer, num_iterations=10):
    all_models_grid_search = []
    best_auprc_score = float('-inf')
    best_roc_auc_score = float('-inf')
    best_model = None
    best_model_iteration = None  # Guarda a iteração do melhor modelo

    with open('all_models_grid_search.txt', 'w') as f_all, open('best_model.txt', 'w') as f_best:
        for i in range(num_iterations):
            best_models = []

            smote_params, gbm_params, model_gbm, roc_score, auprc_score = evaluate_algorithm(x_train, y_train, kfold, scorer)
            best_models.append((auprc_score, roc_score, smote_params, gbm_params, model_gbm))
            all_models_grid_search.append((auprc_score, roc_score, smote_params, gbm_params, model_gbm))

            # Atualiza o melhor modelo
            if auprc_score > best_auprc_score:
                best_auprc_score = auprc_score
                best_roc_auc_score = roc_score
                best_model = model_gbm
                best_model_iteration = i  # Atualiza a iteração do melhor modelo
            elif auprc_score == best_auprc_score and roc_score > best_roc_auc_score:
                best_roc_auc_score = roc_score
                best_model = model_gbm
                best_model_iteration = i  # Atualiza a iteração do melhor modelo

            # Salva todos os modelos da iteração atual no arquivo de texto
            save_models_to_file(f_all, best_models, i)

        # Salva o melhor modelo em um arquivo separado
        if best_model is not None:
            save_models_to_file(f_best, [(best_auprc_score, best_roc_auc_score, smote_params, gbm_params, best_model)],
                                best_model_iteration)

    return all_models_grid_search, best_model, best_roc_auc_score, best_auprc_score

def save_models_to_file(file, models, iteration):
    file.write("Lista dos modelos da iteração {}:\n".format(iteration + 1))
    for idx, (auprc_score, roc_score, smote_params, gbm_params, model) in enumerate(models, start=1):
        file.write("AUPRC Score: {}\n".format(auprc_score))
        file.write("ROC AUC Score: {}\n".format(roc_score))
        file.write("SMOTE Params: {}\n".format(smote_params))
        file.write("GBM Params: {}\n".format(gbm_params))
        file.write("Modelo:\n{}\n".format(model))
        file.write('\n')

# Supondo que evaluate_algorithm, x_train, y_train, kfold e scorer já estejam definidos
all_models_grid_search, best_model, best_roc_auc_score, best_auprc_score = run_multiple_gbm_iterations(x_train, y_train, kfold, scorer, num_iterations=10)

print("Melhor modelo:")
print("AUPRC Score:", best_auprc_score)
print("ROC AUC Score:", best_roc_auc_score)
print("Modelo:", best_model)

smote_params_best = best_model.named_steps['smote'].get_params()
gbm_params_best = best_model.named_steps['gbm'].get_params()

def feature_selection_rfe(gbm_params_best, smote_params_best, x_train, x_test, y_train, kfold, feature_names, iteration, output_file):
    smote_best = SMOTE(**smote_params_best)
    x_train_resampled, y_train_resampled = smote_best.fit_resample(x_train, y_train)

    rfe = RFECV(estimator=GradientBoostingClassifier(**gbm_params_best), step=1, cv=kfold, scoring=custom_auprc_scorer, n_jobs=-1, verbose=0)
    rfe.fit(x_train_resampled, y_train_resampled)

    selected_features_indices = rfe.support_
    selected_feature_names = [feature_names[i] for i, selected in enumerate(selected_features_indices) if selected]

    # Utilizando cv_results_ para obter o melhor score
    best_score = max(rfe.cv_results_['mean_test_score'])

    print(f"Iteração {iteration}: Nomes das features selecionadas:", selected_feature_names)
    print(f"Iteração {iteration}: Melhor score: {best_score}")

    with open(output_file, 'a') as file:
        file.write(f"Iteração {iteration}\n")
        file.write(f"Melhor score: {best_score}\n")
        file.write("Features selecionadas:\n")
        for feature in selected_feature_names:
            file.write(f"{feature}\n")
        file.write("\n")  # Adiciona uma linha em branco entre as iterações

    # Depuração
    debug_params(gbm_params_best, gbm_params_best, gbm_params_best)

    x_train_selected = rfe.transform(x_train_resampled)
    x_test_selected = rfe.transform(x_test)

    return x_train_selected, x_test_selected, y_train_resampled, selected_feature_names, best_score

def multiple_feature_selection_iterations(gbm_params_best, smote_params_best, x_train, x_test, y_train, kfold, feature_names, n_iterations=10, output_file="features_selection_results.txt"):
    if os.path.exists(output_file):
        os.remove(output_file)

    best_combination = None
    best_score_overall = -np.inf
    best_features_overall = None

    for i in range(1, n_iterations + 1):
        x_train_selected, x_test_selected, y_train_resampled, selected_feature_names, best_score = feature_selection_rfe(gbm_params_best, smote_params_best, x_train, x_test, y_train, kfold, feature_names, i, output_file)

        if best_score > best_score_overall:
            best_score_overall = best_score
            best_features_overall = selected_feature_names
            best_combination = (x_train_selected, x_test_selected, y_train_resampled)

    with open('best_features_importance.txt', 'w') as file:
        file.write("Importância das features:\n")
        for feature_name in best_features_overall:
            file.write(f"{feature_name}\n")

    return best_combination, best_features_overall

# Executar o processo
best_combination, best_features_overall = multiple_feature_selection_iterations(gbm_params_best, smote_params_best, x_train, x_test, y_train, kfold, feature_names)

x_train_best, x_test_best, y_train_best = best_combination

# Imprimir as melhores features do melhor modelo
print("Melhores features do melhor modelo:")
for feature in best_features_overall:
    print(feature)

# Listas para armazenar os resultados de cada iteração
auprc_scores = []
roc_auc_scores = []

# Realizar 10 iterações de validação cruzada usando 5 folds
for iteration in range(10):
    # Cross-validation com AUPRC
    cv_results_gbm_auprc = cross_val_score(GradientBoostingClassifier(**gbm_params_best), x_train_best, y_train_best, cv=kfold, scoring=custom_auprc_scorer, n_jobs=-1)
    auprc_scores.append(cv_results_gbm_auprc)

    # Cross-validation com ROC-AUC
    cv_results_gbm_roc_auc = cross_val_score(GradientBoostingClassifier(**gbm_params_best), x_train_best, y_train_best, cv=kfold, scoring='roc_auc', n_jobs=-1)
    roc_auc_scores.append(cv_results_gbm_roc_auc)

# Converter listas em arrays numpy
auprc_scores = np.array(auprc_scores)
roc_auc_scores = np.array(roc_auc_scores)

# Calcular médias e desvios padrão globais
auprc_mean_global = auprc_scores.mean()
roc_auc_mean_global = roc_auc_scores.mean()
auprc_std_global = auprc_scores.std()
roc_auc_std_global = roc_auc_scores.std()

# Calcular a melhor iteração em termos de média
best_iteration_idx = auprc_scores.mean(axis=1).argmax()
auprc_best_iteration = auprc_scores[best_iteration_idx].mean()
roc_auc_best_iteration = roc_auc_scores[best_iteration_idx].mean()

# Exibir os resultados da validação cruzada
print("Média da AUPRC na validação cruzada:", auprc_mean_global)
print("Desvio padrão da AUPRC na validação cruzada:", auprc_std_global)
print("Média do ROC-AUC na validação cruzada:", roc_auc_mean_global)
print("Desvio padrão do ROC-AUC na validação cruzada:", roc_auc_std_global)

# Exibir os resultados da melhor iteração
print("Média da AUPRC na melhor iteração:", auprc_best_iteration)
print("Desvio padrão da AUPRC na melhor iteração:", auprc_scores[best_iteration_idx].std())
print("Média do ROC-AUC na melhor iteração:", roc_auc_best_iteration)
print("Desvio padrão do ROC-AUC na melhor iteração:", roc_auc_scores[best_iteration_idx].std())

# Salvar resultados da validação cruzada em arquivos .txt
with open('cv_results_gbm_auprc_all.txt', 'w') as f:
    f.write(f'AUPRC Scores\nMédia Global: {auprc_mean_global}\nDesvio Padrão Global: {auprc_std_global}\n\n')
    f.write(np.array2string(auprc_scores, separator=', '))
    f.write(f'\n\nMédia Melhor Iteração: {auprc_best_iteration}\nDesvio Padrão Melhor Iteração: {auprc_scores[best_iteration_idx].std()}')

with open('cv_results_gbm_roc_auc_all.txt', 'w') as f:
    f.write(f'ROC-AUC Scores\nMédia Global: {roc_auc_mean_global}\nDesvio Padrão Global: {roc_auc_std_global}\n\n')
    f.write(np.array2string(roc_auc_scores, separator=', '))
    f.write(f'\n\nMédia Melhor Iteração: {roc_auc_best_iteration}\nDesvio Padrão Melhor Iteração: {roc_auc_scores[best_iteration_idx].std()}')

# Treinar modelo com as features selecionadas pelo RFECV
trained_best_model = GradientBoostingClassifier(**gbm_params_best).fit(x_train_best, y_train_best)

# Depuração
debug_params(gbm_params_best, gbm_params_best, gbm_params_best)

# Abrir o arquivo para salvar os resultados
with open('performance_results.txt', 'w') as f:
    print("\n=========================================================================", file=f)
    print("Access the parameters of the trained model", file=f)
    # Acessar os parâmetros do modelo treinado
    print(trained_best_model.get_params(deep=True), file=f)
    print('=========================================================', file=f)
    # Realizando predições no dataset de teste 'x_test'
    pred_labels_gbm = trained_best_model.predict(x_test_best)
    pred_proba_gbm = trained_best_model.predict_proba(x_test_best)
    print(file=f)
    print('=========================================================', file=f)
    # Avaliação de desempenho
    print('Evaluation of the trained model Gradient Boosting: ', file=f)
    print(file=f)
    print('=========================================================', file=f)
    accuracy = accuracy_score(y_test, pred_labels_gbm)
    print('Accuracy Gradient Boosting:', accuracy, file=f)
    print(file=f)
    print('=========================================================', file=f)
    precision = precision_score(y_test, pred_labels_gbm, pos_label='positivo')
    print('Precision Gradient Boosting:', precision, file=f)
    print(file=f)
    print('=========================================================', file=f)
    recall = recall_score(y_test, pred_labels_gbm, pos_label='positivo')
    print('Recall Score Gradient Boosting:', recall, file=f)
    print(file=f)
    print('=========================================================', file=f)
    f1 = f1_score(y_test, pred_labels_gbm, pos_label='positivo')
    print('F1 Score Gradient Boosting:', f1, file=f)
    print(file=f)
    print('=========================================================', file=f)
    confusion_mat = confusion_matrix(y_test, pred_labels_gbm)
    print('Confusion Matrix Gradient Boosting:\n', confusion_mat, file=f)
    print(file=f)
    print('=========================================================', file=f)
    class_report = classification_report(y_test, pred_labels_gbm)
    print('Classification Report Gradient Boosting:\n', class_report, file=f)
    print(file=f)
    print('=========================================================', file=f)
    kappa_score = cohen_kappa_score(y_test, pred_labels_gbm)
    print('Kappa Score Gradient Boosting:', kappa_score, file=f)
    print(file=f)
    print('=========================================================', file=f)

# Diretório onde os plots serão salvos
save_dir = 'plots/'

# Verificar se o diretório existe, senão, criar
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Curva de aprendizado
skplt.estimators.plot_learning_curve(best_model, x_train_best, y_train_best, figsize=(8, 6))
plt.title("Curva de Aprendizado (Gradient Boosting)")
plt.savefig(os.path.join(save_dir, 'learning_curve.png'))  # Salvar a curva de aprendizado
plt.close()

# ROC Curve
skplt.metrics.plot_roc(y_test, pred_proba_gbm, figsize=(8, 6))
plt.savefig(os.path.join(save_dir, 'roc_curve.png'))  # Salvar a curva ROC
plt.close()

# Matriz de Confusão
skplt.metrics.plot_confusion_matrix(y_test, pred_labels_gbm, figsize=(8, 6))
plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))  # Salvar a matriz de confusão
plt.close()

# Curva de Precisão-Revocação
skplt.metrics.plot_precision_recall(y_test, pred_proba_gbm, title='Precision-Recall Curve', plot_micro=True, figsize=(8, 6))
plt.savefig(os.path.join(save_dir, 'precision_recall_curve.png'))  # Salvar a curva de precisão e revocação
plt.close()

# Explanação SHAP
explainer = shap.TreeExplainer(trained_best_model)
shap_values = explainer.shap_values(x_test_best)
shap_importance = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({'features': best_features_overall,
                              'importance': shap_importance})
importance_df.sort_values(by='importance', ascending=False, inplace=True)

shap_exp = shap.Explanation(values=shap_values, base_values=explainer.expected_value, data=x_test_best,
                            feature_names=best_features_overall)

# Gráfico de abelhas SHAP
shap.plots.beeswarm(shap_exp, max_display=len(best_features_overall), show=False)
plt.ylim(-0.5,
         len(best_features_overall) - 0.5)  # Defina os limites do eixo Y para evitar o corte dos nomes das features
plt.subplots_adjust(left=0.5, right=0.9)  # Ajuste as margens esquerda e direita do gráfico
plt.savefig(os.path.join(save_dir, 'shap_beeswarm.png'))
plt.close()

# Gráfico de barras SHAP
shap.plots.bar(shap_exp, max_display=len(best_features_overall), show=False)
plt.subplots_adjust(left=0.5, right=0.9)  # Ajuste as margens esquerda e direita do gráfico
plt.savefig(os.path.join(save_dir, 'shap_bar.png'))
plt.close()

# Adicionando a linha para salvar a importância das features em um arquivo .txt
with open('importance_of_features.txt', 'a') as file:
    file.write("\n\nExplanação SHAP - Importância das Features\n")
    file.write(importance_df.to_string(index=False))

def save_model(model, filename):
    with open(filename, 'wb') as f:
        pk.dump(model, f)

# Save the trained model
save_model(trained_best_model, '10km_mass_idade.pickle')

