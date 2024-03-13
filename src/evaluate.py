from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_multi_output_model(predictions, actual, metrics=['MSE', 'MAE', 'R2']):
    """
    Évalue les performances d'un modèle multi-sorties en utilisant différentes métriques.

    :param predictions: Les prédictions du modèle. Attendu d'avoir une forme [n_samples, n_outputs].
    :param actual: Les valeurs réelles. Doit avoir la même forme que les prédictions.
    :param metrics: Liste des métriques d'évaluation à utiliser. Par défaut : MSE, MAE et R².
    :return: Un dictionnaire de dictionnaires contenant les scores pour chaque métrique spécifiée pour chaque cible.
    """
    results = {}
    n_outputs = predictions.shape[1]

    for i in range(n_outputs):
        target_results = {}
        if 'MSE' in metrics:
            target_results['MSE'] = mean_squared_error(actual[:, i], predictions[:, i])
        if 'MAE' in metrics:
            target_results['MAE'] = mean_absolute_error(actual[:, i], predictions[:, i])
        if 'R2' in metrics:
            target_results['R2'] = r2_score(actual[:, i], predictions[:, i])

        results[f'Target {i + 1}'] = target_results

    return results

def print_evaluation_results(results):
    """
    Affiche les résultats d'évaluation pour un modèle multi-sorties.

    :param results: Dictionnaire contenant les résultats d'évaluation pour chaque cible.
    """
    print("Résultats de l'évaluation multi-sorties :")
    for target, metrics in results.items():
        print(f"\n{target}:")
        for metric, score in metrics.items():
            print(f"  {metric}: {score}")
