import mlflow
import yaml
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from joblib import load
import os
import calendar
from datetime import datetime

# Définir les chemins
predictions_save_path = "C:\\Users\\amine\\Downloads\\riskkcreditpredicted.csv"

with open("C:\\Users\\amine\\Downloads\\mlflowm\\config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

model_path = config['model']['path']
tracking_uri = config['tracking_uri']
experiment_name = config['experiment']['name']
test_data_path = config['data']['test_data_path']
data_path=config['data']['data_path']
target_path = config['data']['target_path']
metrics_save_path = config['metrics']['save_path']
plot_save_path = config['metrics']['plot_path']
plot_save_path1 = config['metrics']['plot_path1']

# Initialiser MLflow
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(experiment_name)
model = load(model_path)
datat=pd.read_csv(test_data_path)
# Lire les données avec pandas
X_test = pd.read_csv(test_data_path)
y_test = pd.read_csv(target_path)

# Assurez-vous que 'y_test' est une série
y_test = y_test.squeeze()

# Vérifier les données lues
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)
print("X_test preview:\n", X_test.head())
print("y_test preview:\n", y_test.head())

# Liste des colonnes attendues par le modèle
expected_columns = ['Customer_ID', 'Month', 'lag_1', 'lag_2', 'lag_3', 'rolling_mean', 'rolling_std', 'trend', 'seasonal', 'residual', 'fft_real', 'fft_imag']

# Ajouter la colonne 'Customer_ID' avec des valeurs par défaut (par exemple, des indices)
if 'Customer_ID' not in X_test.columns:
    X_test['Customer_ID'] = range(len(X_test))

# Convertir les colonnes en types numériques ou catégoriels
X_test = X_test.apply(pd.to_numeric, errors='coerce')

# Ajouter les colonnes manquantes pour correspondre aux colonnes attendues
missing_columns = set(expected_columns) - set(X_test.columns)
for col in missing_columns:
    X_test[col] = np.nan

# Réordonner les colonnes pour correspondre à l'ordre attendu
X_test = X_test[expected_columns]

# Convertir X_test en DMatrix
X_test_dmatrix = xgb.DMatrix(X_test)

# Calculer les métriques
y_pred = model.predict(X_test_dmatrix)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test, y_pred)

# Sauvegarder les métriques dans un fichier JSON
metrics = {
    "mse": mse,
    "rmse": rmse,
    "mape": mape
}

with open(metrics_save_path, "w") as metrics_file:
    json.dump(metrics, metrics_file)

# Chemin du fichier de compteur
counter_file = "C:\\Users\\amine\\Downloads\\mlflowm\\counter.txt"

def get_next_run_number(counter_file):
    if not os.path.exists(counter_file):
        with open(counter_file, 'w') as f:
            f.write("1")
        return 1
    with open(counter_file, 'r') as f:
        run_number = int(f.read().strip())
    return run_number

def update_run_number(counter_file, next_number):
    with open(counter_file, 'w') as f:
        f.write(str(next_number))

# Obtenir le numéro de run suivant
run_number = get_next_run_number(counter_file)
run_name = f"Check {run_number}"

# Démarrer l'enregistrement dans MLflow
with mlflow.start_run(run_name=run_name) as run:
    with open("C:\\Users\\amine\\Downloads\\mlflowm\\params.json", "r") as params_file:
        params = json.load(params_file)
        for key, value in params.items():
            mlflow.log_param(key, value)

    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("MAPE", mape)

    mlflow.log_artifact(model_path, artifact_path="models")
    mlflow.log_artifact(metrics_save_path, artifact_path="metrics")

    # Génération des graphiques
    data = pd.DataFrame({
        'Month': X_test['Month'],
        'Customer_ID': X_test['Customer_ID'],
        'y_test': y_test,
        'Amount_unpaid': y_pred
    })
    data['Month'] = data['Month'].apply(lambda x: f'{calendar.month_name[x]} 2024')
    months_order = [f'{calendar.month_name[i]} 2024' for i in range(1, 13)]
    data['Month'] = pd.Categorical(data['Month'], categories=months_order, ordered=True)
    data1=pd.DataFrame({'Month': X_test['Month'],
        'Customer_ID': datat['Customer_ID'],
        'Amount_unpaid': y_pred})
    data1['Month'] = data1['Month'].apply(lambda x: f'{calendar.month_name[x]} 2024')
    months_order = [f'{calendar.month_name[i]} 2024' for i in range(1, 13)]
    data1['Month'] = pd.Categorical(data1['Month'], categories=months_order, ordered=True)

    monthly_averages = data.groupby('Month', observed=True).mean().reset_index()
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_averages['Month'], monthly_averages['y_test'], marker='o', linestyle='-', color='b', label='Valeurs Réelles')
    plt.plot(monthly_averages['Month'], monthly_averages['Amount_unpaid'], marker='o', linestyle='-', color='r', label='Valeurs Prédites')
    plt.xlabel('Mois/Année')
    plt.ylabel('Impayés')
    plt.title('Moyenne des valeurs réelles VS Moyenne des valeurs prédites')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.savefig(plot_save_path)
    mlflow.log_artifact(plot_save_path)

    monthly_sums = data.groupby('Month', observed=True).sum().reset_index()
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_sums['Month'], monthly_sums['y_test'], marker='o', linestyle='-', color='b', label='Valeurs Réelles')
    plt.plot(monthly_sums['Month'], monthly_sums['Amount_unpaid'], marker='o', linestyle='-', color='r', label='Valeurs Prédites')
    plt.xlabel('Mois/Année')
    plt.ylabel('Impayés')
    plt.title('Valeurs réelles des impayés VS Valeurs prédites des impayés')
    plt.xticks(rotation=45)
    plt.ticklabel_format(style='plain', axis='y')
    plt.tight_layout()
    plt.legend()
    plt.savefig(plot_save_path1)
    mlflow.log_artifact(plot_save_path1)

    # Créer un DataFrame avec les résultats à sauvegarder
    results_df = data1.copy()
    results_df['Run_ID'] = run.info.run_id
    results_df['Execution_Date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Charger les données existantes ou créer un nouveau DataFrame
    if os.path.exists(predictions_save_path) and os.path.getsize(predictions_save_path) > 0:
        existing_df = pd.read_csv(predictions_save_path)
        existing_df = existing_df.dropna(axis=1, how='all')
        updated_df = pd.concat([existing_df, results_df], ignore_index=True)
    else:
        updated_df = results_df
    if not updated_df.empty:
        updated_df.to_csv(predictions_save_path, index=False)
        print(f"Les prédictions et les informations d'exécution ont été sauvegardées dans {predictions_save_path}")
    else:
        print("Le DataFrame est vide. Aucune donnée n'a été sauvegardée.")


    # Mettre à jour le compteur
    update_run_number(counter_file, run_number + 1)
