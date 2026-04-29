import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import scipy.stats as stats
from glob import glob

# 1. PREPARACION DEL ENTORNO
output_dir = 'plots'
if not os.path.exists(output_dir):
	os.makedirs(output_dir)
	print(f"Directorio '{output_dir}' creado.")

# 2. CARGA DE DATOS
files = sorted(glob('data/dataset_train_*.csv'))
if not files:
	raise FileNotFoundError("No se encontraron archivos dataset_train en data/")

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# 3. CARGA DEL MODELO (Desde model_final.json)
with open('data/model_final.json', 'r') as f:
	model_data = json.load(f)

features = model_data['meta']['features']
weights = np.array(model_data['model']['weights'])
bias = model_data['model']['bias']

# 4. CALCULO DE PREDICCIONES Y RESIDUOS
X = df[features].values
y_real = df['log_price'].values
y_pred = np.dot(X, weights) + bias
residuos = y_real - y_pred

# --- FUNCION AUXILIAR PARA GUARDAR ---
def save_plot(name):
	path = os.path.join(output_dir, name)
	plt.savefig(path, bbox_inches='tight', dpi=300)
	plt.close()
	print(f"Guardado: {path}")

sns.set_theme(style="whitegrid")

# A. HISTOGRAMA (Distribucion de la variable dependiente)
plt.figure(figsize=(8, 6))
sns.histplot(df['log_price'], kde=True, color='blue')
plt.title('Histograma de Log Price')
save_plot('1_histograma_precio.png')

# B. BOX PLOT (Distribucion por categoria de privacidad)
plt.figure(figsize=(8, 6))
df['tipo_alojamiento'] = 'Otro'
if 'room_entire' in df: df.loc[df['room_entire'] == 1, 'tipo_alojamiento'] = 'Casa Completa'
if 'room_shared' in df: df.loc[df['room_shared'] == 1, 'tipo_alojamiento'] = 'Compartida'
sns.boxplot(x='tipo_alojamiento', y='log_price', data=df)
plt.title('Box Plot: Log Price por Tipo de Alojamiento')
save_plot('2_boxplot_privacidad.png')

# C. DISPERSION (Variable independiente vs Dependiente)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['accommodates'], y=y_real, alpha=0.3)
plt.title('Dispersion: Capacidad (Accommodates) vs Precio')
save_plot('3_dispersion_capacidad.png')

# D. MATRIZ DE CORRELACION
plt.figure(figsize=(10, 8))
corr = df[features + ['log_price']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
plt.title('Matriz de Correlacion')
save_plot('4_matriz_correlacion.png')

# E. REAL VS PREDICCION
plt.figure(figsize=(8, 6))
plt.scatter(y_real, y_pred, alpha=0.3, color='green')
plt.plot([y_real.min(), y_real.max()], [y_real.min(), y_real.max()], 'r--', lw=2)
plt.xlabel('Valor Real (Log)')
plt.ylabel('Prediccion (Log)')
plt.title(f'Real vs Prediccion (R²: {model_data["metrics"]["r2"]:.3f})')
save_plot('5_real_vs_pred.png')

# F. GRAFICO Q-Q (Normalidad de Residuos)
plt.figure(figsize=(8, 6))
stats.probplot(residuos, dist="norm", plot=plt)
plt.title('Grafico Q-Q de los Residuos')
save_plot('6_grafico_qq.png')

# G. RESIDUOS VS AJUSTADOS (Homocedasticidad)
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuos, alpha=0.3, color='orange')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Valores Ajustados (Predicciones)')
plt.ylabel('Residuos')
plt.title('Residuos vs Valores Ajustados')
save_plot('7_residuos_vs_ajustados.png')

# H. IMPORTANCIA DE CARACTERISTICAS
plt.figure(figsize=(10, 6))
pd.Series(weights, index=features).sort_values().plot(kind='barh', color='teal')
plt.title('Importancia de Caracteristicas (Coeficientes del Modelo)')
save_plot('8_importancia_features.png')

print("\nAnalisis completado. Todas las graficas estan en el directorio /plots")