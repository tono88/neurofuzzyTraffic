import pandas as pd
from sklearn.preprocessing import MinMaxScaler  # o el scaler que hayas usado

# Paso 1/6: Cargando los datos
print("Paso 1/6: Cargando datos desde Excel...")
df = pd.read_excel('D:/Doctorado/BolognaSUMO/NeuroFuzzy/details_seed1096.xlsx')
print(f"   → Total de filas: {len(df)}\n")

# Paso 2/6: Mostrando rangos iniciales
print("Paso 2/6: Estadísticas antes de des-normalizar")
print("  y_true  :", df['y_true'].describe().to_dict())
print("  y_pred  :", df['y_pred'].describe().to_dict())
print("  med_lb  :", df['med_lb'].describe().to_dict())
print("  med_ub  :", df['med_ub'].describe().to_dict(), "\n")

# Paso 3/6: Cargando el scaler
# Sustituye aquí la forma en que guardaste tu scaler tras el fit() en entrenamiento.
# Por ejemplo, si lo guardaste con joblib:
# from joblib import load
# scaler_y = load('scaler_y.joblib')
# En este ejemplo supongo un MinMaxScaler con rango (0, 500)
scaler_y = MinMaxScaler(feature_range=(0, 500))
scaler_y.min_, scaler_y.scale_ = (0,), (1/500,)  # ejemplo de parámetros

print("Paso 3/6: Parámetros del scaler cargado")
print("   feature_range:", scaler_y.feature_range)
print("   min_         :", scaler_y.min_)
print("   scale_       :", scaler_y.scale_, "\n")

# Paso 4/6: Des-normalizando predicciones
print("Paso 4/6: Aplicando inverse_transform a y_pred, med_lb y med_ub...")
y_pred_inv  = scaler_y.inverse_transform(df[['y_pred']])
med_lb_inv  = scaler_y.inverse_transform(df[['med_lb']])
med_ub_inv  = scaler_y.inverse_transform(df[['med_ub']])
print("   ✔ Des-normalización completa\n")

# Paso 5/6: Montando el DataFrame resultante
print("Paso 5/6: Construyendo DataFrame con valores reales")
df_real = df[['step', 'lane_id', 'y_true']].copy()
df_real['y_pred_real'] = y_pred_inv
df_real['lb_real']     = med_lb_inv
df_real['ub_real']     = med_ub_inv
print("   → Primeras 10 filas tras des-normalizar:")
print(df_real.head(10), "\n")

# Paso 6/6: Fin del proceso
print("Paso 6/6: ¡Listo! Ahora y_pred_real está en la misma escala que y_true.")
