import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Importando los datos
temperature_df = pd.read_csv("celsius_a_fahrenheit.csv")

# Visualizacion de datos
sns.scatterplot(x=temperature_df["Celsius"], y=temperature_df["Fahrenheit"])
# plt.show()

# Cargando los datos
X_train = temperature_df["Celsius"]

Y_train = temperature_df["Fahrenheit"]

# Crear Modelo
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_shape=[1]))

# model.summary()

# Compilado
model.compile(optimizer=tf.keras.optimizers.Adam(1), loss="mean_squared_error")

# Entrenando el modelo
epochs_hist = model.fit(x=X_train, y=Y_train, epochs=100)

# Evaluando modelo
epochs_hist.history.keys()

# graficando
plt.plot(epochs_hist.history["loss"])
# plt.show()

model.get_weights()

# Predicciones
Temp_C = 0
Temp_F = model.predict([Temp_C])
print(Temp_F)
