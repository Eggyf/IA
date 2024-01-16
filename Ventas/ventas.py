import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Importar datos
sales_df = pd.read_csv("datos_de_ventas.csv")

# Visualizar
sns.scatterplot(x=sales_df["Temperature"], y=sales_df["Revenue"])
# plt.show()

# Creando set de entrenamiento
X_train = sales_df["Temperature"]
Y_train = sales_df["Revenue"]

# Creando Modelo
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_shape=[1]))

model.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss="mean_squared_error")

epochs_hist = model.fit(X_train, Y_train, epochs=500)

keys = epochs_hist.history.keys()

weights = model.get_weights()

# Temp = 30
# Revenue = model.predict([Temp])
# print(Revenue)
# Grafico de prediccion
plt.scatter(X_train, Y_train, color="gray")
plt.plot(X_train, model.predict([X_train]), color="red")
plt.ylabel("Ganancia")
plt.xlabel("temperatura")
plt.title("Ganancia vs Temperatura")
plt.savefig(
    "Grafico.png",
)
