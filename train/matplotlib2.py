import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 5, 11)
y = x**2

fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes.plot(x, y, label="X Square", color="r")
axes.plot(x, x**3, label="X Cube", color="green")
axes.legend()

fig.savefig("My Plot.png", dpi=200)
