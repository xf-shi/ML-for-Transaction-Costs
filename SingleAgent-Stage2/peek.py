import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

with open("eva.txt", "r") as f:
    G_MAP = np.array([float(x.strip()) for x in f.readlines()])
x = np.linspace(0, 50, 500000 + 1)
df = pd.DataFrame.from_dict({"X": x, "Y": G_MAP})
df = df[df["X"] <= 5]
plt.plot(df["X"], df["Y"])
plt.axline((0, 0), (5, -5), color = "red")
plt.show()
