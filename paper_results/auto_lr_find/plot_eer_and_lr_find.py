import json
import pathlib

import pandas as pd
import matplotlib.pyplot as plt

network_name = "xvector"
zoomed = True
zoom_min = 0
zoom_max = 0.15

# import data
df = pd.read_csv("grid_search_results.csv", sep=",")
df = df.loc[df['network'] == network_name]
x_eer = df["learning rate"].tolist()
y_eer = df["eer"].tolist()

data_path = pathlib.Path(
    f"{network_name}/data.json"
)

with data_path.open("r") as f:
    data = json.load(f)

x_loss = data["data"]["lr"]
y_loss = data["data"]["loss"]

# draw graph
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)

# line plot of learning rate versus loss
loss_line, = ax1.plot(
    x_loss,
    y_loss,
    "C1"
)

ax1.set_xscale("log")
ax1.set_xlabel("learning rate")
ax1.set_ylabel("loss")

# scatter plot of EER result at certain LR values
ax2 = plt.twinx()

eer_scatter = ax2.scatter(x=x_eer, y=y_eer, marker="x")

ax2.set_ylabel("EER")
if zoomed:
    ax2.set_ylim(zoom_min, zoom_max)
else:
    ax2.set_ylim(0, 0.6)

plt.legend([loss_line, eer_scatter], ["loss", "EER"], loc=2)
plt.suptitle(network_name)

plt.savefig(f'{network_name}/plot_lr_eer{"_zoomed" if zoomed else ""}.png')
plt.show()
