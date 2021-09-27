import json
import pathlib

import pandas as pd
import matplotlib.pyplot as plt

# import data
df = pd.read_csv("grid_search_results.csv", sep=";")
x_eer = df["learning rate"].tolist()
y_eer = df["eer"].tolist()

data_path = pathlib.Path(
    "wav2vec2-sv-ce/23fb5940c4c94ab39ff4ab74c3852857/lr_find_20210907-215014.json"
)

with data_path.open("r") as f:
    data = json.load(f)

x_loss = data["data"]["lr"]
y_loss = data["data"]["loss"]

# draw graph
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

# scatter plot of EER result at certain LR values
eer_scatter_down = ax1.scatter(x=x_eer, y=y_eer, marker="x")
eer_scatter_up = ax2.scatter(x=x_eer, y=y_eer, marker="x")

ax1.set_ylabel("EER")
ax2.set_ylabel("EER")

ax1.set_ylim(0.45, 0.55)
ax2.set_ylim(0, 0.07)

ax1.set_xscale("log")
ax2.set_xscale("log")

# line plot of learning rate versus loss
(loss_line,) = ax3.plot(x_loss, y_loss, "C1")

ax3.set_xscale("log")
ax3.set_xlabel("learning rate")
ax3.set_ylabel("loss")
ax3.set_xlim(1e-6, 5e-2)

# hide the spines between ax1 and ax2, and ax2 and ax3
ax1.spines.bottom.set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False, labelbottom=False)

ax2.spines.top.set_visible(False)
ax2.spines.bottom.set_visible(False)
ax2.tick_params(bottom=False, top=False, labeltop=False, labelbottom=False)
ax2.xaxis.set_visible(False)

ax3.spines.top.set_visible(False)
ax3.xaxis.tick_bottom()
ax3.tick_params(labeltop=False)

# add discontinuity
d = 0.5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(
    marker=[(-1, -d), (1, d)],
    markersize=12,
    linestyle="none",
    color="k",
    mec="k",
    mew=1,
    clip_on=False,
)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

# create legend
plt.legend([eer_scatter_down, loss_line], ["EER", "loss"], loc=2)

# show plot
plt.show()
