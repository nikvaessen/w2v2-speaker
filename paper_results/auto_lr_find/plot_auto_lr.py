import json
from os import path
import pathlib

import pandas as pd
import matplotlib.pyplot as plt

# import data
data_path_ce = pathlib.Path(
    "wav2vec2-sv-ce/23fb5940c4c94ab39ff4ab74c3852857/lr_find_20210907-215014.json"
)
data_path_aam = pathlib.Path(
    "wav2vec2-sv-aam/06c91df465da4d55bed874caf6fa1da5/lr_find_20210907-221822.json"
)
data_path_ctc = pathlib.Path()
data_path_bce = pathlib.Path("wav2vec2-sv-bce/65f16f5c0860494187135a30e48097c7/lr_find_20210908-171251.json")

data_path = data_path_bce
with data_path.open("r") as f:
    data = json.load(f)

x_loss = data["data"]["lr"]
y_loss = data["data"]["loss"]

# draw graph
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)

# line plot of learning rate versus loss
(loss_line,) = ax1.plot(x_loss, y_loss, "C1")

ax1.set_xscale("log")
ax1.set_xlabel("learning rate")
ax1.set_ylabel("loss")
ax1.set_ylim(0.4, 0.9)

plt.show()
