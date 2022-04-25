import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.preprocessing import LabelEncoder
from mpl_toolkits import mplot3d

# Create dataset metrics plot.
'''
# Dataset Metrics
'''

citrus_data = pd.read_csv("citrus.csv")

encoded_name = list(map((lambda name: 0 if name == "grapefruit" else 1),
                        citrus_data["name"]))
# Create figure to contain subplots.
fig = plt.figure(figsize=(12, 5))

# Create first axis and subplot.
ax1 = fig.add_subplot(1, 2, 1, label="ax1")
ax1.set(title="Grapefruit v. Orange Proportions",
        xlabel="Diameter (cm)",
        ylabel="Weight (g)")
diameter_v_weight_scatter = ax1.scatter(citrus_data["diameter"],
                                        citrus_data["weight"],
                                        c=encoded_name)
plot_handles = diameter_v_weight_scatter.legend_elements()[0]
ax1.legend(title="Fruit Type", handles=plot_handles,
           labels=["Grapefruit", "Orange"])

# Create second axis and subplot.
ax2 = fig.add_subplot(1, 2, 2, projection='3d', label="ax2")
ax2.set(title="Grapefruit v. Orange Color Distribution (by RGB Value)",
        xlabel="red",
        ylabel="green",
        zlabel="blue")
color_scatter = ax2.scatter3D(citrus_data["red"],
                              citrus_data["green"],
                              citrus_data["blue"],
                              c=encoded_name)
ax2.view_init(15, 60)

fig
