import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import pickle

from sklearn.preprocessing import LabelEncoder
from mpl_toolkits import mplot3d
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


pickle_reader = open("fruit-classifier.pkl", "rb")
fruit_classifier = pickle.load(pickle_reader)
pickle_reader = open("input-scaler.pkl", "rb")
input_scaler = pickle.load(pickle_reader)
fig = plt.figure(figsize=(12, 5))


def get_class_probabilities(input_array):
    scaled_input = input_scaler.transform([input_array])
    return fruit_classifier.predict_proba(scaled_input)


def load_citrus_dataframe():
    return pd.read_csv("citrus.csv")


def encode_name_data():
    return list(map((lambda name: 0 if name == "grapefruit" else 1), citrus_data["name"]))


def generate_fruit_proportion_graph():
    graph_axes = fig.add_subplot(1, 2, 1, label="ax1")
    graph_axes.set(title="Grapefruit v. Orange Proportions",
                   xlabel="Diameter (cm)", ylabel="Weight (g)")
    diameter_v_weight_scatter = graph_axes.scatter(citrus_data["diameter"], citrus_data["weight"],
                                                   c=encoded_names)
    plot_handles = diameter_v_weight_scatter.legend_elements()[0]
    graph_axes.legend(title="Fruit Type", handles=plot_handles,
                      labels=["Grapefruit", "Orange"])


def generate_color_distribution_graph():
    graph_axes = fig.add_subplot(1, 2, 2, projection='3d', label="ax2")
    graph_axes.set(title="Grapefruit v. Orange Color Distribution (by RGB Value)", xlabel="red",
                   ylabel="green", zlabel="blue")
    graph_axes.scatter3D(citrus_data["red"], citrus_data["green"], citrus_data["blue"],
                         c=encoded_names)
    graph_axes.view_init(15, 60)


def render_userinput_and_prediction_display(col1, col2):
    with col1:
        diameter_input = st.text_input("Diameter (cm)")
        weight_input = st.text_input("Weight (g)")
        red_input = st.text_input("Red (0 - 255)")
        green_input = st.text_input("Green (0 - 255)")
        blue_input = st.text_input("Blue (0 - 255)")
    with col2:
        prediction_placeholder = st.empty()
        output_placeholder = st.empty()
        prediction_output = prediction_placeholder.text_input(
            "This fruit is most likely a...")
        output_confidence = output_placeholder.metric(
            "Prediction Confidence", value="")
        if st.button("Predict"):
            class_probabilities = get_class_probabilities([diameter_input, weight_input, red_input,
                                                           green_input, blue_input])
            prediction_output = prediction_placeholder.text_input(
                "This fruit is most likely a(n)...", value=classify_fruit(class_probabilities))
            output_confidence = output_placeholder.metric(
                "Prediction Confidence", value=select_guess_confidence(class_probabilities))
            if classify_fruit(class_probabilities) == "Grapefruit":
                # Photo from https://unsplash.com/photos/GLFt9RL9kDY
                st.image("grapefruit.jpg")
            else:
                # Photo from https://unsplash.com/photos/jBHv766AKrE
                st.image("orange.jpg")


def classify_fruit(class_probabilities):
    return ("Grapefruit" if
            class_probabilities[0][0] >= class_probabilities[0][1]
            else "Orange")


def select_guess_confidence(class_probabilities):
    grapefruit_guess_confidence = class_probabilities[0][0]
    orange_guess_confidence = class_probabilities[0][1]
    return (grapefruit_guess_confidence if
            grapefruit_guess_confidence >= orange_guess_confidence
            else orange_guess_confidence)


def render_dataset_metrics(fig):
    st.markdown("### Dataset Metrics")
    generate_fruit_proportion_graph()
    generate_color_distribution_graph()
    with st.empty():
        fig


def render_header():
    st.title("Citrificate")
    st.markdown("### Fruit Parameters")


citrus_data = load_citrus_dataframe()
encoded_names = encode_name_data()


def main():
    render_header()
    col1, col2 = st.columns(2)
    render_userinput_and_prediction_display(col1, col2)
    render_dataset_metrics(fig)
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### ML Training/Test Dataset")
        st.dataframe(citrus_data)
    with col4:
        st.image("confusion-matrix-test-results.png")
        st.caption(
            "Visualized false positive and false negative results from the ML model's final test on a dataset of 1600 samples.")


main()
