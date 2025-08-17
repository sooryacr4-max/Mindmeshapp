# mindmesh_app.py

import re
import joblib
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Load model
model = joblib.load('mindmesh_model.pkl')  # Only used for geometry classification

st.title("🧠 MindMesh: Natural Language to Mesh Generator")

# Default values
DEFAULT_LENGTH = 100.0  # mm
DEFAULT_HEIGHT = 20.0   # mm
DEFAULT_BIAS_FACTOR = 10.0  # ANSYS-like bias factor

# NLP extractors
def extract_length(text):
    match = re.search(r'(\d+(?:\.\d+)?)\s*mm.*length', text.lower())
    return float(match.group(1)) if match else DEFAULT_LENGTH

def extract_height(text):
    match = re.search(r'(\d+(?:\.\d+)?)\s*mm.*height', text.lower())
    return float(match.group(1)) if match else DEFAULT_HEIGHT

def detect_bias(text):
    text = text.lower()
    if "no bias" in text or "without bias" in text:
        return "none"
    elif "bias" in text:
        return "pending"
    return "none"

# ANSYS-like biasing (exponential spacing)
def ansys_biased_spacing(n, L, bias_factor):
    r = bias_factor ** (1 / (n - 1))
    spacing = L * (1 - r) / (1 - r**n)
    coords = [0]
    for i in range(1, n):
        coords.append(coords[-1] + spacing * r**(i - 1))
    return np.array(coords)

# Mesh generators
def generate_flat_plate_mesh(length, height, nx=100, ny=40, bias='none', bias_factor=None):
    if bias == 'wall_high':
        bias_factor = float(bias_factor) if bias_factor and bias_factor != 'default' else DEFAULT_BIAS_FACTOR
        y = ansys_biased_spacing(ny, height, bias_factor)
    else:
        y = np.linspace(0, height, ny)
    x = np.linspace(0, length, nx)
    return np.meshgrid(x, y)

def generate_bfs_mesh(length, height, step_height=10.0, step_length=20.0, nx=100, ny=40, bias='none', bias_factor=None):
    x = np.linspace(0, length, nx)
    if bias == 'wall_high':
        bias_factor = float(bias_factor) if bias_factor and bias_factor != 'default' else DEFAULT_BIAS_FACTOR
        y_upper = ansys_biased_spacing(ny, height, bias_factor)
    else:
        y_upper = np.linspace(0, height, ny)
    X, Y = np.meshgrid(x, y_upper)

    step_index = int((step_length / length) * nx)
    step_rows = int((step_height / height) * ny)
    for i in range(step_rows):
        for j in range(step_index):
            Y[i, j] = np.nan
    return X, Y

# Plotter
def plot_mesh(X, Y, title):
    fig, ax = plt.subplots(figsize=(10, 3))
    for i in range(Y.shape[0]):
        ax.plot(X[i, :], Y[i, :], 'b-', lw=0.5)
    for j in range(X.shape[1]):
        ax.plot(X[:, j], Y[:, j], 'b-', lw=0.5)
    ax.set_title(title)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.grid(True)
    ax.set_aspect('equal')
    return fig

# Input prompt
user_input = st.text_area("Enter your meshing request:", placeholder="e.g., create a backward-facing step of 200 mm length and 30 mm height with wall bias")

# Generate button
if st.button("Generate Mesh") and user_input.strip():
    # Geometry prediction
    predicted_geom = model.predict([user_input])[0]

    # Rule-based extractions
    length = extract_length(user_input)
    height = extract_height(user_input)
    bias_status = detect_bias(user_input)

    # If bias mentioned but unspecified
    bias_factor = None
    if bias_status == "pending":
        bias_factor = st.text_input("Bias requested. Please specify bias factor:", value="default")
        if bias_factor.strip().lower() == "default":
            bias_status = "wall_high"
        else:
            bias_status = "wall_high"
    else:
        bias_factor = None

    # Output
    st.markdown("### 🧠 Parsed Inputs")
    st.write(f"**Geometry Type:** {predicted_geom}")
    st.write(f"**Length:** {length} mm")
    st.write(f"**Height:** {height} mm")
    st.write(f"**Bias Type:** {bias_status}")
    st.write(f"**Bias Factor:** {bias_factor}")

    # Generate and plot mesh
    if predicted_geom == 'flat_plate':
        X, Y = generate_flat_plate_mesh(length, height, bias=bias_status, bias_factor=bias_factor)
        fig = plot_mesh(X, Y, title="Flat Plate Mesh")
        st.pyplot(fig)
    elif predicted_geom == 'bfs':
        X, Y = generate_bfs_mesh(length, height, bias=bias_status, bias_factor=bias_factor)
        fig = plot_mesh(X, Y, title="Backward-Facing Step Mesh")
        st.pyplot(fig)
    else:
        st.error("Unsupported geometry type.")
