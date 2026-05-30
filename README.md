# DIY Neural Networks: The Glass Box Sandbox

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![NumPy](https://img.shields.io/badge/NumPy-Matrix_Calculus-013243?style=for-the-badge&logo=numpy)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit)
![Status](https://img.shields.io/badge/Status-Complete-success?style=for-the-badge)

**Author:** Anubhav Sarkar (B.Tech CSE | MBA-DSDA)  
**GitHub:** [@wattacodeanubhav](https://github.com/wattacodeanubhav)

## Core Purpose
"DIY Neural Networks" is an interactive, 3D educational sandbox that strips away the "black box" of modern machine learning. By restricting the core engine to pure NumPy matrix calculus, this application allows users to build a Multi-Layer Perceptron (MLP) from scratch and visually audit how it bends geometric space to solve complex data boundaries.

## Tech Stack & Tools
* **Core Engine:** Python, NumPy (Matrix Calculus)
* **Frontend & UI:** Streamlit, CSS (Glassmorphism aesthetics)
* **3D Visualization:** Plotly WebGL
* **Data Utility:** Scikit-Learn, Pandas

---

## Key Features & Methodology

* **Pure NumPy Math Engine:** The neural network (`FNN.py`) is written entirely from scratch without high-level APIs like TensorFlow or PyTorch. It dynamically calculates matrix tensor shapes, executes forward/backward propagation, and applies mathematical penalties (L1/L2 Regularization) and activation limits (ReLU, Sigmoid, Tanh).
* **3D Geometric Data Generation:** The pipeline synthesizes non-linear mathematical structures (Nested Spheres, Double Helices). These shapes are mathematically impossible to solve with standard linear regression, practically demonstrating why deep hidden layers are necessary.
* **Explainable AI (XAI) Visualizations:**
  The UI translates raw matrix math into readable 3D formats. This includes a dynamic breakdown of single-neuron anatomy, a 3D trajectory plot tracking Gradient Descent over time, and a volumetric "Frosted Glass" isosurface that renders the exact geometric decision boundary.

---

## Architecture & Data Flow

The application isolates the heavy mathematical processing from the UI rendering layer to maintain performance.

1. **Initialization:** The UI reads user inputs (dataset choice, hidden layers, learning rate) and injects custom CSS to bypass standard Streamlit constraints.
2. **Data Pipeline:** `generate_data.py` creates the 3D coordinates. The pipeline standardizes the features and enforces a strict 70/15/15 Train/Validation/Test split to prevent data leakage.
3. **Topology Rendering:** The math engine calculates the exact matrix dimensions ($W$ and $b$) for the requested architecture and renders an interactive 3D network graph.
4. **Iterative Training:** The network loops through epochs, recording snapshots of the loss and accuracy/MAE at regular intervals to plot the 3D learning trajectory.
5. **Boundary Audit:** The fully trained weights are used to predict outcomes across a massive 35x35x35 3D grid, outputting a translucent decision boundary surface.

---
## Instructions 

### Clone the GitHub Repository
git clone [https://github.com/wattacodeanubhav/diy-neural-networks.git](https://github.com/wattacodeanubhav/diy-neural-networks.git)
cd diy-neural-networks

### Installation of Dependencies
Please refer to 'requirements.txt'

### Generate the datasets
Run in your terminal- ' python generate_data.py '

### Launch the application
Run in your terminal- 
' streamlit run app.py '; ' python3 -m streamlit run app.py '

## Repository Structure

```text
├── sample_data/
│   ├── 3d_plane.csv
│   ├── 3d_spherical.csv
│   └── 3d_helix.csv
├── app.py                   # Streamlit frontend and Plotly WebGL rendering
├── FNN.py                   # NumPy neural network core engine
├── generate_data.py         # 3D dataset synthesizer
├── data_pipeline.py         # Scaling and matrix splitting logic
├── requirements.txt         
└── README.md
