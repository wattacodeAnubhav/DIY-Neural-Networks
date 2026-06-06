import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from data_pipeline import DataPipeline
from FNN import GlassBoxNetwork

# --- PAGE SETUP & CUSTOM BACKGROUND ---
st.set_page_config(page_title="DIY Neural Networks", layout="wide")

page_bg_css = """
<style>
/* 1. NUKE ALL STREAMLIT BACKGROUND LAYERS */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stMain"] {
    background: transparent !important;
    background-color: transparent !important;
}

/* 2. Base dark color */
body {
    background-color: #060b21 !important;
}

/* 3. The cloud container */
.cloud-canvas {
    position: fixed;
    top: 0; left: 0; width: 100vw; height: 100vh;
    z-index: -1; 
    pointer-events: none; 
    overflow: hidden;
}

/* 4. Base cloud physics */
.cloud {
    position: absolute; width: 70vw; height: 70vw; border-radius: 50%;
    filter: blur(90px); opacity: 0.6; mix-blend-mode: screen; 
    animation: drift 25s infinite alternate ease-in-out;
}

/* 5. The Colors */
.cloud-blue { background: radial-gradient(circle, rgba(33, 150, 243, 0.7) 0%, transparent 60%); top: -20%; left: -20%; animation-duration: 28s; }
.cloud-cyan { background: radial-gradient(circle, rgba(0, 210, 255, 0.7) 0%, transparent 60%); bottom: -20%; right: -10%; animation-duration: 32s; animation-delay: -5s; }
.cloud-purple { background: radial-gradient(circle, rgba(156, 39, 176, 0.6) 0%, transparent 60%); top: 30%; left: 40%; animation-duration: 35s; animation-delay: -12s; }
.cloud-pink { background: radial-gradient(circle, rgba(233, 30, 99, 0.6) 0%, transparent 60%); top: -10%; right: 10%; animation-duration: 24s; animation-delay: -8s; }
.cloud-green { background: radial-gradient(circle, rgba(0, 200, 83, 0.5) 0%, transparent 60%); bottom: 10%; left: -10%; animation-duration: 30s; animation-delay: -18s; }

/* 6. The Movement */
@keyframes drift {
    0%   { transform: translate(0, 0) scale(1); }
    33%  { transform: translate(15vw, -15vh) scale(1.2); }
    66%  { transform: translate(-15vw, 15vh) scale(0.9); }
    100% { transform: translate(5vw, 5vh) scale(1.1); }
}

/* --- UI GLASSMORPHISM UPGRADES --- */
[data-testid="stSidebar"] { background-color: rgba(6, 11, 33, 0.4) !important; backdrop-filter: blur(15px); border-right: 1px solid rgba(255, 255, 255, 0.1); }
.stSlider div, .stSelectbox div, .stNumberInput div { color: white !important; }
div[data-baseweb="select"] > div, input { background-color: rgba(255, 255, 255, 0.05) !important; border: 1px solid rgba(255, 255, 255, 0.1) !important; color: white !important; }
[data-testid="stMetricValue"] { text-shadow: 0px 0px 10px rgba(0, 210, 255, 0.5); }
.stButton > button { background: rgba(255, 255, 255, 0.1) !important; border: 1px solid rgba(255, 255, 255, 0.2) !important; backdrop-filter: blur(10px); border-radius: 50px; transition: all 0.3s ease; }
.stButton > button:hover { background: rgba(0, 210, 255, 0.2) !important; border-color: rgba(0, 210, 255, 0.5) !important; box-shadow: 0px 0px 15px rgba(0, 210, 255, 0.3); }
</style>

<div class="cloud-canvas">
    <div class="cloud cloud-blue"></div>
    <div class="cloud cloud-cyan"></div>
    <div class="cloud cloud-purple"></div>
    <div class="cloud cloud-pink"></div>
    <div class="cloud cloud-green"></div>
</div>
"""
st.markdown(page_bg_css, unsafe_allow_html=True)


# --- HELPER FUNCTION: 3D NETWORK VISUALIZER ---
def draw_3d_network_plotly(architecture):
    fig = go.Figure()
    node_x, node_y, node_z, node_colors, hover_texts = [], [], [], [], []
    for i, layer_size in enumerate(architecture):
        x = i * 2 
        for j in range(layer_size):
            angle = (2 * np.pi * j) / layer_size if layer_size > 1 else 0
            radius = 1 if layer_size > 1 else 0
            node_x.append(x)
            node_y.append(radius * np.cos(angle))
            node_z.append(radius * np.sin(angle))
            if i == 0:
                node_colors.append('#2196F3'); hover_texts.append(f"Input Feature {j+1}")
            elif i == len(architecture) - 1:
                node_colors.append('#4CAF50'); hover_texts.append(f"Final Prediction")
            else:
                node_colors.append('#FF9800'); hover_texts.append(f"Layer {i} | Neuron {j+1}")

    fig.add_trace(go.Scatter3d(
        x=node_x, y=node_y, z=node_z, mode='markers',
        marker=dict(size=16, color=node_colors, line=dict(width=3, color='rgba(255, 255, 255, 0.4)')), 
        text=hover_texts, hoverinfo='text'
    ))

    edge_x, edge_y, edge_z = [], [], []
    current_node = 0
    for i in range(len(architecture) - 1):
        nodes_in_current, nodes_in_next = architecture[i], architecture[i+1]
        for j in range(nodes_in_current):
            for k in range(nodes_in_next):
                edge_x.extend([node_x[current_node + j], node_x[current_node + nodes_in_current + k], None])
                edge_y.extend([node_y[current_node + j], node_y[current_node + nodes_in_current + k], None])
                edge_z.extend([node_z[current_node + j], node_z[current_node + nodes_in_current + k], None])
        current_node += nodes_in_current

    fig.add_trace(go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines', line=dict(color='rgba(150, 150, 150, 0.2)', width=1), hoverinfo='none'))
    fig.update_layout(scene=dict(xaxis=dict(showbackground=False, showticklabels=False, title=''), yaxis=dict(showbackground=False, showticklabels=False, title=''), zaxis=dict(showbackground=False, showticklabels=False, title='')), margin=dict(l=0, r=0, b=0, t=0), showlegend=False, height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

# --- HELPER FUNCTION: SINGLE NEURON VISUALIZER ---
def draw_single_neuron_plotly(activation_name):
    fig = go.Figure()

    # The glowing Neuron Body
    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode='markers+text',
        marker=dict(size=140, color='rgba(0, 210, 255, 0.1)', line=dict(width=3, color='#00d2ff')),
        text=[f"<b>Σ = Wx + b</b><br><br>f(z) = {activation_name}"],
        textfont=dict(color='white', size=14),
        hoverinfo='none'
    ))

    # The Input, Bias, and Output Arrows (Annotations)
    annotations = [
        # Input 1
        dict(ax=-2, ay=1, x=-0.7, y=0.35, xref='x', yref='y', axref='x', ayref='y', text='x₁ (Feature)', showarrow=True, arrowhead=2, arrowsize=1.5, arrowcolor='white', font=dict(color='white', size=13)),
        dict(x=-1.5, y=0.9, text='w₁', showarrow=False, font=dict(color='#00d2ff', size=14)), # Weight 1 label
        
        # Input 2
        dict(ax=-2, ay=-1, x=-0.7, y=-0.35, xref='x', yref='y', axref='x', ayref='y', text='x₂ (Feature)', showarrow=True, arrowhead=2, arrowsize=1.5, arrowcolor='white', font=dict(color='white', size=13)),
        dict(x=-1.5, y=-0.9, text='w₂', showarrow=False, font=dict(color='#00d2ff', size=14)), # Weight 2 label
        
        # Bias
        dict(ax=0, ay=-1.5, x=0, y=-0.7, xref='x', yref='y', axref='x', ayref='y', text='b (Bias)', showarrow=True, arrowhead=2, arrowsize=1.5, arrowcolor='#ffeb3b', font=dict(color='#ffeb3b', size=13)),
        
        # Output
        dict(ax=0.7, ay=0, x=2, y=0, xref='x', yref='y', axref='x', ayref='y', text='ŷ (Output)', showarrow=True, arrowhead=2, arrowsize=1.5, arrowcolor='#4CAF50', font=dict(color='#4CAF50', size=16))
    ]

    fig.update_layout(
        annotations=annotations,
        xaxis=dict(range=[-2.5, 2.5], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[-2, 2], showgrid=False, zeroline=False, visible=False),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=10, b=10), height=300, showlegend=False
    )
    return fig

# --- STATE MACHINE (WIZARD ROUTING) ---
if 'step' not in st.session_state:
    st.session_state.step = 0

def next_step():
    st.session_state.step += 1
def prev_step():
    st.session_state.step -= 1

# ==========================================
# STEP 0: FRONT PAGE
# ==========================================
if st.session_state.step == 0:
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    _, center_col, _ = st.columns([1, 2, 1])
    
    with center_col:
        st.title("🧠 DIY Neural Networks")
        st.markdown("#### By Anubhav Sarkar")
        st.caption("B.Tech CSE | MBA-DSDA | GitHub: [@wattacodeanubhav](https://github.com/wattacodeanubhav)")
        st.divider()
        st.markdown("Welcome to the ultimate glass box. Ever wanted to see a neural network literally fold and bend three-dimensional space to its will? Now you can. Build your architecture, break it, and watch the matrix calculus unfold in real-time.")
        st.info("⚠️ **A friendly warning:** The limit of this engine is exactly as much as your PC can handle.")
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🚀 Let's Get Started", type="primary", use_container_width=True, on_click=next_step):
            pass

# ==========================================
# WIZARD: STEPS 1 TO 4
# ==========================================
else:
    # --- SIDEBAR CONTROLS ---
    st.sidebar.header("📁 3D Data Geometry")
    dataset_choice = st.sidebar.selectbox("Select a Dataset to solve:", ["3D Plane (Linear)", "3D Spherical (Nested)", "3D Double Helix (Spiral)"])
    file_map = {"3D Plane (Linear)": "3d_plane.csv", "3D Spherical (Nested)": "3d_spherical.csv", "3D Double Helix (Spiral)": "3d_helix.csv"}
    filepath = f"sample_data/{file_map[dataset_choice]}"

    st.sidebar.header("🎓 Interactive Presets")
    preset = st.sidebar.selectbox("Choose a learning scenario:", ["Manual Control", "The Exploding Gradient", "The Memorizer", "The Lazy Model"])
    
    if preset == "The Exploding Gradient": default_layers, default_neurons, default_lr, default_epochs, default_act, default_reg = 4, 16, 1.8, 800, "ReLU", "None"
    elif preset == "The Memorizer": default_layers, default_neurons, default_lr, default_epochs, default_act, default_reg = 3, 32, 0.4, 3000, "Tanh", "None"
    elif preset == "The Lazy Model": default_layers, default_neurons, default_lr, default_epochs, default_act, default_reg = 0, 1, 0.001, 100, "Sigmoid", "None"
    else: default_layers, default_neurons, default_lr, default_epochs, default_act, default_reg = 2, 8, 0.05, 1000, "ReLU", "None"

    st.sidebar.header("1. Network Architecture")
    num_layers = st.sidebar.number_input("Number of Hidden Layers", min_value=0, max_value=10, value=default_layers,help="Setting this to 0 turns the network into standard Linear/Logistic Regression!")
    hidden_layers = [st.sidebar.slider(f"Neurons in Layer {i+1}", min_value=1, max_value=64, value=default_neurons, help="More layers and neurons can capture complex patterns but may require more training time and risk overfitting.") for i in range(num_layers)]

    st.sidebar.header("2. Hyperparameters")
    learning_rate = st.sidebar.slider("Learning Rate", 0.001, 2.0, default_lr, 0.001, help="Step size taken during gradient descent.")
    epochs = st.sidebar.slider("Epochs", 100, 5000, default_epochs, 100, help="How many times the network reads the dataset.")

    st.sidebar.header("3. Network Behavior")
    problem_type = st.sidebar.selectbox("Problem Type", ["Classification", "Regression"], help="Classification predicts probabilities (Yes/No). Regression predicts continuous raw numbers.")
    activation_func = st.sidebar.selectbox("Activation Function", ["ReLU", "Sigmoid", "Tanh"], index=["ReLU", "Sigmoid", "Tanh"].index(default_act), help="Function applied to each neuron's output.")
    regularization = st.sidebar.selectbox("Regularization", ["None", "L1", "L2"], index=["None", "L1", "L2"].index(default_reg), help="Prevents overfitting by adding penalty to the loss function.")

   
    # --- GLOBAL DATA PREP ---
    try:
        pipeline = DataPipeline(filepath=filepath)
        clean_df = pipeline.load_data().dropna()
        X, Y = pipeline.prepare_matrices(target_column='label')
        
        # Standardize data for 3D geometric stability
        X_mean, X_std = np.mean(X, axis=1, keepdims=True), np.std(X, axis=1, keepdims=True)
        X_scaled = (X - X_mean) / (X_std + 1e-8)
        
        # Custom 70/15/15 Train/Val/Test Split specifically for the Glass Box
        m = X.shape[1]
        np.random.seed(42)
        indices = np.random.permutation(m)
        train_end, val_end = int(m * 0.7), int(m * 0.85)
        
        train_idx, val_idx, test_idx = indices[:train_end], indices[train_end:val_end], indices[val_end:]
        
        X_train, Y_train = X_scaled[:, train_idx], Y[:, train_idx]
        X_val, Y_val = X_scaled[:, val_idx], Y[:, val_idx]
        X_test, Y_test = X_scaled[:, test_idx], Y[:, test_idx]
        
        architecture = [X_train.shape[0]] + hidden_layers + [1]
        
        # Tag the raw dataframe so Plotly knows how to color the right-side plot
        split_labels = np.empty(m, dtype=object)
        split_labels[train_idx] = 'Train'
        split_labels[val_idx] = 'Validation'
        split_labels[test_idx] = 'Test'
        clean_df['Dataset Split'] = split_labels
        
        # UI Header & Progress
        st.title("🧠 DIY Neural Networks")
        st.progress(st.session_state.step / 4.0)

# --- STEP 1 UI ---
        if st.session_state.step == 1:
            st.header("Step 1: 3D Space & Data Splitting")
            
            col_plot1, col_plot2 = st.columns(2)
            
            # Left Plot: Raw Data Space (Cyan vs Yellow)
            with col_plot1:
                fig_data = go.Figure()
                fig_data.add_trace(go.Scatter3d(
                    x=clean_df['x1'], y=clean_df['x2'], z=clean_df['x3'], mode='markers',
                    marker=dict(size=9, color=clean_df['label'], colorscale=[[0, '#00d2ff'], [1, '#ffeb3b']], opacity=0.9, line=dict(width=2, color='rgba(255, 255, 255, 0.3)')),
                    projection=dict(x=dict(show=True, opacity=0.1), y=dict(show=True, opacity=0.1), z=dict(show=True, opacity=0.1))
                ))
                fig_data.update_layout(
                    title="Target Classes (What to learn)", 
                    template="plotly_dark", margin=dict(l=0, r=0, b=20, t=40), height=450, 
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                    scene=dict(xaxis=dict(showbackground=False, gridcolor='rgba(255,255,255,0.1)'), yaxis=dict(showbackground=False, gridcolor='rgba(255,255,255,0.1)'), zaxis=dict(showbackground=False, gridcolor='rgba(255,255,255,0.1)'))
                )
                st.plotly_chart(fig_data, use_container_width=True)
            
            # Right Plot: Data Split (White, Blue, Red)
            with col_plot2:
                fig_split = go.Figure()
                color_map = {'Train': '#ffffff', 'Validation': '#2196F3', 'Test': '#FF5252'}
                
                # Iterate through the dictionary to assign proper colors and legend names
                for split_name, hex_color in color_map.items():
                    subset = clean_df[clean_df['Dataset Split'] == split_name]
                    fig_split.add_trace(go.Scatter3d(
                        x=subset['x1'], y=subset['x2'], z=subset['x3'], mode='markers', name=split_name,
                        marker=dict(size=8, color=hex_color, opacity=0.9, line=dict(width=2, color='rgba(255, 255, 255, 0.3)')),
                        projection=dict(x=dict(show=True, opacity=0.1), y=dict(show=True, opacity=0.1), z=dict(show=True, opacity=0.1))
                    ))
                
                fig_split.update_layout(
                    title="Data Partition (70% / 15% / 15%)", 
                    template="plotly_dark", margin=dict(l=0, r=0, b=20, t=40), height=450, 
                    showlegend=True, legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.05, bgcolor="rgba(0,0,0,0.5)"),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                    scene=dict(xaxis=dict(showbackground=False, gridcolor='rgba(255,255,255,0.1)'), yaxis=dict(showbackground=False, gridcolor='rgba(255,255,255,0.1)'), zaxis=dict(showbackground=False, gridcolor='rgba(255,255,255,0.1)'))
                )
                st.plotly_chart(fig_split, use_container_width=True)

            st.markdown("""
                            ### 1. The Mathematical Space (Left Graph)
                            This shows your raw dataset before the neural network processes it.
                            * **The 3D Space:** The x, y, and z axes represent the input variables (features) of your data.
                            * **Target Classes:** The different colors (yellow, cyan, dark blue) are the categories your network needs to learn to identify and separate. 

                            ### 2. Data Partitioning (Right Graph)
                            If you show the network all the data at once, it might just memorize the coordinates instead of learning the underlying patterns. To prevent this "cheating," the data is randomly split into three groups:

                            * **Training Data (70% - White):** The *study material*. The network uses these points to learn patterns and adjust itself.
                            * **Validation Data (15% - Cyan):** The *practice test*. Used during training to check if the network is actually learning the rules or just blindly memorizing the training data.
                            * **Testing Data (15% - Red):** The *final exam*. This data is hidden until training is completely finished to prove the network can accurately classify brand-new, unseen data in the real world.
                            """)
                
            st.divider()
            st.button("Next ➡️", type="primary", on_click=next_step)

# --- STEP 2 UI ---
        elif st.session_state.step == 2:
            st.header("Step 2: Network Topology & Matrix Calculus")
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write("Rotate and zoom the network. Watch how the Input Layer now perfectly accepts 3 features (x1, x2, x3).")
                st.plotly_chart(draw_3d_network_plotly(architecture), use_container_width=True)
            with col2:
                st.write("### Matrix Shapes")
                for i in range(1, len(architecture)):
                    st.latex(rf"W^{{({i})}} \in \mathbb{{R}}^{{{architecture[i]} \times {architecture[i-1]}}}")
                    st.latex(rf"b^{{({i})}} \in \mathbb{{R}}^{{{architecture[i]} \times 1}}")
                    st.divider()
            
            st.markdown("""

This interactive graph shows the physical structure (or architecture) of your neural network.
* **Input Layer (Green):** Notice that it perfectly accepts 3 features ($x_1$, $x_2$, $x_3$). These correspond directly to the x, y, and z coordinates from the 3D data space in Step 1.
* **Hidden Layers (Orange):** Based on your settings, there are 2 hidden layers with 8 neurons each. These act as the "brain" of the network, transforming the raw inputs into complex patterns. 
* **Connections:** The web of lines represents the pathways data flows through. Every connection carries a "Weight" that adjusts the signal as it passes from one neuron to the next.

""")
            # Navigation
            nav_col1, nav_col2, _ = st.columns([1, 1, 4])
            with nav_col1: st.button("⬅️ Back", on_click=prev_step)
            with nav_col2: st.button("Next ➡️", type="primary", on_click=next_step)

# --- STEP 3 UI ---
        elif st.session_state.step == 3:
            st.header("Step 3: The Math Engine")
            st.markdown("A neural network is just a massive web of linear equations wrapped in non-linear wrappers. Let's look inside a single node.")
            
            # --- THE NEURON ANATOMY ---
            st.subheader("1. The Anatomy of a Neuron")
            st.plotly_chart(draw_single_neuron_plotly(activation_func), use_container_width=True, config={'displayModeBar': False})
            st.markdown("""
### The Two-Step Process of a Neuron

Inside every single neuron, data undergoes a rigid, two-step mathematical transformation before being passed to the next layer. Here is exactly what happens:

**1. The Dot Product ($\Sigma$)**
First, the neuron calculates a weighted sum of its inputs to assess their importance. 
* **The Mechanics:** It takes each incoming feature and multiplies it by a corresponding **weight** ($W$). These weights act as "importance filters." It then adds a **bias** ($b$) to shift the baseline activation threshold up or down.
* **The Math:** $\Sigma = Wx + b$
* **The Geometry:** Mathematically, this step is purely linear, creating a flat, rigid plane. If neural networks stopped at this step, the entire network would mathematically collapse into one giant, flat linear regression model.

**2. The Activation Function ($f(\Sigma)$)**
To solve complex, real-world problems, straight lines are not enough. We must pass the flat output from step one ($\Sigma$) through a non-linear wrapper called an activation function.
* **The Mechanics:** The activation function acts as a mathematical "gatekeeper." It assesses the raw number from the dot product and decides exactly what signal (if any) is significant enough to be fired forward to the next layer.
* **The Math:** $\t{Output} = f(\Sigma)$
* **The Geometry:** This is the secret sauce of neural networks. By applying this non-linear wrapper, the activation function "bends," "curves," or "clips" the flat plane from Step 1. When thousands of these bent planes are combined across multiple layers, the network can mold its decision boundaries around highly complex, multidimensional data shapes.
""")

            st.info("💡 Change the Activation Function in the sidebar to see how this neuron behaves!")
                
                # Dynamic explanations based on sidebar selection
            if activation_func == "ReLU":
                st.write("**ReLU (Rectified Linear Unit) - 'The Switch'**")
                st.latex(r"f(x) = \max(0, x)")
                st.caption("If the mathematical output is negative, ReLU forces it to 0 (turns the neuron off). If it's positive, it passes it through unchanged. It is computationally lightning fast and solves the vanishing gradient problem.")
            elif activation_func == "Sigmoid":
                st.write("**Sigmoid - 'The Probabilizer'**")
                st.latex(r"f(x) = \frac{1}{1 + e^{-x}}")
                st.caption("Squashes any number, no matter how large or small, into a smooth curve exactly between 0 and 1. Perfect for predicting binary probabilities, but struggles in deep networks because it causes gradients to vanish at the extremes.")
            elif activation_func == "Tanh":
                st.write("**Tanh (Hyperbolic Tangent) - 'The Zero-Centered S-Curve'**")
                st.latex(r"f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}")
                st.caption("Similar to Sigmoid, but it squashes values between -1 and 1. Because it is zero-centered, it often helps the network's optimization converge much faster than Sigmoid.")

            st.divider()

            # --- LOSS & REGULARIZATION ---
            st.subheader("2. Loss & Regularization")
            math_col1, math_col2 = st.columns(2)
            
            with math_col1:
                st.write("### The Cost Function")
                if problem_type == "Classification":
                    st.write("**Binary Cross-Entropy (BCE)**")
                    st.latex(r"\mathcal{L} = -\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i) \right]")
                    st.caption("BCE heavily penalizes the network if it is highly confident but mathematically wrong. The derivative passed backward to update the network simplifies elegantly to $\hat{y} - y$.(It provides a clean, mathematical slope that tells the network exactly how much to adjust its weights)")
                else:
                    st.write("**Mean Squared Error (MSE)**")
                    st.latex(r"\mathcal{L} = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2")
                    st.caption("MSE measures geometric variance. It squares the errors, meaning large mistakes are punished exponentially more than tiny mistakes.")

            with math_col2:
                st.write("### The Regularization Penalty")
                if regularization == "None": 
                    st.latex(r"\text{Penalty} = 0")
                    st.warning("No penalty. The network is free to mutate its weights to any extreme size, risking severe overfitting (memorizing the training data).")
                elif regularization == "L1": 
                    st.latex(r"\text{Penalty} = \frac{\lambda}{m} \sum |W|")
                    st.info("**L1 (Lasso): The Bouncer.** It penalizes the absolute value of the weights. This acts as feature selection, driving useless connection weights exactly to 0.")
                elif regularization == "L2": 
                    st.latex(r"\text{Penalty} = \frac{\lambda}{2m} \sum W^2")
                    st.info("**L2 (Ridge): The Micromanager.** It penalizes the squared value of the weights. It refuses to let any single weight dominate, spreading the 'learning' evenly across the network.")

            # Navigation
            st.divider()
            nav_col1, nav_col2, _ = st.columns([1, 1, 4])
            with nav_col1: st.button("⬅️ Back", on_click=prev_step)
            with nav_col2: st.button("Next ➡️", type="primary", on_click=next_step)


# --- STEP 4 UI ---
        elif st.session_state.step == 4:
            st.header("Step 4: Training & 3D Decision Boundary Audit")
            
            nav_col1, nav_col2, _ = st.columns([1, 1, 4])
            with nav_col1: st.button("⬅️ Back", on_click=prev_step)
            st.divider()

            if st.button("🚀 Execute Training", type="primary"):
                with st.spinner("Executing Gradient Descent & Mapping 3D Trajectory..."):
                    
                    nn = GlassBoxNetwork(layer_sizes=architecture, learning_rate=learning_rate, activation=activation_func, regularization=regularization, problem_type=problem_type)
                    
                    # --- ZERO-CHANGE TRAJECTORY LOOP ---
                    # We train 1 epoch at a time to record both Loss and Metrics dynamically
                    epochs_hist, loss_hist, metric_hist = [], [], []
                    record_interval = max(1, epochs // 100) # Record 100 snapshots to make a smooth 3D line
                    
                    for e in range(epochs):
                        current_loss = nn.train(X_train, Y_train, epochs=1)[0]
                        
                        if e % record_interval == 0 or e == epochs - 1:
                            epochs_hist.append(e)
                            loss_hist.append(current_loss)
                            
                            test_preds = nn.forward_propagation(X_test)
                            if problem_type == "Classification":
                                acc = np.mean((test_preds > 0.5).astype(int) == Y_test) * 100
                                metric_hist.append(acc)
                            else:
                                mae = np.mean(np.abs(test_preds - Y_test))
                                metric_hist.append(mae)
                    
                    t_col1, t_col2 = st.columns([1, 1])
                    with t_col1:
                        st.subheader("3D Learning Trajectory")
                        metric_name = "Accuracy %" if problem_type == "Classification" else "MAE"
                        st.write(f"""
This tracks the network's performance journey over time as it adjusts its internal weights.
* **The Axes:** It plots the training cycles (**Epochs**) against the network's **Loss** (its error rate) and **Accuracy** (the Z-Axis). 
* **The Path:** As the line travels across the graph, you are watching the network actively "learn" through Gradient Descent. A successful training run will show the trajectory climbing higher on the Z-axis (approaching 100% accuracy) while moving toward zero loss.""")
                        
                        fig_traj = go.Figure()
                        
                        # Draw the glowing 3D path of the network's learning journey
                        fig_traj.add_trace(go.Scatter3d(
                            x=epochs_hist, y=loss_hist, z=metric_hist,
                            mode='lines+markers',
                            marker=dict(size=4, color=epochs_hist, colorscale='Turbo', showscale=False),
                            line=dict(width=6, color=epochs_hist, colorscale='Turbo')
                        ))
                        
                        fig_traj.update_layout(
                            template="plotly_dark",
                            scene=dict(
                                xaxis_title="Epochs", yaxis_title="Loss", zaxis_title=metric_name,
                                xaxis=dict(showbackground=False, gridcolor='rgba(255,255,255,0.1)'),
                                yaxis=dict(showbackground=False, gridcolor='rgba(255,255,255,0.1)'),
                                zaxis=dict(showbackground=False, gridcolor='rgba(255,255,255,0.1)')
                            ),
                            margin=dict(l=0, r=0, b=0, t=0), height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig_traj, use_container_width=True)
                        
                        # Final Metrics Printout
                        m_col1, m_col2 = st.columns(2)
                        m_col1.metric("Final Loss", f"{loss_hist[-1]:.4f}")
                        m_col2.metric(f"Final {metric_name}", f"{metric_hist[-1]:.2f}")
                    
                    with t_col2:
                        st.subheader("3D Decision Boundary Render")
                        st.write("""
This is the visual proof of what your neural network actually figured out.
* **The Frosted Glass:** This translucent surface represents the exact geometric boundary where the network's prediction probability is exactly 50%. It is the dividing line (or "fence") between the different classes.
* **The Separation:** You can see how the network positioned this boundary to separate the yellow and blue data clusters. Because your current dataset is a simple linear plane, the network only needed to draw a perfectly flat piece of glass to separate them!
""")
                        
                        grid_x, grid_y, grid_z = np.mgrid[-3.5:3.5:35j, -3.5:3.5:35j, -3.5:3.5:35j]
                        grid_points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
                        grid_preds = nn.forward_propagation(grid_points).ravel()
                        
                        fig_boundary = go.Figure()
                        fig_boundary.add_trace(go.Isosurface(
                            x=grid_x.ravel(), y=grid_y.ravel(), z=grid_z.ravel(), value=grid_preds,
                            isomin=0.49, isomax=0.51, surface_count=1, colorscale=[[0, 'white'], [1, 'cyan']], opacity=0.25,
                            caps=dict(x_show=False, y_show=False, z_show=False), lighting=dict(specular=2.0, fresnel=1.0, roughness=0.2, ambient=0.5), name="Boundary Surface"
                        ))
                        
                        scaled_df_x1, scaled_df_x2, scaled_df_x3 = X_scaled[0, :], X_scaled[1, :], X_scaled[2, :]
                        fig_boundary.add_trace(go.Scatter3d(
                            x=scaled_df_x1, y=scaled_df_x2, z=scaled_df_x3, mode='markers',
                            marker=dict(size=8, color=Y.ravel(), colorscale=[[0, '#00d2ff'], [1, '#ffeb3b']], line=dict(width=2, color='rgba(255, 255, 255, 0.3)')), name="True Data"
                        ))
                        
                        fig_boundary.update_layout(template="plotly_dark", scene=dict(xaxis_title="x1", yaxis_title="x2", zaxis_title="x3", xaxis=dict(showbackground=False, gridcolor='rgba(255,255,255,0.1)'), yaxis=dict(showbackground=False, gridcolor='rgba(255,255,255,0.1)'), zaxis=dict(showbackground=False, gridcolor='rgba(255,255,255,0.1)')), margin=dict(l=0, r=0, b=0, t=0), height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
                        st.plotly_chart(fig_boundary, use_container_width=True)

    except FileNotFoundError:
        st.error("Dataset not found! Make sure you run `python3 generate_data.py` first.")
