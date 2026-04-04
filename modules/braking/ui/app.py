# Ensure project root is in sys.path for module imports
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
import io
import json


# Synthetic data generator
def generate_sequence(seq_len = 75, init_speed = 60, aggressiveness = 0.5, noise_level = 0.05):
    speed = init_speed
    brake = 0.0
    data = []

    for _ in range(seq_len):
        brake += (aggressiveness - brake) * np.random.uniform(0.03, 0.08)
        brake += np.random.normal(0, noise_level)
        brake = np.clip(brake, 0, 1)

        accel = -brake * np.random.uniform(2.0, 3.5)
        accel += np.random.normal(0, noise_level)

        speed += accel * 0.1
        speed = max(speed, 0)

        data.append([speed, accel, brake])

    return np.array(data)


# Different Scenario presets
SCENARIOS = {
    "Manual (use sliders)": None,
    "Highway Gentle Slowdown": (90, 0.18, 0.05),
    "City Traffic Braking": (50, 0.45, 0.08),
    "Sudden Emergency Stop": (80, 0.9, 0.04),
    "Stop-and-Go Traffic": (40, 0.6, 0.12),
}


# Load model
@st.cache_resource
def load_model():
    from models.multitask_lstm_cnn_attention import MultitaskLSTMCNNAttention
    model = MultitaskLSTMCNNAttention()
    model.load_state_dict(
        torch.load("models/final_multitask_model.pth", map_location = "cpu")
    )
    model.eval()
    return model


# Page config
st.set_page_config(page_title="Braking Intention Prediction", layout="wide")


# Title
st.markdown(
    """
    <h1 style="text-align:center;">Braking Intention Prediction</h1>
    <p style="text-align:center; color:#b0b0b0;">
    Predicts driver braking intention and brake intensity from vehicle time-series data.
    </p>
    """,
    unsafe_allow_html = True
)


# Top explainers
c1, c2, c3 = st.columns(3)

with c1:
    with st.expander("What does this app do?"):
        st.write(
            "Predicts **driver braking intention** (Light / Normal / Emergency) "
            "and **brake intensity** using a deep learning model."
        )

with c2:
    with st.expander("What data is used?"):
        st.write(
            "- Vehicle speed\n"
            "- Acceleration (deceleration)\n"
            "- Brake pedal input"
        )

with c3:
    with st.expander("How to interpret results?"):
        st.write(
            "- 🟢 Light: Low risk\n"
            "- 🟡 Normal: Moderate braking\n"
            "- 🔴 Emergency: Sudden / high-risk braking"
        )

st.divider()


# Scenario & Controls section
st.subheader("Scenario & Controls")

sc_col, sliders_col, run_col = st.columns([1.1, 2.1, 0.8])

with sc_col:
    st.markdown("### 🚦 Scenario Preset")
    scenario = st.selectbox(
        "Choose a predefined driving scenario:",
        list(SCENARIOS.keys())
    )

    if SCENARIOS[scenario]:
        init_speed, aggressiveness, noise_level = SCENARIOS[scenario]
    else:
        init_speed = aggressiveness = noise_level = None

with sliders_col:
    st.markdown("### 🎛️ Input Controls")

    init_speed = st.slider(
        "Initial Speed (km/h)", 20, 120,
        init_speed if init_speed is not None else 60
    )

    aggressiveness = st.slider(
        "Braking Aggressiveness", 0.1, 1.0,
        aggressiveness if aggressiveness is not None else 0.5
    )

    noise_level = st.slider(
        "Noise Level", 0.0, 0.2,
        noise_level if noise_level is not None else 0.05
    )

with run_col:
    st.markdown("<br><br>", unsafe_allow_html=True)
    run = st.button(
        "🚀 Run Prediction",
        use_container_width=True
    )

st.divider()


# Main layout
left, right = st.columns([1.6, 1])

if run:
    sequence = generate_sequence(
        init_speed = init_speed,
        aggressiveness = aggressiveness,
        noise_level = noise_level
    )

    # LEFT: Time-series plots
    with left:
        st.subheader("Input Time-Series Signals")

        fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

        axs[0].plot(sequence[:, 0], color="#1f77b4")
        axs[0].set_ylabel("Speed")

        axs[1].plot(sequence[:, 1], color="#ff7f0e")
        axs[1].set_ylabel("Acceleration")

        axs[2].plot(sequence[:, 2], color="#d62728")
        axs[2].set_ylabel("Brake Pedal")
        axs[2].set_xlabel("Time Step")

        for ax in axs:
            ax.grid(alpha=0.3)

        st.pyplot(fig)

    # RIGHT: Predictions
    with right:
        model = load_model()
        x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logits, intensity = model(x)
            probs = torch.softmax(logits, dim=1).numpy()[0]

        classes = ["Light Braking", "Normal Braking", "Emergency Braking"]
        colors = ["#2ecc71", "#f1c40f", "#e74c3c"]
        emojis = ["🟢", "🟡", "🔴"]

        pred = probs.argmax()

        st.subheader("Prediction Results")

        st.markdown(
            f"""
            <div style="
                background:{colors[pred]};
                padding : 20px;
                border-radius : 14px;
                text-align : center;
                font-size : 22px;
                font-weight : bold;">
                {emojis[pred]} {classes[pred]}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html = True)
        st.metric("Predicted Brake Intensity", f"{intensity.item():.2f}")

        # Pie chart
        fig2, ax2 = plt.subplots()
        wedges, _, _ = ax2.pie(
            probs,
            autopct = "%1.0f%%",
            startangle = 90,
            colors = colors
        )

        ax2.legend(
            wedges,
            classes,
            title="Braking Classes",
            loc="center left",
            bbox_to_anchor=(1, 0.5)
        )

        ax2.axis("equal")
        st.pyplot(fig2)


        # Interpretation dropdown
        with st.expander("🔍 Interpretation"):
            if pred == 0:
                st.write(
                    "Smooth speed reduction with low brake input. "
                    "Indicates controlled and low-risk braking."
                )
            elif pred == 1:
                st.write(
                    "Moderate deceleration with noticeable brake engagement. "
                    "Common in city or traffic conditions."
                )
            else:
                st.write(
                    "Rapid braking response with high intensity. "
                    "Represents a safety-critical situation."
                )

        # Download report
        report = {
            "scenario": scenario,
            "initial_speed": init_speed,
            "aggressiveness": aggressiveness,
            "noise_level": noise_level,
            "prediction": classes[pred],
            "brake_intensity": round(float(intensity.item()), 3),
            "class_probabilities": dict(zip(classes, probs.round(3).tolist()))
        }

        buffer = io.StringIO()
        json.dump(report, buffer, indent=2)

        st.download_button(
            "📥 Download Prediction Report",
            buffer.getvalue(),
            file_name = "braking_prediction_report.json",
            mime = "application/json"
        )