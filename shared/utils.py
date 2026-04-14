"""
shared/utils.py
Unified EV Smart Management System pipeline.
Connects braking intention module + SOC estimation module.
"""

import os
import json
import numpy as np
import torch


class UnifiedEVPipeline:
    """
    Unified inference pipeline connecting both modules:
    1. Braking Module  → predicts braking class + intensity
    2. SOC Module      → estimates battery state of charge
    3. Integration     → uses braking intensity to compute energy recovered
                         and update SOC prediction
    """

    # Regenerative braking efficiency (typical EV: 60-70%)
    REGEN_EFFICIENCY = 0.65
    # Battery capacity in kWh (typical EV: 40-100 kWh, we use normalised)
    BATTERY_CAPACITY = 1.0
    # Class labels
    CLASS_LABELS = {0: "Light Braking", 1: "Normal Braking", 2: "Emergency Braking"}

    def __init__(
        self,
        braking_model_path: str = "modules/braking/models/final_multitask_model.pth",
        soc_model_path:     str = "modules/soc/models/lstm_cnn_attention_soc.pth",
        braking_hp_path:    str = "modules/braking/models/best_ga_hyperparams.json",
        device: str = None,
    ):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.braking_model = None
        self.soc_model     = None
        self._load_models(braking_model_path, soc_model_path, braking_hp_path)

    def _load_models(self, braking_path, soc_path, hp_path):
        # ── Braking model
        try:
            from modules.braking.models.multitask_lstm_cnn_attention import (
                MultitaskLSTMCNNAttention,
            )
            # Load best GA hyperparams if available
            if os.path.exists(hp_path):
                with open(hp_path) as f:
                    hp = json.load(f)["hyperparams"]
                # Use parameters that match the saved model architecture
                self.braking_model = MultitaskLSTMCNNAttention(
                    input_dim=3,
                    cnn_channels=32,  # Match saved model
                    lstm_hidden=hp.get("lstm_hidden_size", 64),
                    num_lstm_layers=1,  # Match saved model
                    dropout_rate=0.0,  # Match saved model
                )
            else:
                self.braking_model = MultitaskLSTMCNNAttention(
                    input_dim=3,
                    cnn_channels=32,
                    lstm_hidden=64,
                    num_lstm_layers=1,
                    dropout_rate=0.0,
                )

            if os.path.exists(braking_path):
                self.braking_model.load_state_dict(
                    torch.load(braking_path, map_location=self.device)
                )
                self.braking_model.to(self.device).eval()
                print(f"Braking model loaded from {braking_path}")
            else:
                print(f"Braking model weights not found at {braking_path}")
                self.braking_model = None
        except Exception as e:
            print(f"Could not load braking model: {e}")
            self.braking_model = None

        # SOC model

        try:
            from modules.soc.models.lstm_cnn_attention_soc import LSTMCNNAttentionSoC
            self.soc_model = LSTMCNNAttentionSoC()
            if os.path.exists(soc_path):
                self.soc_model.load_state_dict(
                    torch.load(soc_path, map_location=self.device)
                )
                self.soc_model.to(self.device).eval()
                print(f"SOC model loaded from {soc_path}")
            else:
                print(f"SOC model weights not found at {soc_path}")
                self.soc_model = None
        except Exception as e:
            print(f"Could not load SOC model: {e}")
            self.soc_model = None

    def predict_braking(self, driving_window: np.ndarray) -> dict:
        """
        Args:
            driving_window: (75, 3) — speed, acceleration, brake pedal
        Returns:
            dict with class_label, class_id, intensity
        """
        if self.braking_model is None:
            return {"class_label": "Unknown", "class_id": -1, "intensity": 0.0}

        x = torch.tensor(driving_window, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, intensity = self.braking_model(x)
        class_id    = int(torch.argmax(logits, dim=1).item())
        intensity   = float(intensity.item())
        return {
            "class_id":    class_id,
            "class_label": self.CLASS_LABELS[class_id],
            "intensity":   round(intensity, 4),
        }

    def predict_soc(self, battery_window: np.ndarray) -> float:
        """
        Args:
            battery_window: (50, 3) — voltage, current, temperature
        Returns:
            SOC value in [0, 1]
        """
        if self.soc_model is None:
            return -1.0

        x = torch.tensor(battery_window, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            soc = self.soc_model(x).item()
        return round(float(soc), 4)

    def compute_energy_recovered(self, brake_intensity: float) -> float:
        """
        Estimates energy recovered via regenerative braking.
        energy_recovered = intensity × regen_efficiency × capacity (normalised)
        """
        return round(brake_intensity * self.REGEN_EFFICIENCY * self.BATTERY_CAPACITY, 6)

    def run(self, driving_window: np.ndarray, battery_window: np.ndarray,
            current_soc: float = None) -> dict:
        """
        Full unified inference.

        Args:
            driving_window : (75, 3) — speed, accel, brake pedal
            battery_window : (50, 3) — voltage, current, temperature
            current_soc    : optional current SOC for delta computation

        Returns:
            Unified result dict
        """
        # Step 1: Braking prediction
        braking = self.predict_braking(driving_window)

        # Step 2: Energy recovered from regenerative braking
        energy_recovered = self.compute_energy_recovered(braking["intensity"])

        # Step 3: SOC estimation from battery signals
        soc_estimated = self.predict_soc(battery_window)

        # Step 4: SOC update using recovered energy
        if current_soc is not None and soc_estimated >= 0:
            soc_updated = min(1.0, current_soc + energy_recovered)
        else:
            soc_updated = soc_estimated

        result = {
            "braking": {
                "class":     braking["class_label"],
                "class_id":  braking["class_id"],
                "intensity": braking["intensity"],
            },
            "energy": {
                "recovered_normalised": energy_recovered,
                "regen_efficiency":     self.REGEN_EFFICIENCY,
            },
            "soc": {
                "estimated":    soc_estimated,
                "updated":      round(soc_updated, 4) if current_soc is not None else soc_estimated,
                "delta":        round(energy_recovered, 6),
            },
            "system_action": _determine_action(braking["class_id"], soc_estimated),
        }
        return result


def _determine_action(class_id: int, soc: float) -> str:
    """
    Simple rule-based system action based on braking class and SOC.
    This is the integration layer — braking intent informs EV controller.
    """
    if class_id == 2:   # Emergency
        return "ADAS_ALERT: Trigger emergency braking assist + maximum regen"
    elif class_id == 1: # Normal
        if soc < 0.2:
            return "REGEN: Prioritise energy recovery — battery low"
        return "REGEN: Standard regenerative braking active"
    else:               # Light
        return "REGEN: Light regenerative braking — comfort mode"