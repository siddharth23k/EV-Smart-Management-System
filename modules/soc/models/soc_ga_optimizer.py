
# GA hyperparameter optimization for SOC estimation.
# Fitness = validation RMSE (lower is better, so we negate it)


import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import TensorDataset

from modules.soc.models.lstm_cnn_attention_soc import (
    LSTMCNNAttentionSoC, train_soc_model, evaluate_soc_model
)


@dataclass
class SoCHyperParams:
    learning_rate:   float
    batch_size:      int
    lstm_hidden:     int
    num_lstm_layers: int
    dropout_rate:    float
    cnn_channels:    int

    def as_dict(self) -> Dict:
        return {
            "learning_rate":   self.learning_rate,
            "batch_size":      self.batch_size,
            "lstm_hidden":     self.lstm_hidden,
            "num_lstm_layers": self.num_lstm_layers,
            "dropout_rate":    self.dropout_rate,
            "cnn_channels":    self.cnn_channels,
        }


class SoCGAOptimizer:
    def __init__(
        self,
        X_train, y_train, X_val, y_val,
        population_size: int = 10,
        generations:     int = 5,
        mutation_rate:   float = 0.15,
        tournament_size: int = 3,
        max_epochs:      int = 10,
        device=None,
    ):
        self.X_train, self.y_train = X_train, y_train
        self.X_val,   self.y_val   = X_val,   y_val
        self.population_size  = population_size
        self.generations      = generations
        self.mutation_rate    = mutation_rate
        self.tournament_size  = min(tournament_size, population_size)
        self.max_epochs       = max_epochs
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._cache: Dict[Tuple, float] = {}

        # Search space
        self.lr_choices          = [1e-4, 5e-4, 1e-3, 5e-3]
        self.batch_choices       = [64, 128, 256]
        self.hidden_choices      = [64, 128, 256]
        self.layers_choices      = [1, 2, 3]
        self.dropout_choices     = [0.1, 0.2, 0.3, 0.4]
        self.cnn_channel_choices = [32, 64, 128]

    def _random_hp(self) -> SoCHyperParams:
        return SoCHyperParams(
            learning_rate=random.choice(self.lr_choices),
            batch_size=random.choice(self.batch_choices),
            lstm_hidden=random.choice(self.hidden_choices),
            num_lstm_layers=random.choice(self.layers_choices),
            dropout_rate=random.choice(self.dropout_choices),
            cnn_channels=random.choice(self.cnn_channel_choices),
        )

    def _encode(self, hp: SoCHyperParams) -> Tuple:
        return tuple(sorted(hp.as_dict().items()))

    def _fitness(self, hp: SoCHyperParams) -> float:
        key = self._encode(hp)
        if key in self._cache:
            return self._cache[key]

        print(f"  Evaluating: lr={hp.learning_rate}, hidden={hp.lstm_hidden}, "
              f"layers={hp.num_lstm_layers}, cnn={hp.cnn_channels}, batch={hp.batch_size}")

        model = LSTMCNNAttentionSoC(
            cnn_channels=hp.cnn_channels,
            lstm_hidden=hp.lstm_hidden,
            num_lstm_layers=hp.num_lstm_layers,
            dropout=hp.dropout_rate,
        )
        _, history = train_soc_model(
            model, self.X_train, self.y_train, self.X_val, self.y_val,
            lr=hp.learning_rate, batch_size=hp.batch_size,
            epochs=self.max_epochs, patience=3,
            device=self.device,
            save_path=f"modules/soc/models/tmp_ga_soc.pth",
        )
        best_rmse = min(history["val_rmse"])
        fitness   = -best_rmse          # GA maximizes, lower RMSE = higher fitness
        self._cache[key] = fitness
        return fitness

    def _tournament(self, pop, fits) -> SoCHyperParams:
        idx  = random.sample(range(len(pop)), self.tournament_size)
        best = max(idx, key=lambda i: fits[i])
        return pop[best]

    def _crossover(self, p1: SoCHyperParams, p2: SoCHyperParams):
        g1, g2 = list(p1.as_dict().values()), list(p2.as_dict().values())
        k  = random.randint(1, len(g1) - 1)
        c1 = g1[:k] + g2[k:]
        c2 = g2[:k] + g1[k:]
        def build(g):
            return SoCHyperParams(
                learning_rate=g[0], batch_size=int(g[1]),
                lstm_hidden=int(g[2]), num_lstm_layers=int(g[3]),
                dropout_rate=float(g[4]), cnn_channels=int(g[5]),
            )
        return build(c1), build(c2)

    def _mutate(self, hp: SoCHyperParams) -> SoCHyperParams:
        d = hp.as_dict()
        if random.random() < self.mutation_rate:
            d["learning_rate"]   = random.choice(self.lr_choices)
        if random.random() < self.mutation_rate:
            d["batch_size"]      = random.choice(self.batch_choices)
        if random.random() < self.mutation_rate:
            d["lstm_hidden"]     = random.choice(self.hidden_choices)
        if random.random() < self.mutation_rate:
            d["num_lstm_layers"] = random.choice(self.layers_choices)
        if random.random() < self.mutation_rate:
            d["dropout_rate"]    = random.choice(self.dropout_choices)
        if random.random() < self.mutation_rate:
            d["cnn_channels"]    = random.choice(self.cnn_channel_choices)
        return SoCHyperParams(**d)

    def run(self):
        population = [self._random_hp() for _ in range(self.population_size)]
        fitnesses  = [self._fitness(hp) for hp in population]

        best_idx  = int(np.argmax(fitnesses))
        best_hp   = population[best_idx]
        best_fit  = fitnesses[best_idx]
        fit_curve = [best_fit]

        for gen in range(1, self.generations + 1):
            new_pop = [best_hp]         # elitism
            while len(new_pop) < self.population_size:
                p1 = self._tournament(population, fitnesses)
                p2 = self._tournament(population, fitnesses)
                c1, c2 = self._crossover(p1, p2)
                new_pop.append(self._mutate(c1))
                if len(new_pop) < self.population_size:
                    new_pop.append(self._mutate(c2))

            population = new_pop
            fitnesses  = [self._fitness(hp) for hp in population]

            gen_best_idx = int(np.argmax(fitnesses))
            if fitnesses[gen_best_idx] > best_fit:
                best_fit = fitnesses[gen_best_idx]
                best_hp  = population[gen_best_idx]

            fit_curve.append(best_fit)
            print(f"Generation {gen}/{self.generations} — Best Val RMSE: {-best_fit:.4f}")

        return best_hp, -best_fit, fit_curve


def run_soc_ga():
    DATA = "modules/soc/data"
    X_train = np.load(f"{DATA}/X_train_soc.npy")
    y_train = np.load(f"{DATA}/y_train_soc.npy")
    X_val   = np.load(f"{DATA}/X_val_soc.npy")
    y_val   = np.load(f"{DATA}/y_val_soc.npy")

    print(f"Train: {X_train.shape}, Val: {X_val.shape}")

    ga = SoCGAOptimizer(
        X_train, y_train, X_val, y_val,
        population_size=6, generations=3,
        mutation_rate=0.15, max_epochs=5,
    )
    best_hp, best_rmse, fit_curve = ga.run()

    print(f"\nBest hyperparameters: {best_hp.as_dict()}")
    print(f"Best Val RMSE       : {best_rmse:.4f}")

    # Save best hyperparams
    os.makedirs("modules/soc/models", exist_ok=True)
    with open("modules/soc/models/best_soc_ga_hyperparams.json", "w") as f:
        json.dump({"best_val_rmse": best_rmse, "hyperparams": best_hp.as_dict()}, f, indent=4)
    print("Saved: modules/soc/models/best_soc_ga_hyperparams.json")

    # Plot fitness curve
    plt.figure(figsize=(6, 4))
    plt.plot(range(len(fit_curve)), [-f for f in fit_curve], marker="o")
    plt.xlabel("Generation")
    plt.ylabel("Best Val RMSE")
    plt.title("SOC GA Optimization — Fitness Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs("assets/img", exist_ok=True)
    plt.savefig("assets/img/soc_ga_fitness_curve.png", dpi=150)
    plt.close()
    print("Saved: assets/img/soc_ga_fitness_curve.png")


if __name__ == "__main__":
    run_soc_ga()