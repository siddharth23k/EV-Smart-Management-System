# Driver Braking Intention Recognition for EVs

## Overview
This project builds a deep learning system that predicts how a driver is braking using vehicle time-series signals such as:

- Vehicle speed
- Acceleration (deceleration)
- Brake pedal input

The model predicts:

- Braking intention — Light / Normal / Emergency
- Brake intensity — a continuous value representing braking aggressiveness

Why this matters?
- Early braking prediction can improve vehicle safety systems and driver assistance technologies.
- If a car detects emergency braking earlier, safety systems can react faster.
- For electric vehicles, braking intensity prediction can also help optimize regenerative braking, improving energy recovery and efficiency.

---

## Demo

![Braking Intention Prediction Demo](assets/braking_intention_demo.gif)

---

## Model Architecture

![Braking Intention Example](assets/img/3.png)

---

## Quantitative Results

| Model | Accuracy | Macro F1 | Normal Braking F1 | Emergency Braking F1 |
|------|----------|----------|-------------------|----------------------|
| Baseline (Single-task) | 69.6% | 70% | ~59% | ~78% |
| AE + Classifier (Best) | 64.1% | 64% | ~56% | ~77% |
| Multitask (λ = 0.5) | 69.0% | 70% | ~57% | ~77% |
| **Multitask (λ = 0.8)** | **71.3%** | **72%** | **59%** | **82%** |
| Multitask + GA (local test) | ~71% | ~72% | - | - |

Key result:
Multitask learning improved overall performance and achieved 82% F1-score for Emergency Braking, the most safety-critical class. A full GA optimization run on GPU is planned to further tune hyperparameters and update these results.

---

## Genetic Algorithm Hyperparameter Optimization

This project implements the Genetic Algorithm (GA) hyperparameter optimization proposed in the original paper, applied to the Multitask LSTM+CNN+Attention model.

### How it works

- **Chromosome**: each individual encodes 6 hyperparameters — `learning_rate`, `batch_size`, `lstm_hidden_size`, `num_lstm_layers`, `dropout_rate`, `cnn_filters`
- **Fitness function**: validation macro F1 score from the multitask classification head
- **Selection**: tournament selection
- **Crossover**: single-point crossover
- **Mutation**: per-gene random mutation with configurable rate
- **Elitism**: best individual always carried forward to next generation

### GA Fitness Curve

![GA Fitness Curve](assets/img/ga_fitness_curve.png)

### Search Space

| Hyperparameter | Search Space |
|---|---|
| learning_rate | log-uniform [1e-4, 1e-2] |
| batch_size | 16, 32, 64, 128 |
| lstm_hidden_size | 64, 128, 256 |
| num_lstm_layers | 1, 2, 3 |
| dropout_rate | uniform [0.1, 0.5] |
| cnn_filters | 32, 64, 128 |

Best hyperparameters found are saved to `models/best_ga_hyperparams.json`.

> **Note**: The local test run used a reduced population (6 individuals, 3 generations, 2 epochs) to validate the pipeline on CPU. A full run (population=20, generations=10) on GPU is needed for meaningful optimization results.

---

## Installation & Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/braking-intention-prediction.git
cd braking-intention-prediction
pip install -r requirements.txt
```

(Optional) If using Jupyter notebooks:
```bash
python -m ipykernel install --user --name braking-intent
```

---

## How to Run

### 1. Generate Dataset
```bash
python data/generate_dataset.py
python data/generate_hard_braking_data.py
python data/generate_hard_braking_data_mtl.py
```
This will create `.npy` files containing time-series samples and labels.

### 2. Train Baseline Model

Open and run:
```
01_train_baseline.ipynb
```

### 3. Train Final Multitask Model

Open and run:
```
02_multitask_training.ipynb
```
All results, confusion matrices, and metrics are produced inside the notebooks.

### 4. Run Genetic Algorithm Optimizer
```bash
PYTHONPATH=. python models/genetic_algorithm_optimizer.py
```
This outputs:
- `models/best_ga_hyperparams.json` — best hyperparameter configuration found
- `assets/img/ga_fitness_curve.png` — fitness across generations

### 5. Run the Interactive Demo
```bash
streamlit run ui/app.py
```

---

## Assumptions & Design Choices

### Key assumptions

- Vehicle signals are synthetically generated to simulate realistic braking scenarios.
- Short time windows of speed, acceleration, and brake input are sufficient to estimate braking intention.
- Brake intensity correlates with braking aggressiveness.

### Limitations
- Model is trained on synthetic data; real-world deployment would require:
  - Sensor calibration
  - Domain adaptation
  - Validation on real driving datasets
- Reaction latency, road conditions, and driver intent beyond braking are not modeled.
- GA optimization results shown are from a reduced local run; a full GPU run is needed for meaningful hyperparameter search.

---

## References

This project reproduces and extends ideas from the following research work:

Wei Yang, Yu Huang, Kongming Jiang, Zhen Zhang, Ketong Zong, Qin Ruan,  
**"Method of Predicting Braking Intention Using LSTM-CNN-Attention With Hyperparameters Optimized by Genetic Algorithm"**,  
*International Journal of Control, Automation and Systems*, Springer, 2024.  
(https://link.springer.com/article/10.1007/s12555-021-1113-x)

The original paper proposes an LSTM–CNN–Attention architecture for braking intention prediction using simulator-based driving data.

This project reimplements the core architecture and introduces:
- Harder ambiguous synthetic datasets
- Systematic ablation studies
- Multitask learning with braking intensity regression
- **Genetic algorithm hyperparameter optimization** (as proposed in the original paper title)