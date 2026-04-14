set -e

echo "🚀 Starting full training pipeline..."
echo "Venv: $(python -c 'import sys; print(sys.executable)')"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"

cd "$(dirname "$0")"

echo "1/3 Training Braking models..."
python train_braking.py --mode full

echo "2/3 Training SoC models..."
python train_soc.py --mode full

echo "3/3 Running GA optimization..."
PYTHONPATH=.. python ../braking/models/genetic_algorithm_optimizer.py
PYTHONPATH=.. python ../soc/models/soc_ga_optimizer.py

echo "ALL MODELS TRAINED!"
echo "Run: cd ../../ && python run_unified.py"
