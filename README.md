# Transformer vs Baseline Seq2Seq Model Comparison

This project implements and compares the Transformer model (as described in "Attention Is All You Need") with a baseline Seq2Seq model for machine translation.

## Setup

1. Install the required packages: 'pip install -r requirements.txt'

2. Download the spaCy models: 
'python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm'

## Usage

1. Train the Transformer model: 'python train.py'

2. Train the Baseline model: 'python train_baseline.py'

3. Evaluate and compare both models: 'python evaluate.py'

## Project Structure

- `models/transformer.py`: Transformer model implementation
- `models/baseline.py`: Baseline Seq2Seq model implementation
- `utils/data_processing.py`: Data loading and preprocessing utilities
- `utils/visualization.py`: Functions for visualizing results
- `train.py`: Script for training the Transformer model
- `train_baseline.py`: Script for training the Baseline model
- `evaluate.py`: Script for evaluating and comparing both models

## Results

After running the scripts, you'll find:
- Trained models: `transformer_model.pth` and `baseline_model.pth`
- Loss plots: `transformer_loss_plot.png` and `baseline_loss_plot.png`
- BLEU score comparison: `bleu_comparison.png`
- Attention visualization: `attention_plot.png`

The evaluation script will print BLEU scores and sample translations for both models.

