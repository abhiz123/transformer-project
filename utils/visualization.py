import matplotlib.pyplot as plt
import seaborn as sns

def plot_attention(attention, source_sentence, translated_sentence, file_name):
    plt.figure(figsize=(10,10))
    sns.heatmap(attention, xticklabels=source_sentence, yticklabels=translated_sentence, cmap='viridis')
    plt.xlabel('Source')
    plt.ylabel('Target')
    plt.title('Attention Weights Visualization')
    plt.savefig(file_name)
    plt.close()

def plot_loss(train_losses, val_losses, file_name):
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(file_name)
    plt.close()

def plot_bleu_comparison(transformer_bleu, baseline_bleu, file_name):
    models = ['Baseline', 'Transformer']
    bleu_scores = [baseline_bleu, transformer_bleu]
    
    plt.figure(figsize=(10,5))
    plt.bar(models, bleu_scores)
    plt.ylabel('BLEU Score')
    plt.title('BLEU Score Comparison')
    plt.savefig(file_name)
    plt.close()