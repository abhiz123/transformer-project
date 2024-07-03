import torch
from models.transformer import Transformer
from models.baseline import BaselineSeq2Seq
from utils import prepare_data
from train_and_evaluate import translate_sentence, plot_attention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_models():
    _, _, _, src_vocab, tgt_vocab = prepare_data(batch_size=1)  # We only need vocabs here

    # Load model configurations (ensure these match your training configuration)
    src_vocab_size, tgt_vocab_size = 10000, 10000
    d_model, num_heads, num_layers, d_ff, max_seq_length, dropout = 256, 8, 3, 512, 100, 0.1
    hidden_size = 256

    # Initialize and load models
    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout).to(device)
    baseline = BaselineSeq2Seq(src_vocab_size, tgt_vocab_size, hidden_size).to(device)

    transformer.load_state_dict(torch.load('transformer_model.pth'))
    baseline.load_state_dict(torch.load('baseline_model.pth'))

    # Test translation
    test_sentence = "The cat sits on the mat ."
    transformer_translation = translate_sentence(transformer, test_sentence, src_vocab, tgt_vocab)
    baseline_translation = translate_sentence(baseline, test_sentence, src_vocab, tgt_vocab)
    
    print(f"Source: {test_sentence}")
    print(f"Transformer: {' '.join(transformer_translation)}")
    print(f"Baseline: {' '.join(baseline_translation)}")

    # Visualize attention (for Transformer only)
    src_tokens = test_sentence.split()
    tgt_tokens = transformer_translation
    attention = transformer.decoder_layers[-1].cross_attention.attention_weights.squeeze(0).cpu().detach().numpy()
    plot_attention(attention, src_tokens, tgt_tokens, 'attention_visualization.png')

if __name__ == "__main__":
    evaluate_models()