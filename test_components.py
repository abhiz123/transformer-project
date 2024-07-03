import torch
from models.transformer import Transformer
from models.baseline import BaselineSeq2Seq
from utils import prepare_data
from train_and_evaluate import translate_sentence, plot_attention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_data_loading():
    train_loader, _, _, src_vocab, tgt_vocab = prepare_data(batch_size=2)
    src, tgt = next(iter(train_loader))
    print(f"Sample batch - Source shape: {src.shape}, Target shape: {tgt.shape}")
    print(f"Vocabulary sizes - Source: {len(src_vocab)}, Target: {len(tgt_vocab)}")

def test_model_forward_pass():
    src_vocab_size, tgt_vocab_size = 10000, 10000
    d_model, num_heads, num_layers, d_ff, max_seq_length, dropout = 256, 8, 3, 512, 100, 0.1
    hidden_size = 256

    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout).to(device)
    baseline = BaselineSeq2Seq(src_vocab_size, tgt_vocab_size, hidden_size).to(device)

    src = torch.randint(0, src_vocab_size, (2, 10)).to(device)
    tgt = torch.randint(0, tgt_vocab_size, (2, 8)).to(device)

    with torch.no_grad():
        transformer_out = transformer(src, tgt[:, :-1])
        baseline_out = baseline(src, tgt[:, :-1])

    print(f"Transformer output shape: {transformer_out.shape}")
    print(f"Baseline output shape: {baseline_out.shape}")

def test_translation():
    _, _, _, src_vocab, tgt_vocab = prepare_data(batch_size=1)
    src_vocab_size, tgt_vocab_size = 10000, 10000
    d_model, num_heads, num_layers, d_ff, max_seq_length, dropout = 256, 8, 3, 512, 100, 0.1
    hidden_size = 256

    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout).to(device)
    baseline = BaselineSeq2Seq(src_vocab_size, tgt_vocab_size, hidden_size).to(device)

    # Load pre-trained models if available
    try:
        transformer.load_state_dict(torch.load('transformer_model.pth'))
        baseline.load_state_dict(torch.load('baseline_model.pth'))
        print("Loaded pre-trained models.")
    except FileNotFoundError:
        print("Pre-trained models not found. Using untrained models.")

    test_sentence = "The cat sits on the mat ."
    transformer_translation = translate_sentence(transformer, test_sentence, src_vocab, tgt_vocab)
    baseline_translation = translate_sentence(baseline, test_sentence, src_vocab, tgt_vocab)
    
    print(f"Source: {test_sentence}")
    print(f"Transformer translation: {' '.join(transformer_translation)}")
    print(f"Baseline translation: {' '.join(baseline_translation)}")

    # Test attention visualization
    src_tokens = test_sentence.split()
    tgt_tokens = transformer_translation
    attention = transformer.decoder_layers[-1].cross_attention.attention_weights.squeeze(0).cpu().detach().numpy()
    plot_attention(attention, src_tokens, tgt_tokens, 'test_attention_visualization.png')
    print("Attention visualization saved as 'test_attention_visualization.png'")

if __name__ == "__main__":
    print("Testing data loading...")
    test_data_loading()
    print("\nTesting model forward pass...")
    test_model_forward_pass()
    print("\nTesting translation...")
    test_translation()