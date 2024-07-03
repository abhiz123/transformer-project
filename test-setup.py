import torch
from utils import prepare_data
from models.transformer import Transformer
from models.baseline import BaselineSeq2Seq

def test_data_loading():
    train_loader, valid_loader, test_loader, src_vocab, tgt_vocab = prepare_data(batch_size=2)
    print(f"Vocabulary sizes: Source: {len(src_vocab)}, Target: {len(tgt_vocab)}")
    
    # Test multiple batches
    for i, (src, tgt) in enumerate(train_loader):
        print(f"Batch {i + 1}:")
        print(f"  Source shape: {src.shape}")
        print(f"  Target shape: {tgt.shape}")
        if i == 2:  # Stop after 3 batches
            break

def test_models():
    # Small model for testing
    src_vocab_size, tgt_vocab_size = 10000, 10000  # Updated to match our actual vocab size
    d_model = 32
    num_heads = 2
    num_layers = 2
    d_ff = 64
    max_seq_length = 50
    dropout = 0.1
    hidden_size = 32

    # Test Transformer
    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
    print(f"Transformer parameters: {sum(p.numel() for p in transformer.parameters())}")

    # Test Baseline
    baseline = BaselineSeq2Seq(src_vocab_size, tgt_vocab_size, hidden_size)
    print(f"Baseline parameters: {sum(p.numel() for p in baseline.parameters())}")

    # Test forward pass
    src = torch.randint(0, src_vocab_size, (2, 10))  # (batch_size, seq_len)
    tgt = torch.randint(0, tgt_vocab_size, (2, 8))   # (batch_size, seq_len)
    
    with torch.no_grad():
        transformer_out = transformer(src, tgt[:, :-1])
        baseline_out = baseline(src, tgt[:, :-1])
    
    print(f"Transformer output shape: {transformer_out.shape}")
    print(f"Baseline output shape: {baseline_out.shape}")

if __name__ == "__main__":
    print("Testing data loading and preprocessing...")
    test_data_loading()
    print("\nTesting model instantiation and forward pass...")
    test_models()