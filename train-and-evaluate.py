import torch
import torch.nn as nn
import torch.optim as optim
from utils import prepare_data
from models.transformer import Transformer
from models.baseline import BaselineSeq2Seq
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_loader, optimizer, criterion, clip=1.0, scheduler=None):
    model.train()
    epoch_loss = 0
    for src, tgt in tqdm(train_loader, desc="Training"):
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        
        if isinstance(model, Transformer):
            output, _ = model(src, tgt[:, :-1])
        else:  # Baseline model
            output = model(src, tgt)
        
        loss = criterion(output.contiguous().view(-1, output.size(-1)), tgt[:, 1:].contiguous().view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        epoch_loss += loss.item()
    return epoch_loss / len(train_loader)

def evaluate(model, val_loader, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, tgt in tqdm(val_loader, desc="Evaluating"):
            src, tgt = src.to(device), tgt.to(device)
            
            if isinstance(model, Transformer):
                output, _ = model(src, tgt[:, :-1])
            else:  # Baseline model
                output = model(src, tgt)
            
            loss = criterion(output.contiguous().view(-1, output.size(-1)), tgt[:, 1:].contiguous().view(-1))
            epoch_loss += loss.item()
    return epoch_loss / len(val_loader)

def translate_sentence(model, sentence, src_vocab, tgt_vocab, max_length=50):
    model.eval()
    tokens = sentence.lower().split()
    src_indexes = [src_vocab.get(token, src_vocab['<unk>']) for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    
    with torch.no_grad():
        if isinstance(model, Transformer):
            memory, src_mask = model.encode(src_tensor)
            ys = torch.ones(1, 1).fill_(tgt_vocab['<bos>']).type(torch.long).to(device)
            for i in range(max_length-1):
                out, _ = model.decode(ys, memory, src_mask)
                prob = model.fc_out(out[:, -1])
                _, next_word = torch.max(prob, dim=1)
                ys = torch.cat([ys, next_word.unsqueeze(0)], dim=1)
                if next_word.item() == tgt_vocab['<eos>']:
                    break
        else:  # Baseline model
            hidden, cell = model.encode(src_tensor)
            ys = torch.ones(1, 1).fill_(tgt_vocab['<bos>']).type(torch.long).to(device)
            for i in range(max_length-1):
                output, hidden, cell = model.decode_step(ys[:, -1:], hidden, cell)
                _, next_word = torch.max(output.squeeze(1), dim=1)
                ys = torch.cat([ys, next_word.unsqueeze(0)], dim=1)
                if next_word.item() == tgt_vocab['<eos>']:
                    break
    
    ys = ys.squeeze().tolist()
    return [list(tgt_vocab.keys())[list(tgt_vocab.values()).index(i)] for i in ys]

def plot_attention(attention, source, translation, filename):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    cax = ax.matshow(attention, cmap='bone')
    fig.colorbar(cax)
    
    ax.set_xticklabels([''] + source, rotation=90)
    ax.set_yticklabels([''] + translation)
    
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    
    plt.savefig(filename)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate Transformer and Baseline models')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--test', action='store_true', help='run in test mode with fewer epochs')
    args = parser.parse_args()

    # Hyperparameters
    batch_size = args.batch_size
    src_vocab_size = 10000
    tgt_vocab_size = 10000
    d_model = 256  
    num_heads = 8
    num_layers = 3
    d_ff = 512
    max_seq_length = 100
    dropout = 0.1
    hidden_size = 256
    num_epochs = 2 if args.test else args.epochs
    learning_rate = args.lr

    # Prepare data
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = prepare_data(batch_size)

    # Initialize models
    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout).to(device)
    baseline = BaselineSeq2Seq(src_vocab_size, tgt_vocab_size, hidden_size).to(device)

    # Define loss function and optimizers
    criterion = nn.CrossEntropyLoss(ignore_index=src_vocab['<pad>'])
    transformer_optimizer = optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    baseline_optimizer = optim.Adam(baseline.parameters(), lr=1e-3)

    # Learning rate scheduler for Transformer
    def lr_lambda(step):
        warmup_steps = 4000
        return d_model**(-0.5) * min((step+1)**(-0.5), (step+1) * warmup_steps**(-1.5))
    
    scheduler = optim.lr_scheduler.LambdaLR(transformer_optimizer, lr_lambda)

    # Training loop
    transformer_losses = []
    baseline_losses = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train and evaluate Transformer
        transformer_train_loss = train(transformer, train_loader, transformer_optimizer, criterion, clip=1.0, scheduler=scheduler)
        transformer_val_loss = evaluate(transformer, val_loader, criterion)
        transformer_losses.append((transformer_train_loss, transformer_val_loss))
        
        # Train and evaluate Baseline
        baseline_train_loss = train(baseline, train_loader, baseline_optimizer, criterion, clip=1.0)
        baseline_val_loss = evaluate(baseline, val_loader, criterion)
        baseline_losses.append((baseline_train_loss, baseline_val_loss))
        
        print(f"Transformer - Train Loss: {transformer_train_loss:.4f}, Val Loss: {transformer_val_loss:.4f}")
        print(f"Baseline - Train Loss: {baseline_train_loss:.4f}, Val Loss: {baseline_val_loss:.4f}")

    # Save models
    torch.save(transformer.state_dict(), 'transformer_model.pth')
    torch.save(baseline.state_dict(), 'baseline_model.pth')

    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot([l[0] for l in transformer_losses], label='Transformer Train')
    plt.plot([l[1] for l in transformer_losses], label='Transformer Val')
    plt.plot([l[0] for l in baseline_losses], label='Baseline Train')
    plt.plot([l[1] for l in baseline_losses], label='Baseline Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_curves.png')
    plt.close()

    # Test translation
    test_sentence = "The cat sits on the mat ."
    transformer_translation = translate_sentence(transformer, test_sentence, src_vocab, tgt_vocab)
    baseline_translation = translate_sentence(baseline, test_sentence, src_vocab, tgt_vocab)
    
    print(f"Source: {test_sentence}")
    print(f"Transformer: {' '.join(transformer_translation)}")
    print(f"Baseline: {' '.join(baseline_translation)}")

   # Visualize attention (for Transformer only)
    src_tokens = test_sentence.split()
    src_tensor = torch.LongTensor([src_vocab.get(token, src_vocab['<unk>']) for token in src_tokens]).unsqueeze(0).to(device)
    
    with torch.no_grad():
        memory, src_mask = transformer.encode(src_tensor)
        ys = torch.ones(1, 1).fill_(tgt_vocab['<bos>']).type(torch.long).to(device)
        for i in range(50):
            out, attentions = transformer.decode(ys, memory, src_mask)
            prob = transformer.fc_out(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            ys = torch.cat([ys, next_word.unsqueeze(0)], dim=1)
            if next_word.item() == tgt_vocab['<eos>']:
                break
    
    tgt_tokens = [list(tgt_vocab.keys())[list(tgt_vocab.values()).index(i)] for i in ys.squeeze().tolist()]
    attention = attentions[-1].squeeze(0).mean(0).cpu().detach().numpy()
    plot_attention(attention, src_tokens, tgt_tokens, 'attention_visualization.png')

if __name__ == "__main__":
    main()