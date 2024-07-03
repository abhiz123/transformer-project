import torch
import torch.nn as nn
import torch.optim as optim
from models.transformer import Transformer
from utils.data_processing import prepare_data
from utils.visualization import plot_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        src, tgt = batch
        src, tgt = src.to(device), tgt.to(device)
        
        optimizer.zero_grad()
        output = model(src, tgt[:, :-1])
        loss = criterion(output.contiguous().view(-1, output.size(-1)), tgt[:, 1:].contiguous().view(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            src, tgt = batch
            src, tgt = src.to(device), tgt.to(device)
            
            output = model(src, tgt[:, :-1])
            loss = criterion(output.contiguous().view(-1, output.size(-1)), tgt[:, 1:].contiguous().view(-1))
            
            total_loss += loss.item()
    return total_loss / len(val_loader)

def main():
    train_loader, valid_loader, _, en_vocab, de_vocab = prepare_data()

    src_vocab_size = len(en_vocab)
    tgt_vocab_size = len(de_vocab)
    d_model = 256
    num_heads = 4
    num_layers = 3
    d_ff = 512
    max_seq_length = 100
    dropout = 0.1

    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    num_epochs = 10
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, valid_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch: {epoch+1}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")

    torch.save(model.state_dict(), "transformer_model.pth")
    plot_loss(train_losses, val_losses, "loss_plot.png")

if __name__ == "__main__":
    main()