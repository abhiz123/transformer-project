import urllib.request
import os
import tarfile
from torch.utils.data import Dataset, DataLoader
import torch

class TranslationDataset(Dataset):
    def __init__(self, src_file, tgt_file, src_vocab, tgt_vocab, max_len=50):
        self.src_lines = open(src_file, 'r', encoding='utf-8').readlines()[:1000]  # Reduced to 1000 sentences
        self.tgt_lines = open(tgt_file, 'r', encoding='utf-8').readlines()[:1000]  # Reduced to 1000 sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        src_tokens = [self.src_vocab.get(token, self.src_vocab['<unk>']) for token in self.src_lines[idx].strip().lower().split()][:self.max_len]
        tgt_tokens = [self.tgt_vocab.get(token, self.tgt_vocab['<unk>']) for token in self.tgt_lines[idx].strip().lower().split()][:self.max_len]
        
        return torch.tensor(src_tokens, dtype=torch.long), torch.tensor(tgt_tokens, dtype=torch.long)

def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)
    
    src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    
    return src_batch, tgt_batch

def download_and_extract_data():
    url = "https://www.statmt.org/europarl/v7/de-en.tgz"
    filename = "de-en.tgz"
    if not os.path.exists(filename):
        print(f"Downloading data from {url}")
        urllib.request.urlretrieve(url, filename)
    
    if not os.path.exists("data/europarl-v7.de-en.en"):
        print("Extracting data...")
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall(path="data")

def build_vocab(file_path, max_size=10000):
    vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            for token in line.strip().lower().split():
                if token not in vocab and len(vocab) < max_size:
                    vocab[token] = len(vocab)
    return vocab

def prepare_data(batch_size=32):
    download_and_extract_data()
    
    src_train = "data/europarl-v7.de-en.en"
    tgt_train = "data/europarl-v7.de-en.de"
    src_vocab = build_vocab(src_train)
    tgt_vocab = build_vocab(tgt_train)
    
    train_dataset = TranslationDataset(src_train, tgt_train, src_vocab, tgt_vocab)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # For simplicity, we're using the same data for validation and testing
    valid_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    return train_loader, valid_loader, test_loader, src_vocab, tgt_vocab