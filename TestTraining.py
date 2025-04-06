# conda: cs4990env

import pretty_midi
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import pickle


##########################################
# Step 0: Data Pre-Processing
##########################################
def load_midi_files(midi_dir):
    """
    Load all MIDI files from the specified directory.
    """
    midi_data_list = []
    for filename in os.listdir(midi_dir):
        if filename.lower().endswith(('.mid', '.midi')):
            try:
                midi_path = os.path.join(midi_dir, filename)
                midi_data = pretty_midi.PrettyMIDI(midi_path)
                midi_data_list.append(midi_data)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return midi_data_list


def tokenize_midi(midi_data, time_resolution=0.05):
    """
    Convert MIDI data to a sequence of tokens.

    Parameters:
    - midi_data: a PrettyMIDI object.
    - time_resolution: the smallest time unit (in seconds) for a time shift.

    Token types include:
      - TIME_SHIFT_n: advance time by n * time_resolution seconds.
      - NOTE_ON_pitch_velocity: a note-on event with the given pitch and velocity.
      - NOTE_OFF_pitch: a note-off event for the given pitch.
    """
    events = []

    # Iterate through all instruments in the MIDI file
    for instrument in midi_data.instruments:
        # Optionally, skip drum instruments if not needed
        if instrument.is_drum:
            continue
        # Add note on/off events for each note
        for note in instrument.notes:
            events.append((note.start, "note_on", note.pitch, note.velocity))
            events.append((note.end, "note_off", note.pitch))

    # Sort events by time
    events.sort(key=lambda x: x[0])

    tokens = []
    previous_time = 0.0

    # Create tokens, including time shift tokens if there is a gap between events
    for event in events:
        current_time = event[0]
        time_diff = current_time - previous_time

        # Quantize the time difference based on the resolution
        steps = int(round(time_diff / time_resolution))
        if steps > 0:
            tokens.append(f"TIME_SHIFT_{steps}")

        # Append the note event token
        if event[1] == "note_on":
            tokens.append(f"NOTE_ON_{event[2]}_{event[3]}")
        elif event[1] == "note_off":
            tokens.append(f"NOTE_OFF_{event[2]}")

        previous_time = current_time

    return tokens

##########################################
# Step 1: Build Vocabulary from Token Data
##########################################
def build_vocab(token_sequences, min_freq=1):
    """
    Build a vocabulary from tokenized sequences.
    Special tokens: <PAD>, <SOS>, <EOS>, <UNK>
    """
    counter = Counter()
    for seq in token_sequences:
        counter.update(seq)
    tokens = [token for token, count in counter.items() if count >= min_freq]
    special_tokens = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
    vocab = special_tokens + tokens
    token_to_idx = {token: idx for idx, token in enumerate(vocab)}
    idx_to_token = {idx: token for token, idx in token_to_idx.items()}
    return token_to_idx, idx_to_token


##########################################
# Step 2: Create a Custom Dataset
##########################################
class TokenDataset(Dataset):
    def __init__(self, token_sequences, token_to_idx, max_length=100):
        """
        token_sequences: List of token lists.
        token_to_idx: Dictionary mapping tokens to indices.
        max_length: Maximum length of a sequence (will pad or truncate).
        """
        self.sequences = token_sequences
        self.token_to_idx = token_to_idx
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    """
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # Add Start Of Sequence and End Of Sequence tokens
        seq = ["<SOS>"] + seq + ["<EOS>"]
        # Convert tokens to indices, using <UNK> if token not found
        seq_idx = [self.token_to_idx.get(token, self.token_to_idx["<UNK>"]) for token in seq]
        # Pad sequence if necessary
        if len(seq_idx) < self.max_length:
            seq_idx += [self.token_to_idx["<PAD>"]] * (self.max_length - len(seq_idx))
        else:
            seq_idx = seq_idx[:self.max_length]
        # Prepare input (all tokens except last) and target (all tokens except first)
        input_seq = torch.tensor(seq_idx[:-1], dtype=torch.long)
        target_seq = torch.tensor(seq_idx[1:], dtype=torch.long)
        return input_seq, target_seq
    """


    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # If seq is a string, convert it into a list of tokens (assuming tokens are space-separated)
        if isinstance(seq, str):
            seq = seq.split()

        # Add Start Of Sequence and End Of Sequence tokens
        seq = ["<SOS>"] + seq + ["<EOS>"]
        # Convert tokens to indices, using <UNK> if token not found
        seq_idx = [self.token_to_idx.get(token, self.token_to_idx["<UNK>"]) for token in seq]
        # Pad sequence if necessary
        if len(seq_idx) < self.max_length:
            seq_idx += [self.token_to_idx["<PAD>"]] * (self.max_length - len(seq_idx))
        else:
            seq_idx = seq_idx[:self.max_length]
        # Prepare input (all tokens except last) and target (all tokens except first)
        input_seq = torch.tensor(seq_idx[:-1], dtype=torch.long)
        target_seq = torch.tensor(seq_idx[1:], dtype=torch.long)
        return input_seq, target_seq


##########################################
# Step 3: Define the VAE Model
##########################################
class MusicVAE(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, latent_size=64, num_layers=1):
        super(MusicVAE, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder_lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_logvar = nn.Linear(hidden_size, latent_size)

        self.latent_to_hidden = nn.Linear(latent_size, hidden_size)
        self.decoder_lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.outputs_fc = nn.Linear(hidden_size, vocab_size)

    def encode(self, x):
        embedded = self.embedding(x)  # (batch, seq_len, embed_size)
        _, (h, _) = self.encoder_lstm(embedded)
        # Use the last hidden state from the final layer
        h_last = h[-1]  # (batch, hidden_size)
        mu = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, input_seq):
        # Prepare initial hidden state for decoder from latent vector
        hidden = self.latent_to_hidden(z)
        hidden = hidden.unsqueeze(0)  # (1, batch, hidden_size)
        cell = torch.zeros_like(hidden)  # Initialize cell state to zeros
        embedded = self.embedding(input_seq)
        output, _ = self.decoder_lstm(embedded, (hidden, cell))
        logits = self.outputs_fc(output)  # (batch, seq_len, vocab_size)
        return logits

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z, x)
        return logits, mu, logvar


##########################################
# Step 4: Define the Loss Function
##########################################
def loss_function(logits, targets, mu, logvar):
    # Reconstruction loss: Flatten predictions and targets
    reconstruction_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss /= logits.size(0)  # normalize by batch size
    return reconstruction_loss + kl_loss


##########################################
# Step 5: Training Loop
##########################################
def train_vae(model, dataloader, optimizer, epochs=10, device='cpu'):
    model.train()
    model.to(device)
    for epoch in range(epochs):
        total_loss = 0.0
        for input_seq, target_seq in dataloader:
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            optimizer.zero_grad()
            logits, mu, logvar = model(input_seq)
            loss = loss_function(logits, target_seq, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")


##########################################
# Main: Putting It All Together
##########################################
if __name__ == "__main__":
    # Load MIDI files for Training.
    midi_directory = '/Users/alimomennasab/Desktop/CS4990/CS4990-Generative-AI/MIDI-VAE_PaperData/Classic and Bach/barock/Bach/Bwv001- 400 Chorales'
    midi_files = load_midi_files(midi_directory)

    # Tokenize all MIDI files into a list of sequences
    tokenized_data = []
    for i, midi_data in enumerate(midi_files):
        tokens = tokenize_midi(midi_data)
        tokenized_data.append(tokens) 
        print(f"MIDI file {i} tokens: {tokens[:20]} ...")

    print("Number of sequences:", len(tokenized_data))
    print("Sample sequence length:", len(tokenized_data[0]))

    # Build vocabulary
    token_to_idx, idx_to_token = build_vocab(tokenized_data)
    vocab_size = len(token_to_idx)
    print("Vocabulary size:", vocab_size)
    print("Sample tokens:", list(token_to_idx.keys())[:10])

    # Save token mappings
    with open("token_to_idx.pkl", "wb") as f:
        pickle.dump(token_to_idx, f)
    with open("idx_to_token.pkl", "wb") as f:
        pickle.dump(idx_to_token, f)

    # Create dataset and dataloader
    max_seq_length = 100  
    dataset = TokenDataset(tokenized_data, token_to_idx, max_length=max_seq_length)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)  

    # Initialize model and optimizer
    model = MusicVAE(vocab_size, embed_size=128, hidden_size=256, latent_size=64)
    optimizer = optim.Adam(model.parameters(), lr=0.001) 

    # Train the VAE model
    train_vae(model, dataloader, optimizer, epochs=20, device='cpu') 

    # Save model
    torch.save(model.state_dict(), "music_vae.pth")
    print("Model saved as music_vae.pth")
