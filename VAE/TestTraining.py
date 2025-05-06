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
from tqdm import tqdm
import matplotlib.pyplot as plt

##########################################
# Step 0: Data Pre-Processing
##########################################


def load_midi_files(midi_dir):
    """
    Load all MIDI files from the specified directory.
    """
    midi_data_list = []
    for filename in os.listdir(midi_dir):
        if filename.lower().endswith((".mid", ".midi")):
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

        # Round the time difference to the nearest step based on a fixed resolution
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
    def __init__(
        self, token_sequences, token_to_idx, genre_labels, max_length=100
    ):
        """
        token_sequences: List of token lists.
        token_to_idx: Dictionary mapping tokens to indices.
        genre_labels: List of genre indices.
        max_length: Maximum length of a sequence (will pad or truncate).
        """
        self.sequences = token_sequences
        self.token_to_idx = token_to_idx
        self.max_length = max_length
        self.genre_labels = genre_labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # If seq is a string, convert it into a list of tokens
        if isinstance(seq, str):
            seq = seq.split()

        # Add Start Of Sequence and End Of Sequence tokens
        seq = ["<SOS>"] + seq + ["<EOS>"]
        # Convert tokens to indices, using <UNK> if token not found
        seq_idx = [
            self.token_to_idx.get(token, self.token_to_idx["<UNK>"])
            for token in seq
        ]
        # Pad the sequence if necessary
        if len(seq_idx) < self.max_length:
            seq_idx += [self.token_to_idx["<PAD>"]] * (
                self.max_length - len(seq_idx)
            )
        else:
            seq_idx = seq_idx[: self.max_length]
        # Prepare input
        input_seq = torch.tensor(
            seq_idx[:-1], dtype=torch.long
        )  # input sequence: every token except the last EOS token
        target_seq = torch.tensor(
            seq_idx[1:], dtype=torch.long
        )  # target sequence: every token except the first SOS token
        genre = torch.tensor(self.genre_labels[idx], dtype=torch.long)
        return input_seq, target_seq, genre


##########################################
# Step 3: Define the VAE Model
##########################################


class MusicVAE(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_size=128,
        hidden_size=256,
        latent_size=64,
        num_layers=1,
        num_genres=3,
    ):
        super(MusicVAE, self).__init__()
        # Embedding layer to convert token indices into embeddings
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Embedding layer for genre labels
        self.genre_embedding = nn.Embedding(num_genres, embed_size)
        # Encoder LSTM to process input sequences
        self.encoder_lstm = nn.LSTM(
            embed_size + embed_size, hidden_size, num_layers, batch_first=True
        )
        # Linear layers to obtain the mean and log variance for the latent distribution
        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_logvar = nn.Linear(hidden_size, latent_size)
        # Linear layer to transform latent vector and genre into initial decoder hidden state
        self.latent_to_hidden = nn.Linear(latent_size + embed_size, hidden_size)
        # Decoder LSTM to generate output sequences
        self.decoder_lstm = nn.LSTM(
            embed_size, hidden_size, num_layers, batch_first=True
        )
        # Final linear layer to map LSTM outputs to vocabulary logits
        self.outputs_fc = nn.Linear(hidden_size, vocab_size)

    def encode(self, x, genre):
        """
        Encode input sequence into a latent space, conditioned on genre.

        Args:
            x: Tensor of token indices with shape (batch, seq_len)
            genre: Tensor of genre indices with shape (batch,)

        Returns:
            mu: Mean of the latent distribution.
            logvar: Log variance of the latent distribution.
        """
        embedded = self.embedding(x)
        genre_emb = (
            self.genre_embedding(genre).unsqueeze(1).repeat(1, x.size(1), 1)
        )
        combined = torch.cat([embedded, genre_emb], dim=2)
        _, (h, _) = self.encoder_lstm(combined)
        h_last = h[-1]
        mu = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Apply the reparameterization trick to sample from the latent distribution.

        Args:
            mu: Mean of the latent distribution.
            logvar: Log variance of the latent distribution.

        Returns:
            z: Sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, input_seq, genre):
        """
        Decode the latent vector to generate an output sequence, conditioned on genre.

        Args:
            z: Latent vector.
            input_seq: Input sequence for the decoder (for teacher forcing during training).
            genre: Genre label tensor.

        Returns:
            logits: Unnormalized scores for each token in the vocabulary.
        """
        genre_emb = self.genre_embedding(genre)
        z_cat = torch.cat([z, genre_emb], dim=1)
        hidden = self.latent_to_hidden(z_cat).unsqueeze(0)
        cell = torch.zeros_like(hidden)
        embedded = self.embedding(input_seq)
        output, _ = self.decoder_lstm(embedded, (hidden, cell))
        logits = self.outputs_fc(output)
        return logits

    def forward(self, x, genre):
        """
        Forward pass through the VAE.

        Args:
            x: Input token sequence tensor.
            genre: Genre label tensor.

        Returns:
            logits: Decoder output logits.
            mu: Mean of the latent distribution.
            logvar: Log variance of the latent distribution.
        """
        mu, logvar = self.encode(x, genre)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z, x, genre)
        return logits, mu, logvar


##########################################
# Step 4: Define the Loss Function
##########################################


def loss_function(logits, targets, mu, logvar):
    # Reconstruction loss: Flatten predictions and targets
    reconstruction_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0
    )
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss /= logits.size(0)  # normalize by batch size
    return reconstruction_loss + kl_loss


##########################################
# Step 5: Training Loop
##########################################


def train_vae(model, dataloader, optimizer, epochs=20, device="cpu"):
    model.train()
    model.to(device)
    total_loss_list = []
    for epoch in range(epochs):
        progress_bar = tqdm(
            dataloader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch"
        )
        total_loss = 0.0
        for input_seq, target_seq, genre in dataloader:
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            genre = genre.to(device)
            optimizer.zero_grad()
            logits, mu, logvar = model(input_seq, genre)
            loss = loss_function(logits, target_seq, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        total_loss_list.append(avg_loss)
        progress_bar.set_postfix(loss=avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    # Plot the loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(total_loss_list, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("VAE Training Loss")
    plt.legend()
    plt.grid(True)
    if run_folder:
        plt.savefig(os.path.join(run_folder, "vae_training_loss.png"))
        print(f"Saved training loss plot to {run_folder}")


##########################################
# Main
##########################################

if __name__ == "__main__":
    run_num = "1k_epochs"
    run_folder = os.path.join("outputs", f"run{run_num}")
    os.makedirs(run_folder, exist_ok=True)
    print(f"Saving outputs to {run_folder}")

    # Load MIDI files for all genres
    genre_paths = {
        "classical": "../MIDI-VAE_PaperData/Classic and Bach/classic/Beethoven",
        "jazz": "../MIDI-VAE_PaperData/Jazz",
        "pop": "../MIDI-VAE_PaperData/Pop",
    }

    tokenized_data = []
    genre_labels = []

    # Tokenize the MIDI files and add genre labels
    for idx, (genre_name, genre_dir) in enumerate(genre_paths.items()):
        midi_files = load_midi_files(genre_dir)
        tokenized = [tokenize_midi(midi) for midi in midi_files]
        tokenized_data.extend(tokenized)
        genre_labels.extend([idx] * len(tokenized))

    print("Number of sequences:", len(tokenized_data))
    print("Sample sequence length:", len(tokenized_data[0]))

    # Build vocabulary library from tokenized data
    token_to_idx, idx_to_token = build_vocab(tokenized_data)

    with open(os.path.join(run_folder, "token_to_idx.pkl"), "wb") as f:
        pickle.dump(token_to_idx, f)

    with open(os.path.join(run_folder, "idx_to_token.pkl"), "wb") as f:
        pickle.dump(idx_to_token, f)

    # Load the vocabulary
    dataset = TokenDataset(
        tokenized_data, token_to_idx, genre_labels, max_length=300
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model
    model = MusicVAE(
        vocab_size=len(token_to_idx),
        embed_size=128,
        hidden_size=256,
        latent_size=64,
        num_genres=len(genre_paths),
    )
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    train_vae(model, dataloader, optimizer, epochs=1000, device="cuda")

    torch.save(
        model.state_dict(),
        os.path.join(run_folder, "music_vae_genre_transfer.pth"),
    )
    print("Model saved!")
