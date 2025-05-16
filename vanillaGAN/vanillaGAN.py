# conda: cs4990env

import pretty_midi
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
import math


##########################################
# Step 0: Data Pre-Processing
##########################################

def load_midi_files(midi_dir):
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

def tokenize_midi(midi_data, time_resolution=0.1):
    shift_buckets = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32] 
    events = []
    for instrument in midi_data.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            events.append((note.start, "note_on", note.pitch, note.velocity))
            events.append((note.end, "note_off", note.pitch))
    events.sort(key=lambda x: x[0])

    tokens = []
    previous_time = 0.0
    for event in events:
        current_time = event[0]
        time_diff = current_time - previous_time
        steps = int(math.floor(time_diff / time_resolution))
        if steps > 0:
            # quantize step to closest bucket
            closest = min(shift_buckets, key=lambda x: abs(x - steps))
            tokens.append(f"TIME_SHIFT_{closest}")

        if event[1] == "note_on":
            tokens.append(f"NOTE_ON_{event[2]}_{event[3]}")
        elif event[1] == "note_off":
            tokens.append(f"NOTE_OFF_{event[2]}")

        previous_time = current_time

    return tokens

##########################################
# Step 1: Build Vocabulary
##########################################

def build_vocab(token_sequences, min_freq=1):
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
# Step 2: Custom Dataset
##########################################

class TokenDataset(Dataset):
    def __init__(self, token_sequences, token_to_idx, genre_labels, max_length=100):
        self.sequences = token_sequences
        self.token_to_idx = token_to_idx
        self.max_length = max_length
        self.genre_labels = genre_labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        seq = ["<SOS>"] + seq + ["<EOS>"]
        seq_idx = [self.token_to_idx.get(token, self.token_to_idx["<UNK>"]) for token in seq]
        if len(seq_idx) < self.max_length:
            seq_idx += [self.token_to_idx["<PAD>"]] * (self.max_length - len(seq_idx))
        else:
            seq_idx = seq_idx[:self.max_length]
        input_seq = torch.tensor(seq_idx[:-1], dtype=torch.long)
        target_seq = torch.tensor(seq_idx[1:], dtype=torch.long)
        genre = torch.tensor(self.genre_labels[idx], dtype=torch.long)
        return input_seq, target_seq, genre

def plot_top_tokens(token_seqs, run_folder, top_n=50, filename="top_tokens.png"):
    """
    Plots the top-N most frequent tokens in the dataset.

    Args:
        token_seqs (List[List[str]]): List of token sequences.
        run_folder (str): Folder path to save the plot.
        top_n (int): Number of top tokens to show.
        filename (str): Output filename for the plot.
    """
    from collections import Counter
    import matplotlib.pyplot as plt
    import os

    flat_tokens = [token for seq in token_seqs for token in seq]
    counter = Counter(flat_tokens)
    top_tokens = counter.most_common(top_n)

    if not top_tokens:
        print("No tokens to plot.")
        return

    tokens, freqs = zip(*top_tokens)
    plt.figure(figsize=(max(12, top_n * 0.4), 5))
    plt.bar(tokens, freqs)
    plt.xticks(rotation=90)
    plt.title(f"Top {top_n} Most Frequent Tokens")
    plt.tight_layout()

    plot_path = os.path.join(run_folder, filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved token frequency plot to {plot_path}")


##########################################
# Step 3: Define GAN 
##########################################

class Generator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, genre_dim):
        super(Generator, self).__init__()
        # Embedding layer to convert token indices into embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Genre embedding to represent 3 genres (classical, jazz, pop)
        self.genre_embedding = nn.Embedding(3, genre_dim)
        # LSTM layer to process the sequence of embeddings
        self.lstm = nn.LSTM(embedding_dim + genre_dim, hidden_dim, batch_first=True)
        # Final linear layer to map LSTM outputs to vocabulary logits
        self.fc = nn.Linear(hidden_dim, vocab_size)

    # Forward pass through the generator
    def forward(self, x, genre):
        embedded = self.embedding(x)
        genre_embed = self.genre_embedding(genre).unsqueeze(1).repeat(1, x.size(1), 1)
        combined = torch.cat([embedded, genre_embed], dim=-1)
        out, _ = self.lstm(combined)
        logits = self.fc(out)
        return logits

class Discriminator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, genre_dim):
        super(Discriminator, self).__init__()
        # Embedding layer to convert token indices into embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Genre embedding to represent 3 genres (classical, jazz, pop)
        self.genre_embedding = nn.Embedding(3, genre_dim)
        # LSTM layer to process the sequence of embeddings
        self.lstm = nn.LSTM(embedding_dim + genre_dim, hidden_dim, batch_first=True)
        # Final linear layer to map LSTM outputs to a single logit (real/fake)
        self.fc = nn.Linear(hidden_dim, 1)

    # Forward pass through the discriminator
    def forward(self, x, genre):
        embedded = self.embedding(x)
        genre_embed = self.genre_embedding(genre).unsqueeze(1).repeat(1, x.size(1), 1)
        combined = torch.cat([embedded, genre_embed], dim=-1)
        out, _ = self.lstm(combined)
        logits = self.fc(out[:, -1])
        return torch.sigmoid(logits)

##########################################
# Step 4: Training Loop
##########################################

def train_gan(generator, discriminator, dataloader, g_optimizer, d_optimizer, criterion, device='cpu', num_epochs=20, run_folder=None):
    generator.to(device)
    discriminator.to(device)

    d_losses = []
    g_losses = []

    for epoch in range(num_epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for input_seq, target_seq, genre in dataloader:
            input_seq, target_seq, genre = input_seq.to(device), target_seq.to(device), genre.to(device)

            d_optimizer.zero_grad()

            # Train discriminator
            real_preds = discriminator(target_seq, genre)
            real_loss = criterion(real_preds, torch.ones_like(real_preds) * 0.9)

            # Generate fake sequences
            with torch.no_grad():
                fake_logits = generator(input_seq, genre)
                fake_seq = fake_logits.argmax(dim=-1)

            # Discriminator on fake sequences
            fake_preds = discriminator(fake_seq.detach(), genre)
            fake_loss = criterion(fake_preds, torch.zeros_like(fake_preds))

            # Backpropagation for discriminator
            d_loss = real_loss + fake_loss
            d_loss.backward()
            d_optimizer.step()

            # Train generator
            g_optimizer.zero_grad()
            fake_logits = generator(input_seq, genre)
            fake_probs = F.softmax(fake_logits, dim=-1)
            token_range = torch.arange(fake_probs.size(-1), device=device).float()
            fake_seq = torch.matmul(fake_probs, token_range).long()  # [batch_size, seq_len]
            fake_preds = discriminator(fake_seq, genre)
            g_loss = criterion(fake_preds, torch.ones_like(fake_preds))
            g_loss.backward()
            g_optimizer.step()

            # Update progress bar
            progress_bar.set_postfix({
                "D Loss": f"{d_loss.item():.4f}",
                "G Loss": f"{g_loss.item():.4f}"
            })

        # Store losses for plotting
        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())
        print(f"Epoch [{epoch+1}/{num_epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    # Plot the losses
    plt.figure(figsize=(8, 5))
    plt.plot(d_losses, label="Discriminator Loss")
    plt.plot(g_losses, label="Generator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GAN Training Losses")
    plt.legend()
    plt.grid(True)

    if run_folder:
        plot_path = os.path.join(run_folder, "training_losses.png")
        plt.savefig(plot_path)
        print(f"Saved loss plot to {plot_path}")
    plt.close()

##########################################
# Step 5: Token-to-MIDI Conversion
##########################################

def tokens_to_midi(tokens, time_resolution=0.05, default_duration=0.5, output_path="generated_output.mid"):
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    current_time = 0.0
    pending_notes = {}

    for token in tokens:
        if token.startswith("TIME_SHIFT_"):
            shift_steps = int(token.split("_")[-1])
            current_time += shift_steps * time_resolution
        elif token.startswith("NOTE_ON_"):
            parts = token.split("_")
            pitch = int(parts[2])
            velocity = int(parts[3])
            pending_notes[pitch] = (current_time, velocity)
        elif token.startswith("NOTE_OFF_"):
            pitch = int(token.split("_")[-1])
            if pitch in pending_notes:
                start_time, velocity = pending_notes.pop(pitch)
                note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start_time, end=current_time)
                piano.notes.append(note)

    for pitch, (start_time, velocity) in pending_notes.items():
        note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start_time, end=start_time + default_duration)
        piano.notes.append(note)

    midi.instruments.append(piano)
    midi.write(output_path)
    print(f"MIDI file saved to {output_path}")

##########################################
# Step 6: Genre Transfer Decoding
##########################################

def test_generate_token_transfer_greedy(generator, input_seq, idx_to_token, target_genre, max_length=100, device='cpu'):
    """
    Perform genre transfer using greedy decoding (argmax at each timestep)
    """
    generator.eval()
    generated_tokens = []

    with torch.no_grad():
        input_seq = input_seq.unsqueeze(0).to(device)
        genre_tensor = torch.tensor([target_genre], dtype=torch.long).to(device)
        logits = generator(input_seq, genre_tensor)
        input_logits = logits.squeeze(0)

        for t in range(min(max_length, input_logits.size(0))):
            token_id = torch.argmax(input_logits[t]).item()
            token_str = idx_to_token[token_id]
            if token_str == "<EOS>":
                break
            generated_tokens.append(token_str)

    return generated_tokens



def test_generate_token_transfer_with_temperature(generator, input_seq, idx_to_token, target_genre, max_length=100, temperature=1.0, device='cpu'):
    """
    Perform genre transfer using temperature-based sampling during decoding.
    """
    generator.eval()
    generated_tokens = []

    with torch.no_grad():
        input_seq = input_seq.unsqueeze(0).to(device)
        genre_tensor = torch.tensor([target_genre], dtype=torch.long).to(device)
        logits = generator(input_seq, genre_tensor)
        input_logits = logits.squeeze(0)

        for t in range(min(max_length, input_logits.size(0))):
            logit = input_logits[t]
            probs = F.softmax(logit / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_id = next_token.item()
            token_str = idx_to_token[token_id]
            if token_str == "<EOS>":
                break
            generated_tokens.append(token_str)

    return generated_tokens

def run_transfers(generator, dataset, idx_to_token, device, run_folder):
    genre_names = {0: "classical", 1: "jazz", 2: "pop"}
    genre_pairs = [(0, 1), (0, 2), (2, 1), (2, 0), (1, 0), (1, 2)]

    genre_to_sample = {}
    for i in range(len(dataset)):
        _, _, genre = dataset[i]
        genre_id = genre.item()
        if genre_id not in genre_to_sample:
            genre_to_sample[genre_id] = i
        if len(genre_to_sample) == 3:
            break

    for (src, tgt) in genre_pairs:
        idx = genre_to_sample[src]
        input_seq, _, _ = dataset[idx]
        src_name = genre_names[src]
        tgt_name = genre_names[tgt]

        greedy_save_path = os.path.join(run_folder, f"{src_name}_to_{tgt_name}.mid")
        temp_save_path = os.path.join(run_folder, f"{src_name}_to_{tgt_name}_temp.mid")


        # Greedy decoding
        generate_tokens = test_generate_token_transfer_greedy(generator, input_seq, idx_to_token, target_genre=tgt, device=device)
        tokens_to_midi(generate_tokens, output_path=greedy_save_path)

        # Temperature sampling
        generate_tokens_temp = test_generate_token_transfer_with_temperature(generator, input_seq, idx_to_token, target_genre=tgt, temperature=1.2, device=device)
        tokens_to_midi(generate_tokens_temp, output_path=temp_save_path)

##########################################
# Main
##########################################
def main():
    epochs = 20
    run_num = "9"

    run_folder = os.path.join("outputs", f"run{run_num}")
    os.makedirs(run_folder, exist_ok=True)
    print(f"Saving outputs to {run_folder}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    genre_paths = {
        "classical": "../MIDI-VAE_PaperData/Classic and Bach/classic/Beethoven",
        "jazz": "../MIDI-VAE_PaperData/Jazz",
        "pop": "../MIDI-VAE_PaperData/Pop"
    }

    genre_to_idx = {"classical": 0, "jazz": 1, "pop": 2}
    token_seqs, genre_labels = [], []

    for genre, path in genre_paths.items():
        midis = load_midi_files(path)
        for i, midi in enumerate(midis):
            tokens = tokenize_midi(midi)
            token_seqs.append(tokens)
            genre_labels.append(genre_to_idx[genre])

            # Debug the first few tokenizations across all genres
            if len(token_seqs) <= 5:
                print(f"{genre.upper()} Sample {i}")
                print("First 30 tokens:", tokens[:30])
                print("Total tokens:", len(tokens))
                print("Unique token types:", len(set(tokens)))
                print()
    plot_top_tokens(token_seqs, run_folder)

    token_to_idx, idx_to_token = build_vocab(token_seqs)
    dataset = TokenDataset(token_seqs, token_to_idx, genre_labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    vocab_size = len(token_to_idx)
    generator = Generator(vocab_size, 128, 256, 16)
    discriminator = Discriminator(vocab_size, 128, 256, 16)

    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
    criterion = nn.BCELoss()

    train_gan(generator, discriminator, dataloader, g_optimizer, d_optimizer, criterion, device=device, num_epochs=epochs, run_folder=run_folder)
    run_transfers(generator, dataset, idx_to_token, device, run_folder)

main()