# conda: cs4990env

import pretty_midi
import torch
import torch.nn as nn
import pickle
import torch.nn.functional as F
import os
from collections import Counter


##########################################
# Step 1: Initialize VAE Model
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
# Step 2: Token-to-MIDI Conversion
##########################################


def tokens_to_midi(
    tokens,
    time_resolution=0.05,
    default_duration=0.5,
    output_path="generated_output.mid",
):
    """
    Convert a list of tokens back into a MIDI file.

    Parameters:
      tokens: List of token strings (e.g., "NOTE_ON_60_64", "TIME_SHIFT_5").
      time_resolution: Base time step in seconds.
      default_duration: Duration assigned to notes (if not explicitly set by NOTE_OFF tokens).
      output_path: Path to save the generated MIDI file.
    """
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)  # MIDI program number for piano
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
            parts = token.split("_")
            pitch = int(parts[-1])
            if pitch in pending_notes:
                start_time, velocity = pending_notes.pop(pitch)
                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=pitch,
                    start=start_time,
                    end=current_time,
                )
                piano.notes.append(note)

    for pitch, (start_time, velocity) in pending_notes.items():
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=start_time,
            end=start_time + default_duration,
        )
        piano.notes.append(note)

    midi.instruments.append(piano)
    midi.write(output_path)
    print(f"MIDI file saved to {output_path}")


##########################################
# Step 3: Genre Transfer Generation
##########################################


def test_generate_token_transfer_greedy(
    model,
    input_seq,
    token_to_idx,
    idx_to_token,
    source_genre,
    target_genre,
    max_length=100,
    device="cpu",
):
    """
    Perform genre transfer by encoding with one genre and decoding with another.
    """
    model.eval()
    with torch.no_grad():
        input_seq = input_seq.to(device).unsqueeze(0)
        source_genre = torch.tensor([source_genre], dtype=torch.long).to(device)
        target_genre = torch.tensor([target_genre], dtype=torch.long).to(device)

        mu, logvar = model.encode(input_seq, source_genre)
        z = model.reparameterize(mu, logvar)

        input_token = torch.tensor(
            [[token_to_idx["<SOS>"]]], dtype=torch.long
        ).to(device)
        hidden = model.latent_to_hidden(
            torch.cat([z, model.genre_embedding(target_genre)], dim=1)
        ).unsqueeze(0)
        cell = torch.zeros_like(hidden)

        generated_tokens = []
        for _ in range(max_length):
            embedded = model.embedding(input_token)
            output, (hidden, cell) = model.decoder_lstm(
                embedded, (hidden, cell)
            )
            logits = model.outputs_fc(output)
            next_token = torch.argmax(logits, dim=-1)
            token_id = next_token.item()
            token_str = idx_to_token[token_id]
            if token_str == "<EOS>":
                break
            generated_tokens.append(token_str)
            input_token = next_token

    return generated_tokens


def test_generate_token_transfer_with_temperature(
    model,
    input_seq,
    token_to_idx,
    idx_to_token,
    source_genre,
    target_genre,
    max_length=100,
    temperature=1.0,
    device="cpu",
):
    """
    Perform genre transfer using temperature-based sampling during decoding.
    """
    model.eval()
    with torch.no_grad():
        input_seq = input_seq.to(device).unsqueeze(0)
        source_genre = torch.tensor([source_genre], dtype=torch.long).to(device)
        target_genre = torch.tensor([target_genre], dtype=torch.long).to(device)
        mu, logvar = model.encode(input_seq, source_genre)
        z = model.reparameterize(mu, logvar)

        input_token = torch.tensor(
            [[token_to_idx["<SOS>"]]], dtype=torch.long
        ).to(device)
        hidden = model.latent_to_hidden(
            torch.cat([z, model.genre_embedding(target_genre)], dim=1)
        ).unsqueeze(0)
        cell = torch.zeros_like(hidden)

        generated_tokens = []
        for _ in range(max_length):
            embedded = model.embedding(input_token)
            output, (hidden, cell) = model.decoder_lstm(
                embedded, (hidden, cell)
            )
            logits = model.outputs_fc(output).squeeze(1)
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_id = next_token.item()
            token_str = idx_to_token[token_id]
            if token_str == "<EOS>":
                break
            generated_tokens.append(token_str)
            input_token = next_token

    return generated_tokens


##########################################
# Step 4: Run Genre Transfer
##########################################

if __name__ == "__main__":
    run_num = "3"
    run_folder = os.path.join("outputs", f"run{run_num}")
    os.makedirs(run_folder, exist_ok=True)
    print(f"Saving outputs to {run_folder}")

    with open(os.path.join(run_folder, "token_to_idx.pkl"), "rb") as f:
        token_to_idx = pickle.load(f)
    with open(os.path.join(run_folder, "idx_to_token.pkl"), "rb") as f:
        idx_to_token = pickle.load(f)

    vocab_size = len(token_to_idx)
    embed_size = 128
    hidden_size = 256
    latent_size = 64
    num_genres = 3

    model_path = os.path.join(run_folder, "music_vae_genre_transfer.pth")
    loaded_model = MusicVAE(
        vocab_size,
        embed_size,
        hidden_size,
        latent_size,
        num_layers=1,
        num_genres=num_genres,
    )
    loaded_model.load_state_dict(
        torch.load(model_path, map_location=torch.device("cpu"))
    )
    loaded_model.eval()
    print("Model loaded successfully!")

    # define genre paths
    genre_dirs = {
        0: (
            "classical",
            "../MIDI-VAE_PaperData/Classic and Bach/classic/Beethoven",
        ),
        1: (
            "jazz",
            "../MIDI-VAE_PaperData/Jazz",
        ),
        2: (
            "pop",
            "../MIDI-VAE_PaperData/Pop",
        ),
    }

    sample_inputs = {}

    for genre_idx, (genre_name, midi_dir) in genre_dirs.items():
        # load the midi files of a genre
        midi_files = []
        for f in os.listdir(midi_dir):
            if f.endswith((".mid", ".midi")):
                try:
                    midi = pretty_midi.PrettyMIDI(os.path.join(midi_dir, f))
                    midi_files.append(midi)
                except Exception as e:
                    print(f"Skipping {f} due to error: {e}")

        tokenized = []
        for midi in midi_files:
            events = []
            for instrument in midi.instruments:
                if instrument.is_drum:
                    continue
                for note in instrument.notes:
                    events.append(
                        (note.start, "note_on", note.pitch, note.velocity)
                    )
                    events.append((note.end, "note_off", note.pitch))
            events.sort(key=lambda x: x[0])

            tokens = []
            previous_time = 0.0
            for event in events:
                current_time = event[0]
                time_diff = current_time - previous_time
                steps = int(round(time_diff / 0.05))
                if steps > 0:
                    tokens.append(f"TIME_SHIFT_{steps}")
                if event[1] == "note_on":
                    tokens.append(f"NOTE_ON_{event[2]}_{event[3]}")
                elif event[1] == "note_off":
                    tokens.append(f"NOTE_OFF_{event[2]}")
                previous_time = current_time
            if tokens:
                sample_inputs[genre_idx] = torch.tensor(
                    [
                        token_to_idx.get(tok, token_to_idx["<UNK>"])
                        for tok in ["<SOS>"] + tokens + ["<EOS>"]
                    ][:100],
                    dtype=torch.long,
                )
                break

    # Generate greedy and temperature token sequences for each genre pair
    for source_genre in genre_dirs:
        for target_genre in genre_dirs:
            if source_genre != target_genre:
                src_name, _ = genre_dirs[source_genre]
                tgt_name, _ = genre_dirs[target_genre]

                # Greedy decoding
                transferred_tokens = test_generate_token_transfer_greedy(
                    model=loaded_model,
                    input_seq=sample_inputs[source_genre],
                    token_to_idx=token_to_idx,
                    idx_to_token=idx_to_token,
                    source_genre=source_genre,
                    target_genre=target_genre,
                    device="cuda",
                )
                print(
                    f"[{src_name.upper()} → {tgt_name.upper()}] Greedy token sequence:"
                )
                print(transferred_tokens)
                tokens_to_midi(
                    transferred_tokens,
                    output_path=os.path.join(
                        run_folder, f"transferred_{src_name}_to_{tgt_name}.mid"
                    ),
                )

                # Temperature decoding
                transferred_tokens_temp = (
                    test_generate_token_transfer_with_temperature(
                        model=loaded_model,
                        input_seq=sample_inputs[source_genre],
                        token_to_idx=token_to_idx,
                        idx_to_token=idx_to_token,
                        source_genre=source_genre,
                        target_genre=target_genre,
                        temperature=1.0,
                        device="cuda",
                    )
                )
                print(
                    f"[{src_name.upper()} to {tgt_name.upper()}] Temperature token sequence:"
                )
                print(transferred_tokens_temp)
                tokens_to_midi(
                    transferred_tokens_temp,
                    output_path=os.path.join(
                        run_folder,
                        f"transferred_{src_name}_to_{tgt_name}_temp.mid",
                    ),
                )
