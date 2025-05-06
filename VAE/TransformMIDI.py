# conda: cs4990env

import pretty_midi
import torch
import torch.nn as nn
import pickle
import torch.nn.functional as F
import os
import pandas as pd


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
        num_genres=2,
    ):
        super(MusicVAE, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.genre_embedding = nn.Embedding(num_genres, embed_size)
        self.encoder_lstm = nn.LSTM(
            embed_size + embed_size, hidden_size, num_layers, batch_first=True
        )
        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_logvar = nn.Linear(hidden_size, latent_size)
        self.latent_to_hidden = nn.Linear(latent_size + embed_size, hidden_size)
        self.decoder_lstm = nn.LSTM(
            embed_size, hidden_size, num_layers, batch_first=True
        )
        self.outputs_fc = nn.Linear(hidden_size, vocab_size)

    def encode(self, x, genre):
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
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, input_seq, genre):
        genre_emb = self.genre_embedding(genre)
        z_cat = torch.cat([z, genre_emb], dim=1)
        hidden = self.latent_to_hidden(z_cat).unsqueeze(0)
        cell = torch.zeros_like(hidden)
        embedded = self.embedding(input_seq)
        output, _ = self.decoder_lstm(embedded, (hidden, cell))
        logits = self.outputs_fc(output)
        return logits

    def forward(self, x, genre):
        mu, logvar = self.encode(x, genre)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z, x, genre)
        return logits, mu, logvar


##########################################
# Step 2: MIDI Processing Functions
##########################################


def tokenize_midi(midi_data, time_resolution=0.05):
    """
    Convert MIDI data to a sequence of tokens.
    """
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
        steps = int(round(time_diff / time_resolution))
        if steps > 0:
            tokens.append(f"TIME_SHIFT_{steps}")
        if event[1] == "note_on":
            tokens.append(f"NOTE_ON_{event[2]}_{event[3]}")
        elif event[1] == "note_off":
            tokens.append(f"NOTE_OFF_{event[2]}")
        previous_time = current_time

    return tokens


def tokens_to_midi(
    tokens,
    time_resolution=0.05,
    default_duration=0.5,
    output_path="generated_output.mid",
):
    """
    Convert a list of tokens back into a MIDI file.
    """
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
# Step 3: Genre Transfer Functions
##########################################


def generate_transfer(
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
    Perform genre transfer using temperature-based sampling.
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
# Main
##########################################


def transform_midi(
    input_midi_path,
    output_midi_path,
    source_genre,
    target_genre,
    run_num="0",
    temperature=1.0,
):
    """
    Transform a MIDI file from one genre to another.

    Args:
        input_midi_path: Path to the input MIDI file
        output_midi_path: Path to save the transformed MIDI file
        source_genre: Source genre (0 for classical, 1 for jazz)
        target_genre: Target genre (0 for classical, 1 for jazz)
        run_num: Run number for loading the model and vocabulary
        temperature: Temperature for sampling (higher = more random)
    """
    # Load model and vocabulary
    run_folder = os.path.join("VAE", "outputs", f"run{run_num}")

    # Check if required files exist
    model_path = os.path.join(run_folder, "music_vae_genre_transfer.pth")
    token_to_idx_path = os.path.join(run_folder, "token_to_idx.pkl")
    idx_to_token_path = os.path.join(run_folder, "idx_to_token.pkl")

    if not all(
        os.path.exists(p)
        for p in [model_path, token_to_idx_path, idx_to_token_path]
    ):
        raise FileNotFoundError(
            f"Required files not found in {run_folder}. Please ensure the model and vocabulary files exist."
        )

    try:
        with open(token_to_idx_path, "rb") as f:
            token_to_idx = pickle.load(f)
        with open(idx_to_token_path, "rb") as f:
            idx_to_token = pickle.load(f)
    except Exception as e:
        raise Exception(f"Error loading vocabulary files: {str(e)}")

    # Initialize model
    model = MusicVAE(
        vocab_size=len(token_to_idx),
        embed_size=128,
        hidden_size=256,
        latent_size=64,
        num_genres=3,
    )

    try:
        model.load_state_dict(
            torch.load(
                model_path,
                map_location=torch.device("cuda"),
            )
        )
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

    model.eval()

    # Load and tokenize input MIDI
    try:
        midi_data = pretty_midi.PrettyMIDI(input_midi_path)
    except Exception as e:
        raise Exception(f"Error loading MIDI file: {str(e)}")

    tokens = tokenize_midi(midi_data)
    input_length = len(tokens)
    print(f"Input MIDI token length: {input_length}")

    # Convert tokens to indices
    input_seq = torch.tensor(
        [
            token_to_idx.get(tok, token_to_idx["<UNK>"])
            for tok in ["<SOS>"] + tokens + ["<EOS>"]
        ][:300],
        dtype=torch.long,
    )

    # Generate transfer with target length
    transferred_tokens = generate_transfer(
        model=model,
        input_seq=input_seq,
        token_to_idx=token_to_idx,
        idx_to_token=idx_to_token,
        source_genre=source_genre,
        target_genre=target_genre,
        temperature=temperature,
        max_length=input_length * 2,
    )

    # Trim or extend to match input length
    if len(transferred_tokens) > input_length:
        transferred_tokens = transferred_tokens[:input_length]
    elif len(transferred_tokens) < input_length:
        while len(transferred_tokens) < input_length:
            transferred_tokens.extend(transferred_tokens[-4:])
        transferred_tokens = transferred_tokens[:input_length]

    print(f"Output MIDI token length: {len(transferred_tokens)}")
    print(
        f"Length difference: {abs(len(transferred_tokens) - input_length)} tokens"
    )

    # Convert back to MIDI
    try:
        tokens_to_midi(transferred_tokens, output_path=output_midi_path)
    except Exception as e:
        raise Exception(f"Error saving output MIDI file: {str(e)}")


if __name__ == "__main__":
    # Example usage with autumn leaves jazz piece
    input_midi = (
        "../MIDI-VAE_PaperData/Jazz/autumn_leaves_jpa.mid"  # Input MIDI file
    )
    output_midi = "outputs/autumn_leaves_classical.mid"  # Output will be saved in outputs directory

    # 1 for jazz (source), 0 for classical (target)
    transform_midi(
        input_midi_path=input_midi,
        output_midi_path=output_midi,
        run_num="1k_epochs",
        source_genre=1,  # jazz
        target_genre=0,  # classical
        temperature=0.6,  # Adjust for more/less randomness
    )
