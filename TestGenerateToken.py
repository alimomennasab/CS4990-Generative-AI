# conda: CS4990Env

import torch
import pretty_midi
import torch
import torch.nn as nn
import pickle
import torch.nn.functional as F

class MusicVAE(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, latent_size=64, num_layers=1):
        super(MusicVAE, self).__init__()
        # Embedding layer to convert token indices into embeddings
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Encoder LSTM to process input sequences
        self.encoder_lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        # Linear layers to obtain the mean and log variance for the latent distribution
        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_logvar = nn.Linear(hidden_size, latent_size)
        # Linear layer to transform latent vector into initial decoder hidden state
        self.latent_to_hidden = nn.Linear(latent_size, hidden_size)
        # Decoder LSTM to generate output sequences
        self.decoder_lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        # Final linear layer to map LSTM outputs to vocabulary logits
        self.outputs_fc = nn.Linear(hidden_size, vocab_size)

    def encode(self, x):
        """
        Encode input sequence into a latent space.

        Args:
            x: Tensor of token indices with shape (batch, seq_len)

        Returns:
            mu: Mean of the latent distribution.
            logvar: Log variance of the latent distribution.
        """
        embedded = self.embedding(x)  # Shape: (batch, seq_len, embed_size)
        _, (h, _) = self.encoder_lstm(embedded)
        h_last = h[-1]  # Use the last hidden state from the final LSTM layer (batch, hidden_size)
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

    def decode(self, z, input_seq):
        """
        Decode the latent vector to generate an output sequence.

        Args:
            z: Latent vector.
            input_seq: Input sequence for the decoder (for teacher forcing during training).

        Returns:
            logits: Unnormalized scores for each token in the vocabulary.
        """
        # Transform latent vector to initial hidden state for the decoder LSTM
        hidden = self.latent_to_hidden(z)
        hidden = hidden.unsqueeze(0)  # Shape: (1, batch, hidden_size)
        cell = torch.zeros_like(hidden)  # Initialize cell state with zeros
        embedded = self.embedding(input_seq)
        output, _ = self.decoder_lstm(embedded, (hidden, cell))
        logits = self.outputs_fc(output)  # Shape: (batch, seq_len, vocab_size)
        return logits

    def forward(self, x):
        """
        Forward pass through the VAE.

        Args:
            x: Input token sequence tensor.

        Returns:
            logits: Decoder output logits.
            mu: Mean of the latent distribution.
            logvar: Log variance of the latent distribution.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z, x)
        return logits, mu, logvar


def generate_music(model, token_to_idx, idx_to_token, latent_size=64, max_length=100, device='cpu'):
    """
    Generate a sequence of music tokens using the trained VAE.

    Parameters:
      model: The trained VAE model.
      token_to_idx: Dictionary mapping tokens to indices.
      idx_to_token: Dictionary mapping indices to tokens.
      latent_size: Dimension of the latent vector.
      max_length: Maximum length of generated token sequence.
      device: 'cpu' or 'cuda'.

    Returns:
      A list of generated tokens.
    """
    model.eval()
    with torch.no_grad():
        # Sample a latent vector from standard normal distribution
        z = torch.randn(1, latent_size).to(device)

        # Prepare initial hidden state for the decoder using the latent vector
        hidden = model.latent_to_hidden(z)
        hidden = hidden.unsqueeze(0)  # shape: (1, batch, hidden_size)
        cell = torch.zeros_like(hidden)

        # Start with the <SOS> token
        input_token = torch.tensor([[token_to_idx["<SOS>"]]], dtype=torch.long).to(device)
        generated_tokens = []

        # Iteratively decode tokens
        for i in range(max_length):
            embedded = model.embedding(input_token)  # shape: (1, 1, embed_size)
            output, (hidden, cell) = model.decoder_lstm(embedded, (hidden, cell))
            logits = model.outputs_fc(output)  # shape: (1, 1, vocab_size)

            # Choose next token (greedy approach)
            next_token = torch.argmax(logits, dim=-1)  # shape: (1, 1)
            token_id = next_token.item()
            token_str = idx_to_token[token_id]

            if token_str == "<EOS>":
                break
            generated_tokens.append(token_str)
            input_token = next_token  # use the predicted token as next input

    return generated_tokens

def generate_music_with_temperature(model, token_to_idx, idx_to_token, latent_size=64, max_length=100, temperature=1.0, device='cpu'):
    model.eval()
    with torch.no_grad():
        z = torch.randn(1, latent_size).to(device)
        hidden = model.latent_to_hidden(z).unsqueeze(0)
        cell = torch.zeros_like(hidden)
        input_token = torch.tensor([[token_to_idx["<SOS>"]]], dtype=torch.long).to(device)
        generated_tokens = []
        for i in range(max_length):
            embedded = model.embedding(input_token)
            output, (hidden, cell) = model.decoder_lstm(embedded, (hidden, cell))
            logits = model.outputs_fc(output).squeeze(1)

            # Apply temperature sampling to encourage diversity
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_id = next_token.item()
            token_str = idx_to_token[token_id]

            if token_str == "<EOS>":
                break
            generated_tokens.append(token_str)
            # input_token = next_token.unsqueeze(0)
            input_token = next_token

    return generated_tokens


def tokens_to_midi(tokens, time_resolution=0.05, default_duration=0.5, output_path="generated_output.mid"):
    """
    Convert a list of tokens back into a MIDI file.

    Parameters:
      tokens: List of token strings (e.g., "NOTE_ON_60_64", "TIME_SHIFT_5").
      time_resolution: Base time step in seconds.
      default_duration: Duration assigned to notes (if not explicitly set by NOTE_OFF tokens).
      output_path: Path to save the generated MIDI file.
    """
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)  # Using piano for example
    current_time = 0.0
    pending_notes = {}  # to store active notes: pitch -> start_time and velocity

    for token in tokens:
        if token.startswith("TIME_SHIFT_"):
            # Advance the current time
            shift_steps = int(token.split("_")[-1])
            current_time += shift_steps * time_resolution
        elif token.startswith("NOTE_ON_"):
            # Format: NOTE_ON_pitch_velocity
            parts = token.split("_")
            pitch = int(parts[2])
            velocity = int(parts[3])
            # Record note start time; actual note_off will set the end time
            pending_notes[pitch] = (current_time, velocity)
        elif token.startswith("NOTE_OFF_"):
            # Format: NOTE_OFF_pitch
            parts = token.split("_")
            pitch = int(parts[-1])
            if pitch in pending_notes:
                start_time, velocity = pending_notes.pop(pitch)
                # Create a note with a duration; if duration not specified, use default_duration
                note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start_time, end=current_time)
                piano.notes.append(note)

    # For any remaining active notes without a NOTE_OFF, end them after default_duration
    for pitch, (start_time, velocity) in pending_notes.items():
        note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start_time, end=start_time + default_duration)
        piano.notes.append(note)

    midi.instruments.append(piano)
    midi.write(output_path)
    print(f"MIDI file saved to {output_path}")


# Example usage:
if __name__ == "__main__":
    # Load my trained model, token_to_idx, and idx_to_token from training.
    with open("token_to_idx.pkl", "rb") as f:
        token_to_idx = pickle.load(f)
    with open("idx_to_token.pkl", "rb") as f:
        idx_to_token = pickle.load(f)

    # Variables are defined as in your training code:
    vocab_size = len(token_to_idx)  # token_to_idx should be available from your training process
    embed_size = 128
    hidden_size = 256
    latent_size = 64

    # Instantiate the model
    loaded_model = MusicVAE(vocab_size, embed_size=embed_size, hidden_size=hidden_size, latent_size=latent_size)
    # Load the state dictionary (map to CPU if needed)
    loaded_model.load_state_dict(torch.load("music_vae.pth", map_location=torch.device('cpu')))
    loaded_model.eval()  # Set to evaluation mode
    print("Model loaded successfully!")

    # Set device and latent_size as used during training.
    device = 'cpu'
    latent_size = 64  # change if needed

    # Generate tokens from the trained VAE model
    generated_tokens = generate_music(loaded_model, token_to_idx, idx_to_token, latent_size=latent_size, max_length=100, device=device)
    print("Generated token sequence:")
    print(generated_tokens)

    generated_tokens_with_temperature = generate_music_with_temperature(loaded_model, token_to_idx, idx_to_token, latent_size=64, max_length=100, temperature=1.0, device='cpu')
    print("Generated token sequence with temperature:")
    print(generated_tokens_with_temperature)

    # Convert the generated tokens to a MIDI file
    # tokens_to_midi(generated_tokens, time_resolution=0.05, default_duration=0.5, output_path="generated_output.mid")
    tokens_to_midi(generated_tokens, time_resolution=0.05, default_duration=0.1, output_path="generated_output.mid")

    tokens_to_midi(generated_tokens_with_temperature, time_resolution=0.05, default_duration=1.0, output_path="generated_output_with_temperature.mid")

