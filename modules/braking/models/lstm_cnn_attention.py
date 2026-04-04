import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sequence_autoencoder import SequenceAutoencoder


class AttentionLayer(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_outputs):
        # lstm_outputs: (batch, time, hidden_dim)
        scores = self.attention(lstm_outputs)  # (batch, time, 1)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights * lstm_outputs, dim=1)
        return context


class LSTMCNNAttention(nn.Module):

    def __init__(self, num_features=3, num_classes=3):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first=True)

        self.attention = AttentionLayer(hidden_dim=64)

        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        # x: (batch, time, features) → CNN expects (batch, channels, time)
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)
        context = self.attention(lstm_out)

        x = F.relu(self.fc1(context))
        return self.fc2(x)


class AE_LSTMCNNAttention(nn.Module):
    """CNN+LSTM+Attention classifier that uses a pretrained encoder as the first stage."""

    def __init__(self, latent_dim=4, num_classes=3):
        super().__init__()

        self.autoencoder = SequenceAutoencoder(input_dim=3, latent_dim=latent_dim)

        # Freeze all encoder layers, then unfreeze only the last one
        # so early layers stay stable while the last layer adapts to the classification task
        for param in self.autoencoder.encoder.parameters():
            param.requires_grad = False
        for param in self.autoencoder.encoder[-1].parameters():
            param.requires_grad = True

        # CNN input channels = latent_dim (not raw features)
        self.conv1 = nn.Conv1d(in_channels=latent_dim, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)

        self.lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first=True)

        self.attention = AttentionLayer(hidden_dim=64)

        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        with torch.no_grad():
            x = self.autoencoder.encode(x)  # (batch, time, latent_dim)

        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)
        context = self.attention(lstm_out)

        x = F.relu(self.fc1(context))
        return self.fc2(x)


if __name__ == "__main__":

    model = LSTMCNNAttention()
    dummy_input = torch.randn(2, 75, 3)
    output = model(dummy_input)
    print("Baseline Output shape:", output.shape)
    print("Baseline Output:", output)

    ae_model = AE_LSTMCNNAttention(latent_dim=4, num_classes=3)
    dummy_input = torch.randn(2, 75, 3)
    ae_output = ae_model(dummy_input)
    print("AE+Classifier Output shape:", ae_output.shape)
    print("AE+Classifier Output:", ae_output)