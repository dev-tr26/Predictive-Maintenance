from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

class _AutoencoderNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(64, 32, 8), dropout: float = 0.1):
        super().__init__()
        dims = [input_dim] + list(hidden_dims)
        
        encoder_layers = []
        for i in range(len(dims) - 1):
            encoder_layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(), nn.Dropout(dropout)]
            
        self.encoder = nn.Sequential(*encoder_layers[:-1])
        
        rev_dims = list(hidden_dims[::-1]) + [input_dim]
        decoder_layers = []
        for i in range(len(rev_dims) - 1):
            decoder_layers.append(nn.Linear(rev_dims[i], rev_dims[i+1]))
            if i < len(rev_dims) - 2:
                decoder_layers += [nn.ReLU(), nn.Dropout(dropout)]
        
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    
    
class TorchAutoencoderAnomalyDetector:
    def __init__(self, hidden_dims=(64, 32, 8), dropout=0.1, lr=1e-3,
                 batch_size=256, max_epochs=200, patience=15,
                 weight_decay=1e-5, threshold_percentile=97.5,
                 random_state=42, device: str | None = None):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.weight_decay = weight_decay
        self.threshold_percentile = threshold_percentile
        self.random_state = random_state
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model: _AutoencoderNet | None = None
        self.threshold_: float | None = None
        self.input_dim_: int | None = None
        self.train_losses_: list[float] = []
        self.val_losses_: list[float] = []
        
    def _split_train_val(self, X:np.ndarray, val_fraction: float = 0.1):
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]
        idx = rng.permutation(n)
        n_val = max(1, int(n * val_fraction))
        return X[idx[n_val:]], X[idx[:n_val]]
    
    def fit(self, X_healthy: np.ndarray, val_fraction : float=0.1, verbose=False):
        torch.manual_seed(self.random_state)
        X_healthy = np.asarray(X_healthy, dtype=np.float32)
        self.input_dim_ = X_healthy.shape[1]
        
        X_train, X_val = self._split_train_val(X_healthy, val_fraction)
        train_t = torch.from_numpy(X_train).to(self.device)
        val_t = torch.from_numpy(X_val).to(self.device)
        
        self.model = _AutoencoderNet(self.input_dim__, self.hidden_dims, self.dropout).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), 1)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0

        n_train = train_t.shape[0]
        for epoch in range(self.max_epochs):
            self.model.train()
            perm = torch.randperm(n_train, device=self.device)
            epoch_loss = 0.0
            for start in range(0, n_train, self.batch_size):
                batch_idx = perm[start:start + self.batch_size]
                batch = train_t[batch_idx]
                optimizer.zero_grad()
                recon = self.model(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.shape[0]
            epoch_loss /= n_train

            self.model.eval()
            with torch.no_grad():
                val_recon = self.model(val_t)
                val_loss = criterion(val_recon, val_t).item()

            self.train_losses_.append(epoch_loss)
            self.val_losses_.append(val_loss)
            if verbose and epoch % 10 == 0:
                print(f"epoch {epoch:3d}  train_mse={epoch_loss:.5f}  val_mse={val_loss:.5f}")

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch} (best val_mse={best_val_loss:.5f})")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        train_errors = self._reconstruction_error(X_healthy)
        self.threshold_ = float(np.percentile(train_errors, self.threshold_percentile))
        return self

    def _reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            X_t = torch.from_numpy(X).to(self.device)
            recon = self.model(X_t).cpu().numpy()
        return np.mean((X - recon) ** 2, axis=1)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Per-sample reconstruction error (higher = more anomalous)."""
        return self._reconstruction_error(X)

    def predict_anomaly(self, X: np.ndarray) -> np.ndarray:
        if self.threshold_ is None:
            raise RuntimeError("Call fit() before predict_anomaly().")
        return (self.score(X) > self.threshold_).astype(int)

    def anomaly_ratio(self, error: float) -> float:
        if not self.threshold_:
            return 0.0
        return float(error / self.threshold_)

    def save(self, path: str):
        torch.save({
            "state_dict": self.model.state_dict(),
            "input_dim": self.input_dim_,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "threshold": self.threshold_,
            "threshold_percentile": self.threshold_percentile,
        }, path)

    @classmethod
    def load(cls, path: str, device: str | None = None) -> "TorchAutoencoderAnomalyDetector":
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        payload = torch.load(path, map_location=device, weights_only=False)
        obj = cls(hidden_dims=payload["hidden_dims"], dropout=payload["dropout"],
                   threshold_percentile=payload["threshold_percentile"], device=device)
        obj.input_dim_ = payload["input_dim"]
        obj.model = _AutoencoderNet(obj.input_dim_, obj.hidden_dims, obj.dropout).to(obj.device)
        obj.model.load_state_dict(payload["state_dict"])
        obj.model.eval()
        obj.threshold_ = payload["threshold"]
        return obj