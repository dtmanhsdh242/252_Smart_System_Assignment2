"""
=============================================================================
Building a Basic Deep Neural Network from Scratch
Course: CO5119 - Intelligent Systems | Student: Đặng Tiến Mạnh - 2470569
=============================================================================
Architecture: Input → Hidden1 (128) → Hidden2 (64) → Output (10)
Dataset:      Synthetic multi-class classification (10 classes)
Key formulas implemented:
    Forward:  z[l] = W[l] @ a[l-1] + b[l]        (Eq. 1)
              a[l] = σ(z[l])                        (Eq. 2)
    Loss:     L = -Σ y_i * log(ŷ_i)               (Eq. 3)
    Backward: ∂L/∂W[l] via Chain Rule              (Eq. 4)
    Update:   W[l] ← W[l] - α * ∂L/∂W[l]          (Eq. 5)
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.random.seed(42)

# ─────────────────────────────────────────────
# 1. ACTIVATION FUNCTIONS & THEIR DERIVATIVES
# ─────────────────────────────────────────────

def relu(z):
    """Rectified Linear Unit: f(z) = max(0, z)"""
    return np.maximum(0, z)

def relu_derivative(z):
    """Derivative of ReLU: f'(z) = 1 if z > 0 else 0"""
    return (z > 0).astype(float)

def softmax(z):
    """
    Numerically stable Softmax for multi-class output layer.
    Subtracts max for numerical stability before exponentiation.
    """
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# ─────────────────────────────────────────────
# 2. DEEP NEURAL NETWORK CLASS
# ─────────────────────────────────────────────

class DeepNeuralNetwork:
    """
    A fully-connected Deep Neural Network with configurable architecture.

    Parameters
    ----------
    layer_dims : list of int
        Number of neurons per layer, e.g. [784, 128, 64, 10]
    learning_rate : float
        Step size α for gradient descent weight updates
    """

    def __init__(self, layer_dims, learning_rate=0.01):
        self.layer_dims    = layer_dims
        self.learning_rate = learning_rate
        self.params        = {}   # Stores W[l] and b[l] for each layer
        self.cache         = {}   # Stores z[l] and a[l] for backprop
        self.loss_history  = []

        self._initialize_parameters()

    # ── 2a. WEIGHT INITIALIZATION (He Initialization for ReLU) ──────────────
    def _initialize_parameters(self):
        """
        He initialization: W ~ N(0, sqrt(2 / n_in))
        Prevents vanishing/exploding gradients, optimized for ReLU activations.
        """
        L = len(self.layer_dims)
        for l in range(1, L):
            n_in  = self.layer_dims[l - 1]
            n_out = self.layer_dims[l]
            # He initialization scale factor
            scale = np.sqrt(2.0 / n_in)
            self.params[f"W{l}"] = np.random.randn(n_in, n_out) * scale
            self.params[f"b{l}"] = np.zeros((1, n_out))

        print(f"✔ Network initialized with {L-1} layers:")
        for l in range(1, L):
            print(f"   Layer {l}: {self.layer_dims[l-1]} → {self.layer_dims[l]}", end="")
            print(f"  [W shape: {self.params[f'W{l}'].shape}]")

    # ── 2b. FORWARD PROPAGATION (Eq. 1 & 2) ────────────────────────────────
    def forward(self, X):
        """
        Forward pass: propagate input X through all layers.

        For hidden layers:   a[l] = ReLU(W[l] @ a[l-1] + b[l])
        For output layer:    a[L] = Softmax(W[L] @ a[L-1] + b[L])

        Returns the final prediction probabilities (batch_size × n_classes).
        """
        self.cache["a0"] = X
        L = len(self.layer_dims) - 1

        for l in range(1, L + 1):
            W = self.params[f"W{l}"]
            b = self.params[f"b{l}"]
            a_prev = self.cache[f"a{l-1}"]

            # Eq. 1: z[l] = W[l] @ a[l-1] + b[l]
            z = a_prev @ W + b
            self.cache[f"z{l}"] = z

            # Eq. 2: a[l] = σ(z[l])
            if l == L:
                a = softmax(z)          # Output layer → probability distribution
            else:
                a = relu(z)             # Hidden layers → ReLU
            self.cache[f"a{l}"] = a

        return self.cache[f"a{L}"]

    # ── 2c. LOSS FUNCTION (Eq. 3) ───────────────────────────────────────────
    def compute_loss(self, y_pred, y_true):
        """
        Categorical Cross-Entropy Loss: L = -Σ y_i * log(ŷ_i)
        Averaged over the batch.

        Parameters
        ----------
        y_pred : (batch, classes) — softmax probabilities
        y_true : (batch, classes) — one-hot encoded labels
        """
        n = y_true.shape[0]
        # Clip predictions to avoid log(0)
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.sum(y_true * np.log(y_pred_clipped)) / n
        return loss

    # ── 2d. BACKPROPAGATION (Eq. 4 & 5) ────────────────────────────────────
    def backward(self, y_true):
        """
        Backpropagation via the Chain Rule:
            ∂L/∂W[l] = (∂L/∂a[l]) * (∂a[l]/∂z[l]) * (∂z[l]/∂W[l])

        Updates self.params with gradient descent step (Eq. 5).
        """
        n = y_true.shape[0]
        L = len(self.layer_dims) - 1
        grads = {}

        # ── Output layer gradient (softmax + cross-entropy combined) ────────
        # For softmax + cross-entropy, the combined gradient simplifies to:
        # δ[L] = ŷ - y
        delta = self.cache[f"a{L}"] - y_true

        # ── Backpropagate through each layer ────────────────────────────────
        for l in range(L, 0, -1):
            a_prev = self.cache[f"a{l-1}"]
            W      = self.params[f"W{l}"]

            # Eq. 4: ∂L/∂W[l] = a[l-1]ᵀ @ δ[l] / n
            grads[f"dW{l}"] = (a_prev.T @ delta) / n
            grads[f"db{l}"] = np.sum(delta, axis=0, keepdims=True) / n

            if l > 1:
                # Propagate delta to previous layer
                delta = (delta @ W.T) * relu_derivative(self.cache[f"z{l-1}"])

        # ── Eq. 5: Parameter update: W[l] ← W[l] - α * ∂L/∂W[l] ───────────
        for l in range(1, L + 1):
            self.params[f"W{l}"] -= self.learning_rate * grads[f"dW{l}"]
            self.params[f"b{l}"] -= self.learning_rate * grads[f"db{l}"]

    # ── 2e. TRAINING LOOP ───────────────────────────────────────────────────
    def train(self, X_train, y_train, X_val, y_val, epochs=200, batch_size=64):
        """
        Mini-batch Stochastic Gradient Descent training loop.
        Logs loss and accuracy every 20 epochs.
        """
        n_samples = X_train.shape[0]
        val_loss_history = []
        val_acc_history  = []
        train_acc_history = []

        print(f"\n{'─'*60}")
        print(f"  Training: {epochs} epochs | batch={batch_size} | lr={self.learning_rate}")
        print(f"{'─'*60}")
        print(f"  {'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>10}  {'Val Acc':>10}")
        print(f"{'─'*60}")

        for epoch in range(1, epochs + 1):
            # Shuffle training data
            idx = np.random.permutation(n_samples)
            X_shuf, y_shuf = X_train[idx], y_train[idx]

            epoch_loss = 0
            n_batches  = 0

            for start in range(0, n_samples, batch_size):
                X_batch = X_shuf[start : start + batch_size]
                y_batch = y_shuf[start : start + batch_size]

                y_pred     = self.forward(X_batch)
                batch_loss = self.compute_loss(y_pred, y_batch)
                epoch_loss += batch_loss
                n_batches  += 1
                self.backward(y_batch)

            avg_loss = epoch_loss / n_batches
            self.loss_history.append(avg_loss)

            # Compute accuracies
            train_acc = self.evaluate(X_train, y_train)
            val_acc   = self.evaluate(X_val,   y_val)
            val_pred  = self.forward(X_val)
            val_loss  = self.compute_loss(val_pred, y_val)
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            train_acc_history.append(train_acc)

            if epoch % 20 == 0 or epoch == 1:
                print(f"  {epoch:>6}  {avg_loss:>10.4f}  {train_acc:>9.1f}%  {val_acc:>9.1f}%")

        print(f"{'─'*60}")
        return val_loss_history, train_acc_history, val_acc_history

    # ── 2f. EVALUATION ──────────────────────────────────────────────────────
    def evaluate(self, X, y_true):
        """Compute accuracy (%) on a given dataset."""
        y_pred = self.forward(X)
        pred_classes = np.argmax(y_pred, axis=1)
        true_classes = np.argmax(y_true, axis=1)
        return np.mean(pred_classes == true_classes) * 100


# ─────────────────────────────────────────────
# 3. DATASET: SYNTHETIC MULTI-CLASS
# ─────────────────────────────────────────────

def generate_dataset(n_samples=2000, n_features=20, n_classes=10):
    """
    Generate a synthetic Gaussian multi-class classification dataset.
    Each class has a distinct mean vector in feature space.
    """
    X_list, y_list = [], []
    for c in range(n_classes):
        mean = np.random.randn(n_features) * 3
        X_c  = np.random.randn(n_samples // n_classes, n_features) + mean
        y_c  = np.full(n_samples // n_classes, c)
        X_list.append(X_c)
        y_list.append(y_c)

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    # One-hot encode labels
    y_onehot = np.zeros((len(y), n_classes))
    y_onehot[np.arange(len(y)), y.astype(int)] = 1

    # Normalize features
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y_onehot = X[idx], y_onehot[idx]

    # Train/val split (80/20)
    split = int(0.8 * len(X))
    return X[:split], y_onehot[:split], X[split:], y_onehot[split:]


# ─────────────────────────────────────────────
# 4. VISUALIZATION
# ─────────────────────────────────────────────

def plot_results(model, val_loss_history, train_acc_history, val_acc_history):
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle("Task 1: Basic DNN Training Results", fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    epochs = range(1, len(model.loss_history) + 1)

    # ── Plot 1: Training Loss ────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(epochs, model.loss_history, color="#2196F3", linewidth=2, label="Train Loss")
    ax1.plot(epochs, val_loss_history,   color="#FF5722", linewidth=2, label="Val Loss",
             linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Loss over Epochs")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── Plot 2: Accuracy ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(epochs, train_acc_history, color="#4CAF50", linewidth=2, label="Train Acc")
    ax2.plot(epochs, val_acc_history,   color="#FF9800", linewidth=2, label="Val Acc",
             linestyle="--")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy over Epochs")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ── Plot 3: Network Architecture Diagram ────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis("off")
    ax3.set_title("DNN Architecture", fontsize=11)

    layer_names  = ["Input\n(20)", "Hidden 1\n(128)", "Hidden 2\n(64)", "Output\n(10)"]
    layer_colors = ["#90CAF9", "#A5D6A7", "#A5D6A7", "#FFCC80"]
    x_positions  = [0.1, 0.37, 0.63, 0.9]

    for i, (xp, name, color) in enumerate(zip(x_positions, layer_names, layer_colors)):
        rect = plt.Rectangle((xp - 0.09, 0.3), 0.18, 0.4,
                              color=color, ec="gray", lw=1.5, zorder=2)
        ax3.add_patch(rect)
        ax3.text(xp, 0.5, name, ha="center", va="center",
                 fontsize=9, fontweight="bold", zorder=3)
        if i > 0:
            ax3.annotate("", xy=(xp - 0.09, 0.5),
                         xytext=(x_positions[i-1] + 0.09, 0.5),
                         arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))

    act_labels = ["", "ReLU", "ReLU", "Softmax"]
    for xp, act in zip(x_positions[1:], act_labels[1:]):
        ax3.text(xp, 0.22, act, ha="center", va="center",
                 fontsize=8, color="#555", style="italic")

    plt.savefig("/mnt/user-data/outputs/task1_basic_dnn_results.png",
                dpi=150, bbox_inches="tight")
    print("\n✔ Plot saved → task1_basic_dnn_results.png")
    plt.close()


# ─────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Building a Basic Deep Neural Network")
    print("=" * 60)

    # Generate dataset
    X_train, y_train, X_val, y_val = generate_dataset(
        n_samples=2000, n_features=20, n_classes=10
    )
    print(f"\n✔ Dataset ready:")
    print(f"   Train: {X_train.shape}  |  Val: {X_val.shape}")
    print(f"   Classes: 10  |  Features: 20")

    # Build DNN: 20 → 128 → 64 → 10
    dnn = DeepNeuralNetwork(
        layer_dims    = [20, 128, 64, 10],
        learning_rate = 0.05
    )

    # Train
    val_loss, train_acc, val_acc = dnn.train(
        X_train, y_train, X_val, y_val,
        epochs=200, batch_size=64
    )

    # Final metrics
    final_train_acc = dnn.evaluate(X_train, y_train)
    final_val_acc   = dnn.evaluate(X_val,   y_val)
    print(f"\n✔ Final Results:")
    print(f"   Train Accuracy : {final_train_acc:.2f}%")
    print(f"   Val   Accuracy : {final_val_acc:.2f}%")

    # Plot
    plot_results(dnn, val_loss, train_acc, val_acc)
    print("\n✔ Task 1 complete.\n")
