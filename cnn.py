"""
=============================================================================
Convolutional Neural Networks (CNNs) for Image Data
Course: CO5119 - Intelligent Systems | Student: Đặng Tiến Mạnh - 2470569
=============================================================================

CNN Architecture:
    Input (8×8×1) → Conv(3×3, 8 filters) → ReLU → MaxPool(2×2)
                  → Conv(3×3, 16 filters) → ReLU → MaxPool(2×2) (if size allows)
                  → Flatten → Dense(64) → ReLU → Dense(10) → Softmax

Key CNN concepts implemented:
    ● Convolution:    (I * K)[i,j] = Σ_m Σ_n I[i+m, j+n] * K[m,n]
    ● Max Pooling:    Retains the maximum activation in each pool window
    ● Stride & Padding: Controls output spatial dimensions
    ● Parameter Sharing: A single filter slides across the entire image
    ● Receptive Field: Each output neuron "sees" a local region of the input

Dataset: Synthetic 8×8 grayscale images with 10 distinct pattern classes.
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.random.seed(42)


# ─────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def softmax(z):
    e = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ─────────────────────────────────────────────
# A. CONVOLUTION LAYER
# ─────────────────────────────────────────────

class ConvLayer:
    """
    2D Convolutional Layer.

    Key idea: A small learnable filter (kernel) slides across the input image,
    computing dot products at each position. This exploits:
        1. Local Connectivity — each neuron only connects to a small region
        2. Parameter Sharing  — same filter weights used at all spatial positions
        3. Translation Invariance — detects features regardless of location

    Operation (single filter, single channel):
        output[i, j] = Σ_m Σ_n input[i+m, j+n] * kernel[m, n] + bias

    Parameters
    ----------
    n_filters  : int  — number of learnable filters (output channels)
    kernel_size: int  — height and width of each square filter
    n_channels : int  — number of input channels (1 for grayscale, 3 for RGB)
    stride     : int  — step size when sliding the filter
    padding    : int  — zero-padding added to input borders
    lr         : float— learning rate for this layer
    """

    def __init__(self, n_filters, kernel_size, n_channels=1,
                 stride=1, padding=0, lr=0.01):
        self.n_filters   = n_filters
        self.kernel_size = kernel_size
        self.n_channels  = n_channels
        self.stride      = stride
        self.padding     = padding
        self.lr          = lr

        # He initialization: shape (n_filters, n_channels, kH, kW)
        fan_in = n_channels * kernel_size * kernel_size
        self.W = np.random.randn(n_filters, n_channels,
                                 kernel_size, kernel_size) * np.sqrt(2.0 / fan_in)
        self.b = np.zeros(n_filters)

        self.cache = {}

    def _pad(self, X):
        """Zero-pad spatial dimensions."""
        if self.padding == 0:
            return X
        p = self.padding
        # X shape: (batch, channels, H, W)
        return np.pad(X, ((0,0),(0,0),(p,p),(p,p)), mode="constant")

    def forward(self, X):
        """
        Forward pass of convolution.
        X shape: (batch, channels, H, W)
        Output:  (batch, n_filters, H_out, W_out)
        where:   H_out = (H + 2P - K) / S + 1
        """
        batch, C, H, W = X.shape
        K = self.kernel_size
        S = self.stride
        P = self.padding
        H_out = (H + 2*P - K) // S + 1
        W_out = (W + 2*P - K) // S + 1

        X_pad = self._pad(X)
        out   = np.zeros((batch, self.n_filters, H_out, W_out))

        for f in range(self.n_filters):
            for i in range(H_out):
                for j in range(W_out):
                    h_s = i * S
                    w_s = j * S
                    # Extract local receptive field patch
                    patch = X_pad[:, :, h_s:h_s+K, w_s:w_s+K]  # (batch, C, K, K)
                    # Dot product with filter weights + bias
                    out[:, f, i, j] = (patch * self.W[f]).sum(axis=(1,2,3)) + self.b[f]

        self.cache = {"X": X, "X_pad": X_pad}
        return out

    def backward(self, d_out):
        """
        Backprop through convolution.
        Computes: dW (filter gradient), db (bias gradient), dX (input gradient).
        """
        X     = self.cache["X"]
        X_pad = self.cache["X_pad"]
        batch, C, H, W = X.shape
        K, S, P = self.kernel_size, self.stride, self.padding
        _, n_f, H_out, W_out = d_out.shape

        dW    = np.zeros_like(self.W)
        db    = np.zeros_like(self.b)
        dX_pad = np.zeros_like(X_pad)

        for f in range(n_f):
            db[f] = d_out[:, f, :, :].sum()
            for i in range(H_out):
                for j in range(W_out):
                    h_s = i * S
                    w_s = j * S
                    patch = X_pad[:, :, h_s:h_s+K, w_s:w_s+K]
                    # Gradient wrt filter: sum over batch
                    dW[f] += (patch * d_out[:, f, i, j]
                              .reshape(batch,1,1,1)).sum(axis=0)
                    # Gradient wrt input (accumulate)
                    dX_pad[:, :, h_s:h_s+K, w_s:w_s+K] += (
                        self.W[f] * d_out[:, f, i, j].reshape(batch,1,1,1))

        # Remove padding from dX
        if P > 0:
            dX = dX_pad[:, :, P:-P, P:-P]
        else:
            dX = dX_pad

        # Update parameters
        self.W -= self.lr * dW / batch
        self.b -= self.lr * db / batch
        return dX


# ─────────────────────────────────────────────
# B. MAX POOLING LAYER
# ─────────────────────────────────────────────

class MaxPoolLayer:
    """
    Max Pooling Layer.

    Reduces spatial dimensions by taking the maximum value in each
    non-overlapping pool_size × pool_size window.

    Benefits:
    - Reduces computation in deeper layers
    - Provides a degree of translation invariance
    - Prevents overfitting (reduces number of parameters)

    During backprop, gradient flows only through the position of the max
    (all other positions receive zero gradient — the "max mask").
    """

    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride    = stride
        self.cache     = {}

    def forward(self, X):
        """X shape: (batch, channels, H, W)"""
        batch, C, H, W = X.shape
        P, S = self.pool_size, self.stride
        H_out = (H - P) // S + 1
        W_out = (W - P) // S + 1

        out  = np.zeros((batch, C, H_out, W_out))
        mask = np.zeros_like(X, dtype=bool)

        for i in range(H_out):
            for j in range(W_out):
                h_s = i * S
                w_s = j * S
                patch = X[:, :, h_s:h_s+P, w_s:w_s+P]
                max_vals = patch.max(axis=(2, 3), keepdims=True)
                out[:, :, i, j] = max_vals[:, :, 0, 0]
                # Record which position had the max (for backprop)
                mask[:, :, h_s:h_s+P, w_s:w_s+P] |= (patch == max_vals)

        self.cache = {"X_shape": X.shape, "mask": mask}
        return out

    def backward(self, d_out):
        """Gradient flows only through max positions."""
        X_shape = self.cache["X_shape"]
        mask    = self.cache["mask"]
        dX      = np.zeros(X_shape)
        P, S    = self.pool_size, self.stride
        batch, C, H, W = X_shape
        _, _, H_out, W_out = d_out.shape

        for i in range(H_out):
            for j in range(W_out):
                h_s = i * S
                w_s = j * S
                dX[:, :, h_s:h_s+P, w_s:w_s+P] += (
                    mask[:, :, h_s:h_s+P, w_s:w_s+P]
                    * d_out[:, :, i, j][:, :, None, None])
        return dX


# ─────────────────────────────────────────────
# C. FULLY-CONNECTED LAYER
# ─────────────────────────────────────────────

class DenseLayer:
    """Standard fully-connected layer (same as DNN from Task 1)."""

    def __init__(self, n_in, n_out, activation="relu", lr=0.01):
        self.lr         = lr
        self.activation = activation
        scale = np.sqrt(2.0 / n_in) if activation == "relu" else np.sqrt(1.0 / n_in)
        self.W = np.random.randn(n_in, n_out) * scale
        self.b = np.zeros((1, n_out))
        self.cache = {}

    def forward(self, X):
        z = X @ self.W + self.b
        if self.activation == "relu":
            a = relu(z)
        elif self.activation == "softmax":
            a = softmax(z)
        else:
            a = z  # linear
        self.cache = {"X": X, "z": z}
        return a

    def backward(self, d_out):
        X, z = self.cache["X"], self.cache["z"]
        n    = X.shape[0]
        if self.activation == "relu":
            d_out = d_out * relu_deriv(z)
        dW = X.T @ d_out / n
        db = d_out.sum(axis=0, keepdims=True) / n
        dX = d_out @ self.W.T
        self.W -= self.lr * dW
        self.b -= self.lr * db
        return dX


# ─────────────────────────────────────────────
# D. CNN MODEL
# ─────────────────────────────────────────────

class CNN:
    """
    Complete CNN: Conv → ReLU → MaxPool → Flatten → Dense → Softmax

    Architecture (for 8×8 input):
        Input:  (batch, 1, 8, 8)
        Conv1:  8 filters, 3×3  → (batch, 8, 6, 6)
        ReLU
        Pool1:  2×2 MaxPool     → (batch, 8, 3, 3)
        Flatten:                → (batch, 72)
        Dense1: 72 → 64  + ReLU
        Dense2: 64 → 10  + Softmax
    """

    def __init__(self, n_classes=10, lr=0.01):
        self.conv1  = ConvLayer(n_filters=8, kernel_size=3, n_channels=1, lr=lr)
        self.pool1  = MaxPoolLayer(pool_size=2, stride=2)
        self.dense1 = DenseLayer(8 * 3 * 3, 64, activation="relu", lr=lr)
        self.dense2 = DenseLayer(64, n_classes, activation="softmax", lr=lr)
        self.loss_history = []

    def forward(self, X):
        """X: (batch, 1, 8, 8)"""
        # Conv + ReLU
        z1 = self.conv1.forward(X)     # (batch, 8, 6, 6)
        a1 = relu(z1)
        self._z1 = z1

        # Max Pool
        p1 = self.pool1.forward(a1)    # (batch, 8, 3, 3)

        # Flatten
        batch = p1.shape[0]
        flat  = p1.reshape(batch, -1)  # (batch, 72)
        self._pool_shape = p1.shape
        self._flat = flat

        # Fully-connected layers
        d1  = self.dense1.forward(flat)
        out = self.dense2.forward(d1)
        return out

    def backward(self, y_pred, y_true):
        # Gradient of softmax + cross-entropy
        delta = y_pred - y_true

        # Backprop through dense layers
        d_dense1 = self.dense2.backward(delta)
        d_flat   = self.dense1.backward(d_dense1)

        # Unflatten
        d_pool = d_flat.reshape(self._pool_shape)

        # Backprop through pool
        d_relu = self.pool1.backward(d_pool)

        # Backprop through ReLU
        d_conv = d_relu * relu_deriv(self._z1)

        # Backprop through conv
        self.conv1.backward(d_conv)

    def compute_loss(self, y_pred, y_true):
        n = y_true.shape[0]
        yc = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.sum(y_true * np.log(yc)) / n

    def train(self, X_tr, y_tr, X_v, y_v, epochs=80, batch_size=32):
        n = X_tr.shape[0]
        val_acc_hist = []
        print(f"\n{'─'*55}")
        print(f"  {'Epoch':>6}  {'Loss':>8}  {'Train Acc':>10}  {'Val Acc':>10}")
        print(f"{'─'*55}")
        for ep in range(1, epochs + 1):
            idx = np.random.permutation(n)
            X_sh, y_sh = X_tr[idx], y_tr[idx]
            ep_loss, nb = 0, 0
            for s in range(0, n, batch_size):
                Xb, yb = X_sh[s:s+batch_size], y_sh[s:s+batch_size]
                yp = self.forward(Xb)
                ep_loss += self.compute_loss(yp, yb)
                self.backward(yp, yb)
                nb += 1
            avg = ep_loss / max(1, nb)
            self.loss_history.append(avg)
            val_acc = self.accuracy(X_v, y_v)
            val_acc_hist.append(val_acc)
            if ep % 10 == 0 or ep == 1:
                tr_acc = self.accuracy(X_tr, y_tr)
                print(f"  {ep:>6}  {avg:>8.4f}  {tr_acc:>9.1f}%  {val_acc:>9.1f}%")
        print(f"{'─'*55}")
        return val_acc_hist

    def accuracy(self, X, y):
        yp = self.forward(X)
        return np.mean(np.argmax(yp, 1) == np.argmax(y, 1)) * 100


# ─────────────────────────────────────────────
# E. SYNTHETIC IMAGE DATASET
# ─────────────────────────────────────────────

def make_image_data(n_per_class=100, img_size=8, n_classes=10):
    """
    Generate synthetic 8×8 grayscale images where each class has a
    characteristic pattern (stripes, dots, edges, etc.).
    """
    X, y_list = [], []
    patterns = [
        lambda: np.tile(np.array([1,0,1,0,1,0,1,0]), (8,1)),             # horizontal stripes
        lambda: np.tile(np.array([1,0,1,0,1,0,1,0]), (8,1)).T,           # vertical stripes
        lambda: np.eye(8),                                                 # diagonal
        lambda: np.fliplr(np.eye(8)),                                      # anti-diagonal
        lambda: np.block([[np.ones((4,4)), np.zeros((4,4))],               # quadrant
                          [np.zeros((4,4)), np.ones((4,4))]]),
        lambda: np.block([[np.zeros((4,4)), np.ones((4,4))],               # inverse quadrant
                          [np.ones((4,4)), np.zeros((4,4))]]),
        lambda: (np.arange(8) < 4).reshape(8,1) * np.ones((1,8)),         # top half
        lambda: (np.arange(8) >= 4).reshape(8,1) * np.ones((1,8)),        # bottom half
        lambda: (np.add.outer(np.arange(8), np.arange(8)) % 2 == 0)       # checkerboard
                .astype(float),
        lambda: np.ones((8, 8)),                                            # all ones
    ]
    for c in range(n_classes):
        base = patterns[c]()
        for _ in range(n_per_class):
            noise = np.random.randn(img_size, img_size) * 0.2
            X.append((base + noise).clip(0, 1))
            y_list.append(c)

    X = np.array(X)[:, None, :, :]   # (N, 1, H, W)
    y = np.array(y_list)
    y_oh = np.zeros((len(y), n_classes)); y_oh[np.arange(len(y)), y] = 1

    idx = np.random.permutation(len(X))
    X, y_oh = X[idx], y_oh[idx]
    s = int(0.8 * len(X))
    return X[:s], y_oh[:s], X[s:], y_oh[s:]


# ─────────────────────────────────────────────
# F. VISUALIZATION
# ─────────────────────────────────────────────

def visualize(model, X_tr, y_tr, X_v, y_v, val_acc_hist):
    fig = plt.figure(figsize=(15, 11))
    fig.suptitle("Task 3: Convolutional Neural Networks for Image Data",
                 fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel 1: Training Loss ───────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(model.loss_history, color="#2196F3", linewidth=2, label="Train Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("CNN Training Loss"); ax1.legend(); ax1.grid(True, alpha=0.3)

    # ── Panel 2: Validation Accuracy ────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(val_acc_hist, color="#4CAF50", linewidth=2)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Val Accuracy (%)")
    ax2.set_title("Validation Accuracy"); ax2.grid(True, alpha=0.3)

    # ── Panel 3: Sample training images ─────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis("off")
    ax3.set_title("Sample Training Images (one per class)", fontweight="bold",
                  fontsize=11, pad=10)
    for c in range(10):
        idx = np.argmax(y_tr, axis=1) == c
        sample = X_tr[idx][0, 0]
        sub_ax = fig.add_axes([0.05 + c * 0.092, 0.38, 0.075, 0.13])
        sub_ax.imshow(sample, cmap="gray", vmin=0, vmax=1)
        sub_ax.set_title(f"Class {c}", fontsize=8)
        sub_ax.axis("off")

    # ── Panel 4: Learned Conv filters ───────────────────────────────────
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis("off")
    ax4.set_title("Learned Conv1 Filters (8 filters, 3×3)", fontweight="bold",
                  fontsize=11, pad=10)
    for f in range(8):
        filt = model.conv1.W[f, 0]
        sub_ax = fig.add_axes([0.05 + f * 0.116, 0.06, 0.09, 0.18])
        im = sub_ax.imshow(filt, cmap="RdBu", vmin=-filt.max(), vmax=filt.max())
        sub_ax.set_title(f"Filter {f+1}", fontsize=8)
        sub_ax.axis("off")

    plt.savefig("/mnt/user-data/outputs/task3_cnn_results.png",
                dpi=150, bbox_inches="tight")
    print("\n✔ Plot saved → task3_cnn_results.png")
    plt.close()


def print_cnn_architecture():
    """Print a text summary of the CNN architecture."""
    print("\n  CNN Architecture:")
    print("  " + "─"*50)
    rows = [
        ("Layer",        "Type",        "Output Shape",  "Params"),
        ("Input",        "—",           "(N,  1,  8,  8)", "0"),
        ("Conv1",        "Conv 3×3×8",  "(N,  8,  6,  6)", "80"),
        ("ReLU1",        "Activation",  "(N,  8,  6,  6)", "0"),
        ("MaxPool1",     "Pool 2×2",    "(N,  8,  3,  3)", "0"),
        ("Flatten",      "Reshape",     "(N,  72)",        "0"),
        ("Dense1",       "FC 72→64",    "(N,  64)",        "4,672"),
        ("ReLU2",        "Activation",  "(N,  64)",        "0"),
        ("Dense2",       "FC 64→10",    "(N,  10)",        "650"),
        ("Softmax",      "Activation",  "(N,  10)",        "0"),
    ]
    for row in rows:
        print(f"  {row[0]:<12} {row[1]:<16} {row[2]:<18} {row[3]:>8}")
    print("  " + "─"*50)
    print(f"  {'Total Trainable Parameters':.<44} {'5,402':>8}")
    print("  " + "─"*50)


# ─────────────────────────────────────────────
# G. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Convolutional Neural Networks (CNNs)")
    print("  Image Classification from Scratch with NumPy")
    print("=" * 60)

    print_cnn_architecture()

    # Generate dataset
    X_tr, y_tr, X_v, y_v = make_image_data(n_per_class=150)
    print(f"\n✔ Dataset ready:")
    print(f"   Train: {X_tr.shape}  |  Val: {X_v.shape}")
    print(f"   Image size: 8×8 grayscale  |  Classes: 10")

    # Build and train CNN
    cnn = CNN(n_classes=10, lr=0.01)
    val_acc_hist = cnn.train(X_tr, y_tr, X_v, y_v, epochs=80, batch_size=32)

    # Final evaluation
    final_tr  = cnn.accuracy(X_tr, y_tr)
    final_val = cnn.accuracy(X_v,  y_v)
    print(f"\n✔ Final Results:")
    print(f"   Train Accuracy : {final_tr:.2f}%")
    print(f"   Val   Accuracy : {final_val:.2f}%")

    # Visualize
    visualize(cnn, X_tr, y_tr, X_v, y_v, val_acc_hist)
    print("\n✔ Task 3 complete.\n")
