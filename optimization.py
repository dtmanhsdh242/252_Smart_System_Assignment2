"""
=============================================================================
Optimization Techniques — Batch Normalization & Weight Initialization
Course: CO5119 - Intelligent Systems | Student: Đặng Tiến Mạnh - 2470569
=============================================================================

(A) WEIGHT INITIALIZATION STRATEGIES
    - Zero Initialization     → all neurons learn the same thing (symmetry problem)
    - Random Normal           → can cause vanishing/exploding gradients
    - Xavier / Glorot (Tanh)  → W ~ U[-√(6/(n_in+n_out)), +√(6/(n_in+n_out))]
    - He / Kaiming (ReLU)     → W ~ N(0, √(2/n_in))

(B) BATCH NORMALIZATION
    - Normalizes layer inputs to zero mean & unit variance per mini-batch
    - Learns affine parameters γ (scale) and β (shift)
    - Forward:  x̂ = (x − μ_B) / √(σ²_B + ε)
                y  = γ * x̂ + β
    - Reduces internal covariate shift, allows higher learning rates,
      and acts as a mild regularizer

Experiment: Trains 4 networks with different initialization strategies,
            then compares DNN without BN vs. with BN on the same task.
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ─────────────────────────────────────────────
# ACTIVATION FUNCTIONS
# ─────────────────────────────────────────────

def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)

def softmax(z):
    e = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ─────────────────────────────────────────────
# A. WEIGHT INITIALIZATION STRATEGIES
# ─────────────────────────────────────────────

class WeightInitializer:
    """Factory for different weight initialization strategies."""

    @staticmethod
    def zeros(n_in, n_out):
        """
        Zero Initialization — PROBLEMATIC
        All neurons output zero → all gradients identical → symmetry never broken.
        Every neuron in a layer learns the exact same feature. Never use in practice.
        """
        return np.zeros((n_in, n_out)), np.zeros((1, n_out))

    @staticmethod
    def random_normal(n_in, n_out, scale=0.01):
        """
        Small Random Normal — may cause vanishing gradients in deep networks.
        Scale=0.01 keeps weights small but gradients shrink exponentially with depth.
        """
        return np.random.randn(n_in, n_out) * scale, np.zeros((1, n_out))

    @staticmethod
    def xavier(n_in, n_out):
        """
        Xavier / Glorot Initialization (Glorot & Bengio, 2010)
        Designed for Tanh/Sigmoid activations.
        Uniform: W ~ U[-limit, +limit]  where limit = √(6 / (n_in + n_out))
        Ensures variance of activations remains roughly constant across layers.
        """
        limit = np.sqrt(6.0 / (n_in + n_out))
        return np.random.uniform(-limit, limit, (n_in, n_out)), np.zeros((1, n_out))

    @staticmethod
    def he(n_in, n_out):
        """
        He / Kaiming Initialization (He et al., 2015)
        Specifically designed for ReLU activations.
        W ~ N(0, √(2 / n_in))
        Accounts for ReLU zeroing half of activations by doubling the variance.
        """
        std = np.sqrt(2.0 / n_in)
        return np.random.randn(n_in, n_out) * std, np.zeros((1, n_out))


# ─────────────────────────────────────────────
# B. BATCH NORMALIZATION LAYER
# ─────────────────────────────────────────────

class BatchNormLayer:
    """
    Batch Normalization (Ioffe & Szegedy, 2015)

    Forward pass (training):
        μ_B  = (1/m) Σ x_i                    # batch mean
        σ²_B = (1/m) Σ (x_i - μ_B)²           # batch variance
        x̂_i  = (x_i - μ_B) / √(σ²_B + ε)     # normalize
        y_i  = γ * x̂_i + β                    # scale & shift (learnable)

    During inference uses running statistics (exponential moving average).

    Parameters
    ----------
    n_features : int    — number of neurons in this layer
    epsilon    : float  — small constant for numerical stability
    momentum   : float  — smoothing factor for running stats
    """

    def __init__(self, n_features, epsilon=1e-8, momentum=0.9):
        self.n_features = n_features
        self.epsilon    = epsilon
        self.momentum   = momentum

        # Learnable affine parameters
        self.gamma = np.ones((1, n_features))   # Scale (initially 1)
        self.beta  = np.zeros((1, n_features))  # Shift (initially 0)

        # Running statistics (used during inference)
        self.running_mean = np.zeros((1, n_features))
        self.running_var  = np.ones((1, n_features))

        # Cache for backpropagation
        self.cache = {}

    def forward(self, x, training=True):
        """Normalize, scale, and shift the input."""
        if training:
            mu    = x.mean(axis=0, keepdims=True)
            var   = x.var(axis=0, keepdims=True)
            x_hat = (x - mu) / np.sqrt(var + self.epsilon)
            out   = self.gamma * x_hat + self.beta

            # Update running statistics for inference
            self.running_mean = (self.momentum * self.running_mean
                                 + (1 - self.momentum) * mu)
            self.running_var  = (self.momentum * self.running_var
                                 + (1 - self.momentum) * var)
            # Cache intermediate values for backprop
            self.cache = {"x": x, "x_hat": x_hat, "mu": mu,
                          "var": var, "gamma": self.gamma}
        else:
            x_hat = ((x - self.running_mean)
                     / np.sqrt(self.running_var + self.epsilon))
            out = self.gamma * x_hat + self.beta

        return out

    def backward(self, d_out):
        """
        Backprop through BN layer.
        Returns gradient with respect to input, and updates dγ/dβ.
        """
        x, x_hat = self.cache["x"], self.cache["x_hat"]
        mu, var   = self.cache["mu"], self.cache["var"]
        m         = x.shape[0]
        std_inv   = 1.0 / np.sqrt(var + self.epsilon)

        # Gradients for learnable params
        d_gamma = np.sum(d_out * x_hat, axis=0, keepdims=True)
        d_beta  = np.sum(d_out,         axis=0, keepdims=True)

        # Gradient wrt input
        d_xhat  = d_out * self.gamma
        d_var   = np.sum(d_xhat * (x - mu) * (-0.5) * std_inv**3, axis=0, keepdims=True)
        d_mu    = (np.sum(d_xhat * (-std_inv), axis=0, keepdims=True)
                   + d_var * np.mean(-2.0 * (x - mu), axis=0, keepdims=True))
        d_x     = d_xhat * std_inv + d_var * 2 * (x - mu) / m + d_mu / m

        # Update params
        self.gamma -= 0.01 * d_gamma
        self.beta  -= 0.01 * d_beta

        return d_x


# ─────────────────────────────────────────────
# C. COMPARISON DNN (supports BN + init choice)
# ─────────────────────────────────────────────

class OptimizedDNN:
    """
    A DNN that can be configured with different initialization strategies
    and optionally with Batch Normalization.
    """

    def __init__(self, layer_dims, init_strategy="he",
                 use_batch_norm=False, lr=0.05):
        self.layer_dims    = layer_dims
        self.init_strategy = init_strategy
        self.use_bn        = use_batch_norm
        self.lr            = lr
        self.params        = {}
        self.bn_layers     = {}
        self.cache         = {}
        self.loss_history  = []

        self._init_params()
        if use_batch_norm:
            L = len(layer_dims) - 1
            for l in range(1, L):   # BN on hidden layers only
                self.bn_layers[l] = BatchNormLayer(layer_dims[l])

    def _init_params(self):
        init_fn = {
            "zeros":   WeightInitializer.zeros,
            "random":  WeightInitializer.random_normal,
            "xavier":  WeightInitializer.xavier,
            "he":      WeightInitializer.he,
        }[self.init_strategy]

        L = len(self.layer_dims)
        for l in range(1, L):
            W, b = init_fn(self.layer_dims[l-1], self.layer_dims[l])
            self.params[f"W{l}"] = W
            self.params[f"b{l}"] = b

    def forward(self, X, training=True):
        self.cache["a0"] = X
        L = len(self.layer_dims) - 1

        for l in range(1, L + 1):
            W      = self.params[f"W{l}"]
            b      = self.params[f"b{l}"]
            a_prev = self.cache[f"a{l-1}"]
            z      = a_prev @ W + b

            if l < L:
                if self.use_bn and l in self.bn_layers:
                    z = self.bn_layers[l].forward(z, training)
                a = relu(z)
            else:
                a = softmax(z)

            self.cache[f"z{l}"] = z
            self.cache[f"a{l}"] = a

        return self.cache[f"a{L}"]

    def compute_loss(self, y_pred, y_true):
        n = y_true.shape[0]
        y_c = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.sum(y_true * np.log(y_c)) / n

    def backward(self, y_true):
        n = y_true.shape[0]
        L = len(self.layer_dims) - 1
        grads = {}

        delta = self.cache[f"a{L}"] - y_true

        for l in range(L, 0, -1):
            a_prev = self.cache[f"a{l-1}"]
            W      = self.params[f"W{l}"]
            grads[f"dW{l}"] = (a_prev.T @ delta) / n
            grads[f"db{l}"] = delta.mean(axis=0, keepdims=True)
            if l > 1:
                delta = (delta @ W.T) * relu_deriv(self.cache[f"z{l-1}"])
                if self.use_bn and (l - 1) in self.bn_layers:
                    delta = self.bn_layers[l-1].backward(delta)

        for l in range(1, L + 1):
            self.params[f"W{l}"] -= self.lr * grads[f"dW{l}"]
            self.params[f"b{l}"] -= self.lr * grads[f"db{l}"]

    def train_epoch(self, X, y, batch_size=64):
        n = X.shape[0]
        idx = np.random.permutation(n)
        X, y = X[idx], y[idx]
        total_loss = 0
        for s in range(0, n, batch_size):
            Xb, yb = X[s:s+batch_size], y[s:s+batch_size]
            yp = self.forward(Xb, training=True)
            total_loss += self.compute_loss(yp, yb)
            self.backward(yb)
        avg = total_loss / max(1, n // batch_size)
        self.loss_history.append(avg)
        return avg

    def accuracy(self, X, y):
        yp = self.forward(X, training=False)
        return np.mean(np.argmax(yp, 1) == np.argmax(y, 1)) * 100


# ─────────────────────────────────────────────
# D. DATASET
# ─────────────────────────────────────────────

def make_data(n=2000, d=20, c=10):
    X, y = [], []
    for k in range(c):
        mean = np.random.randn(d) * 3
        X.append(np.random.randn(n // c, d) + mean)
        y.append(np.full(n // c, k))
    X = np.vstack(X)
    y = np.concatenate(y)
    yh = np.zeros((len(y), c)); yh[np.arange(len(y)), y.astype(int)] = 1
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    idx = np.random.permutation(len(X))
    X, yh = X[idx], yh[idx]
    s = int(0.8 * len(X))
    return X[:s], yh[:s], X[s:], yh[s:]


# ─────────────────────────────────────────────
# E. EXPERIMENTS & PLOTTING
# ─────────────────────────────────────────────

def run_experiments():
    X_tr, y_tr, X_v, y_v = make_data()
    EPOCHS = 150
    DIMS   = [20, 128, 64, 10]

    # ── Experiment 1: Weight Initialization Comparison ──────────────────────
    print("=" * 60)
    print("  Experiment 1: Weight Initialization Strategies")
    print("=" * 60)

    init_strategies = {
        "Zero Init\n(broken)":    "zeros",
        "Random Normal\n(naive)": "random",
        "Xavier/Glorot\n(Tanh)":  "xavier",
        "He/Kaiming\n(ReLU)":     "he",
    }

    results_init = {}
    for label, strategy in init_strategies.items():
        print(f"\n  ► {label.replace(chr(10),' ')}")
        model = OptimizedDNN(DIMS, init_strategy=strategy, lr=0.05)
        for ep in range(EPOCHS):
            model.train_epoch(X_tr, y_tr)
        acc = model.accuracy(X_v, y_v)
        print(f"     Val Accuracy: {acc:.1f}%")
        results_init[label] = {
            "loss": model.loss_history,
            "acc":  acc
        }

    # ── Experiment 2: With vs Without Batch Normalization ───────────────────
    print("\n" + "=" * 60)
    print("  Experiment 2: With vs. Without Batch Normalization")
    print("=" * 60)

    bn_results = {}
    for use_bn in [False, True]:
        tag = "With BN" if use_bn else "Without BN"
        print(f"\n  ► {tag}")
        model = OptimizedDNN(DIMS, init_strategy="he",
                             use_batch_norm=use_bn, lr=0.05)
        accs = []
        for ep in range(EPOCHS):
            model.train_epoch(X_tr, y_tr)
            if (ep + 1) % 10 == 0:
                accs.append(model.accuracy(X_v, y_v))
        final_acc = model.accuracy(X_v, y_v)
        print(f"     Val Accuracy: {final_acc:.1f}%")
        bn_results[tag] = {
            "loss": model.loss_history,
            "acc_curve": accs,
            "final_acc": final_acc,
        }

    # ── Plot all results ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Task 2: Optimization Techniques\nBatch Normalization & Weight Initialization",
                 fontsize=13, fontweight="bold")

    colors = ["#e74c3c", "#e67e22", "#3498db", "#2ecc71"]

    # Plot 1: Loss curves per init strategy
    ax = axes[0, 0]
    for (label, data), color in zip(results_init.items(), colors):
        clean = label.replace("\n", " ")
        ax.plot(data["loss"], label=clean, color=color, linewidth=1.8)
    ax.set_title("Loss by Initialization Strategy", fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Plot 2: Final accuracy bar chart per init strategy
    ax = axes[0, 1]
    labels = [k.replace("\n", "\n") for k in results_init.keys()]
    accs   = [v["acc"] for v in results_init.values()]
    bars = ax.bar(range(len(labels)), accs, color=colors, edgecolor="white",
                  linewidth=1.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_title("Final Val Accuracy by Init Strategy", fontweight="bold")
    ax.set_ylabel("Accuracy (%)"); ax.grid(True, alpha=0.3, axis="y")

    # Plot 3: Loss curves With/Without BN
    ax = axes[1, 0]
    for (tag, data), color in zip(bn_results.items(), ["#e74c3c", "#2ecc71"]):
        ax.plot(data["loss"], label=tag, color=color, linewidth=2)
    ax.set_title("Training Loss: BN vs No BN", fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Plot 4: BN theory diagram
    ax = axes[1, 1]
    ax.axis("off")
    ax.set_title("Batch Normalization — Forward Pass", fontweight="bold")
    steps = [
        ("Input batch  x ∈ ℝ^(m×d)", "#AED6F1"),
        ("μ_B = (1/m) Σ xᵢ", "#A9DFBF"),
        ("σ²_B = (1/m) Σ (xᵢ − μ_B)²", "#A9DFBF"),
        ("x̂ᵢ = (xᵢ − μ_B) / √(σ²_B + ε)", "#FAD7A0"),
        ("yᵢ = γ · x̂ᵢ + β   (learnable γ, β)", "#F1948A"),
    ]
    for i, (text, color) in enumerate(steps):
        y_pos = 0.85 - i * 0.17
        rect = plt.Rectangle((0.05, y_pos - 0.06), 0.9, 0.13,
                              color=color, ec="gray", lw=1, zorder=2,
                              transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(0.5, y_pos, text, ha="center", va="center",
                fontsize=9, transform=ax.transAxes, zorder=3)
        if i < len(steps) - 1:
            ax.annotate("", xy=(0.5, y_pos - 0.06), xytext=(0.5, y_pos - 0.04),
                        xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", color="gray"))

    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/task2_optimization_results.png",
                dpi=150, bbox_inches="tight")
    print("\n✔ Plot saved → task2_optimization_results.png")
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("  TASK 2: Optimization Techniques")
    print("  Batch Normalization & Weight Initialization")
    print("=" * 60)
    run_experiments()
    print("\n✔ Task 2 complete.\n")
