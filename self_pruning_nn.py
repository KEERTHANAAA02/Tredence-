"""
Self-Pruning Neural Network on CIFAR-10
========================================
Implements a feed-forward network with learnable gate parameters that
drive weights to zero during training via L1 sparsity regularization.

Author: AI Engineer Case Study Solution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time

# ─────────────────────────────────────────────
# Part 1 – PrunableLinear Layer
# ─────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that multiplies each weight element
    by a learnable scalar gate in (0, 1).

    Forward pass:
        gates        = sigmoid(gate_scores)          # element-wise, same shape as weight
        pruned_w     = weight * gates                # gated weights
        output       = input @ pruned_w.T + bias     # standard affine transform

    Because all operations are differentiable, gradients flow into both
    `weight` and `gate_scores` automatically.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Standard weight + bias (same initialisation as nn.Linear)
        self.weight      = nn.Parameter(torch.empty(out_features, in_features))
        self.bias        = nn.Parameter(torch.zeros(out_features))

        # Gate score tensor – same shape as weight; initialised near 0 so that
        # sigmoid(gate_scores) ≈ 0.5, giving the network a neutral start.
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        # Kaiming uniform init for weights (mirrors nn.Linear default)
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates         = torch.sigmoid(self.gate_scores)       # ∈ (0, 1)
        pruned_weight = self.weight * gates                    # element-wise gate
        return F.linear(x, pruned_weight, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Return gate values (after sigmoid) as a detached flat tensor."""
        return torch.sigmoid(self.gate_scores).detach().cpu().flatten()

    def sparsity_fraction(self, threshold: float = 1e-2) -> float:
        """Fraction of gates below `threshold` (i.e., effectively pruned)."""
        gates = self.get_gates()
        return (gates < threshold).float().mean().item()


# ─────────────────────────────────────────────
# Feed-Forward Network built from PrunableLinear
# ─────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """
    Three-hidden-layer MLP for CIFAR-10 classification.
    All linear projections use PrunableLinear so that sparsity can be learned.

    Architecture:  3072 → 512 → 256 → 128 → 10
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            PrunableLinear(3 * 32 * 32, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            PrunableLinear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            PrunableLinear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            PrunableLinear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)          # flatten image
        return self.layers(x)

    def prunable_layers(self):
        """Yield every PrunableLinear sub-module."""
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                yield m

    # ── Part 2 – Sparsity Loss ────────────────────────────────────────────

    def sparsity_loss(self) -> torch.Tensor:
        """
        L1 norm of all gate values across every PrunableLinear layer.

        L1 on sigmoid outputs encourages exact zeros because:
          d/d(score) [ sigmoid(score) ] → 0 as score → -∞
        The gradient pushes gate_scores toward -∞ whenever the
        classification gradient doesn't resist, collapsing that gate to 0.
        """
        return sum(
            torch.sigmoid(layer.gate_scores).sum()
            for layer in self.prunable_layers()
        )

    def global_sparsity(self, threshold: float = 1e-2) -> float:
        """Percentage of gates below threshold across the whole network."""
        all_gates = torch.cat([layer.get_gates() for layer in self.prunable_layers()])
        return (all_gates < threshold).float().mean().item() * 100.0

    def all_gate_values(self) -> torch.Tensor:
        """Flat tensor of all gate values (for histogram)."""
        return torch.cat([layer.get_gates() for layer in self.prunable_layers()])


# ─────────────────────────────────────────────
# Part 3 – Data Loading
# ─────────────────────────────────────────────

def get_dataloaders(batch_size: int = 256):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.CIFAR10(root="./data", train=True,  download=True, transform=train_tf)
    test_ds  = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


# ─────────────────────────────────────────────
# Training & Evaluation Helpers
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, lam, device):
    model.train()
    total_loss = 0.0
    correct    = 0
    n          = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)

        cls_loss  = F.cross_entropy(logits, labels)
        spar_loss = model.sparsity_loss()
        loss      = cls_loss + lam * spar_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += imgs.size(0)

    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    n       = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        correct += (logits.argmax(1) == labels).sum().item()
        n       += imgs.size(0)
    return correct / n


def train_model(lam: float, epochs: int, device, train_loader, test_loader, verbose=True):
    """Full training run for a given lambda. Returns (test_acc, sparsity, model)."""
    model     = SelfPruningNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n{'='*60}")
    print(f"  Training with λ = {lam}   ({epochs} epochs)")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, lam, device)
        scheduler.step()
        sparsity = model.global_sparsity()

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"loss={tr_loss:.4f}  train_acc={tr_acc:.3f}  "
                  f"sparsity={sparsity:.1f}%  ({time.time()-t0:.1f}s)")

    test_acc = evaluate(model, test_loader, device)
    sparsity = model.global_sparsity()

    print(f"\n  ✓  Test accuracy : {test_acc*100:.2f}%")
    print(f"  ✓  Sparsity level: {sparsity:.2f}%")
    return test_acc, sparsity, model


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────

def plot_gate_distribution(model, lam: float, save_path="gate_distribution.png"):
    gates = model.all_gate_values().numpy()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    # Full histogram
    n_bins = 80
    counts, edges = np.histogram(gates, bins=n_bins, range=(0, 1))
    bar_colors = plt.cm.plasma(np.linspace(0.2, 0.9, n_bins))
    ax.bar(edges[:-1], counts, width=np.diff(edges), color=bar_colors,
           edgecolor="none", alpha=0.9)

    # Annotations
    frac_zero = (gates < 1e-2).mean() * 100
    ax.axvline(1e-2, color="#ff6b6b", linestyle="--", linewidth=1.5,
               label=f"Prune threshold (1e-2)\n{frac_zero:.1f}% pruned")

    ax.set_xlabel("Gate Value (sigmoid output)", color="white", fontsize=12)
    ax.set_ylabel("Count", color="white", fontsize=12)
    ax.set_title(f"Gate Value Distribution  |  λ = {lam}", color="white", fontsize=14, pad=12)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.legend(facecolor="#1a1a2e", edgecolor="#555", labelcolor="white")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved → {save_path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, test_loader = get_dataloaders(batch_size=256)

    EPOCHS  = 30
    LAMBDAS = [0.0, 1e-4, 5e-4, 1e-3]   # baseline + low / medium / high

    results = {}
    best_model, best_lam = None, None

    for lam in LAMBDAS:
        test_acc, sparsity, model = train_model(
            lam, EPOCHS, device, train_loader, test_loader
        )
        results[lam] = {"accuracy": test_acc * 100, "sparsity": sparsity}
        if best_model is None or lam == 5e-4:          # save "medium" as best
            best_model, best_lam = model, lam

    # ── Results table ──────────────────────────────────────────────────────
    print("\n\n" + "="*55)
    print("  λ (Lambda)   |  Test Accuracy  |  Sparsity Level")
    print("-"*55)
    for lam, res in results.items():
        tag = " (baseline)" if lam == 0.0 else ""
        print(f"  {lam:<12} |  {res['accuracy']:>10.2f}%  |  {res['sparsity']:>10.2f}%{tag}")
    print("="*55)

    # ── Gate distribution plot for best / most interesting model ──────────
    plot_gate_distribution(best_model, best_lam, save_path="gate_distribution.png")


if __name__ == "__main__":
    main()