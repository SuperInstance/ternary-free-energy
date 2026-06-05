# ternary-free-energy

Free Energy Principle computations over ternary distributions `{-1, 0, +1}`.

## Components

- `TernaryEntropy` — Shannon entropy H(Q) for ternary distributions
- `VariationalFreeEnergy` — F = KL[Q||P] − E_Q[log P(o|s)]
- `SurpriseMinimization` — gradient descent on KL divergence
- `MarkovBlanket` — blanket partition for directed ternary networks
- `HierarchicalGenerativeModel` — multi-level free-energy accumulation

## Usage

```rust
use ternary_free_energy::{VariationalFreeEnergy, TernaryEntropy, uniform};

let q = [0.5, 0.3, 0.2];
let p = uniform();
let kl = VariationalFreeEnergy::kl_divergence(&q, &p);
let entropy = TernaryEntropy::compute(&q);
```
