# ternary-free-energy

**Free Energy Principle computations for ternary {-1, 0, +1} systems.**

The Free Energy Principle (Friston, 2006) says: an agent should minimize its variational free energy to stay in a narrow band of states compatible with survival. This crate provides the mathematical primitives for computing free energy, entropy, surprise, and Bayesian updates on ternary distributions.

---

## Key Equations

**Variational Free Energy:**
```
F = KL[q(s) || p(s|o)] - ln p(o)
  = complexity - accuracy
```

**Ternary Entropy** (max = log₂3 ≈ 1.585 bits):
```
H(p) = -Σ pᵢ log₂(pᵢ)  for i ∈ {-1, 0, +1}
```

**KL Divergence** (closed form for ternary):
```
KL(p || q) = Σ pᵢ ln(pᵢ/qᵢ)  for i ∈ {-1, 0, +1}
```

**Why ternary carries 58.5% more information than binary:**
- Binary: max entropy = 1 bit
- Ternary: max entropy = log₂(3) ≈ 1.585 bits

Every ternary symbol carries more information. This is free — no extra computation, just a richer alphabet.

---

## Architecture

- **`TernaryDist`** — Probability distribution over {-1, 0, +1} with normalization, mean, variance
- **`ternary_entropy()`** — Shannon entropy (max 1.585 bits)
- **`kl_divergence(p, q)`** — KL(p||q) for ternary distributions
- **`VariationalFreeEnergy`** — KL term + log-likelihood = total free energy
- **`surprise()`** — Negative log probability of observation
- **`SurpriseTracker`** — Running average with adaptation detection
- **`MarkovBlanket`** — Parents/children/co-parents identification
- **`bayesian_update()`** — posterior ∝ likelihood × prior
- **`expected_free_energy()`** — EFE for active inference policy selection

---

## Quick Start

```rust
use ternary_free_energy::{TernaryDist, ternary_entropy, kl_divergence, bayesian_update};

let prior = TernaryDist::uniform(); // [1/3, 1/3, 1/3]
let likelihood = TernaryDist::new(0.1, 0.2, 0.7); // biased toward +1

let posterior = bayesian_update(&prior, &likelihood);
println!("P(+1) after update: {:.3}", posterior.p_pos);

println!("Entropy of uniform: {:.3} bits", ternary_entropy(&TernaryDist::uniform()));
println!("KL divergence: {:.3}", kl_divergence(&posterior, &prior));
```

---

## Ecosystem

- **ternary-active-inference** — Full active inference agent using these primitives
- **ternary-belief** — Belief propagation for inference
- **ternary-entropy** — Entropy measures for ternary systems
- **ternary-bayes** — Bayesian inference on ternary distributions

## License

MIT
