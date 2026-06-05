# ternary-free-energy

**Free Energy Principle computations for ternary {-1, 0, +1} systems.**

## What problem does this solve?

Before you can build an agent that minimizes surprise, you need to be able to *measure* surprise — and the quantities that bound it. This crate provides the mathematical bedrock of the Free Energy Principle (FEP) for ternary-valued variables: Shannon entropy, Kullback-Leibler divergence, variational free energy, expected free energy, and surprise tracking. If your random variables take values in ℤ₃ = {-1, 0, +1} (ternary sensors, three-class classifiers, or spin-like states), this library gives you the information-theoretic tools to compute how far your beliefs are from reality, and how much information you expect to gain by observing the world.

Use this crate when you need correct, numerically stable information-theoretic primitives over a three-outcome alphabet — whether as a foundation for active inference, Bayesian filtering, or surprisal-based anomaly detection.

## Mathematical foundations

### Shannon Entropy

For a ternary distribution P over {-1, 0, +1}:

```
H(P) = - Σ_{x∈{-1,0,+1}} P(x) log₂ P(x)
```

Entropy is maximized for the uniform distribution:

```
H_max = log₂(3) ≈ 1.585 bits
```

and collapses to zero when P is deterministic. In the FEP, entropy quantifies uncertainty; minimizing expected entropy is equivalent to maximizing epistemic value.

### Kullback-Leibler Divergence

The relative entropy (KL divergence) measures the extra cost of coding samples from P using a code optimized for Q:

```
KL(P || Q) = Σ_{x∈{-1,0,+1}} P(x) ln(P(x) / Q(x))
```

Properties you should internalize:
- **Non-negativity**: KL(P||Q) ≥ 0, with equality iff P = Q almost everywhere.
- **Asymmetry**: KL(P||Q) ≠ KL(Q||P). In variational inference we typically minimize KL(Q||P) (the inclusive, zero-avoiding form).

### Variational Free Energy (VFE)

Given an approximate posterior q, a prior p, and a likelihood model P(o|s), the variational free energy is:

```
F = KL(q || p) - ⟨ln P(o|s)⟩_q
  = KL(q || p) - L
```

where L is the expected log-likelihood. F is an upper bound on surprise (-ln P(o)):

```
F ≥ -ln P(o)
```

Tightening the posterior reduces F and brings the agent's model closer to the true evidence.

### Expected Free Energy (EFE)

For a policy (sequence of actions) π, the expected free energy over future outcomes is:

```
G(π) = E_{q(o|π)} [ -H[q(s|o)] + KL[q(s|o) || q(s)] ]
     = -H[q(o|π)] + E_{q(o|π)}[ KL[q(s|o) || q(s)] ]
```

The first term is epistemic (reduce uncertainty); the second is pragmatic (stay close to prior preferences). This crate provides `expected_free_energy` for ternary state and observation distributions.

### Surprise

The self-information of an observation under a predictive distribution:

```
S(P, o) = -ln P(o)
```

High surprise means the model is poorly calibrated. The `SurpriseTracker` maintains a sliding window to detect whether the model is adapting (surprise trending downward).

## Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                      TernaryDist                              │
│         p_neg    p_zero    p_pos   (normalized)              │
│            \        |        /                               │
│             \       |       /                                │
│              ▼      ▼      ▼                                 │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐  │
│  │   entropy    │ │ kl_divergence│ │      surprise        │  │
│  │   H(P)       │ │  KL(P||Q)    │ │   -ln P(obs)         │  │
│  └──────────────┘ └──────────────┘ └──────────────────────┘  │
│              \       |       /                               │
│               \      |      /                                │
│                ▼     ▼     ▼                                 │
│  ┌────────────────────────────────────────────────────────┐  │
│  │           VariationalFreeEnergy                         │  │
│  │   F = KL(posterior || prior) - log_likelihood          │  │
│  └────────────────────────────────────────────────────────┘  │
│                          │                                   │
│                          ▼                                   │
│  ┌────────────────────────────────────────────────────────┐  │
│  │              SurpriseTracker                            │  │
│  │   sliding-window monitor; detects adaptation trends    │  │
│  └────────────────────────────────────────────────────────┘  │
│                          │                                   │
│                          ▼                                   │
│  ┌────────────────────────────────────────────────────────┐  │
│  │              MarkovBlanket                              │  │
│  │   parents / children / co-parents set utility          │  │
│  └────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Getting Started

```bash
cargo add ternary-free-energy
```

Minimal example: compute entropy, KL divergence, and variational free energy for a ternary distribution:

```rust
use ternary_free_energy::{TernaryDist, ternary_entropy, kl_divergence, VariationalFreeEnergy};

fn main() {
    let prior = TernaryDist::uniform();           // P(s)
    let posterior = TernaryDist::deterministic(1); // Q(s) = δ_{+1}
    let likelihood = TernaryDist::new(0.1, 0.1, 0.8); // P(o=+1|s)

    println!("H(prior) = {:.4} bits", ternary_entropy(&prior));
    println!("KL(post||prior) = {:.4}", kl_divergence(&posterior, &prior));

    let vfe = VariationalFreeEnergy::compute(&posterior, &prior, 1, &likelihood);
    println!("VFE = {:.4}", vfe.total);
    println!("  (KL term: {:.4}, log-likelihood: {:.4})", vfe.kl_term, vfe.log_likelihood);
}
```

Compile and run:

```bash
cargo run --example compute_vfe
```

## Running the Tests

```bash
cargo test
```

| Test | What it verifies |
|------|------------------|
| `uniform_entropy_is_max` | A uniform ternary distribution has entropy log₂(3) ≈ 1.585 bits, the theoretical maximum. |
| `deterministic_entropy_zero` | A Dirac delta distribution carries zero uncertainty. |
| `kl_same_distribution_zero` | KL divergence vanishes when the two distributions are identical. |
| `kl_asymmetric` | KL(P\|\|Q) ≠ KL(Q\|\|P), demonstrating the directional nature of relative entropy. |
| `vfe_positive` | Variational free energy is positive for a non-trivial posterior/likelihood pair. |
| `surprise_high_for_unexpected` | Observing an event with probability ≈0 yields very high surprise. |
| `surprise_zero_for_certain` | Observing a certain event yields zero surprise. |
| `surprise_tracker_adapting` | The sliding-window tracker detects a downward surprise trend as the model adapts. |
| `markov_blanket_set` | The Markov blanket correctly aggregates parents, children, and co-parents into a sorted set. |
| `bayesian_update_shifts_posterior` | Posterior ∝ Likelihood × Prior shifts mass toward the evidence-compatible outcome. |
| `distribution_mean` | The mean of a ternary distribution is `-p_neg + p_pos`. |
| `distribution_variance` | A deterministic distribution has zero variance. |
| `expected_free_energy_positive` | EFE returns a finite scalar for valid ternary input distributions. |

## Related Crates

- [`ternary-active-inference`](https://crates.io/crates/ternary-active-inference) — Full perception-action loop using the primitives defined here to select ternary actions.
- [`ternary-belief`](https://crates.io/crates/ternary-belief) — Belief propagation on factor graphs; compose with this crate to compute marginal entropies and KL divergences on structured models.
- [`ternary-entropy`](https://crates.io/crates/ternary-entropy) — Specialized entropy and mutual-information estimators for ternary time series.
- [`ternary-inference`](https://crates.io/crates/ternary-inference) — General inference algorithms (sampling, message passing) for ternary state spaces.

## License

MIT
