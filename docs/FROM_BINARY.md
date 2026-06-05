# From Binary to Ternary: Free Energy Principle

## The Trap

The Free Energy Principle is mathematically agnostic about the number of states — it works for any probability distribution over any outcome space. But practical implementations almost always binary-encode the world: "is this sensor reading normal or anomalous?" "Should I approach or avoid?" The binary encoding forces every observation into a yes/no, every action into a binary choice.

Real agents don't perceive a binary world. A sensor might return "hot, cold, or lukewarm." A decision might be "pursue, wait, or abandon." The Free Energy Principle gives you the mathematical framework to compute surprise, information gain, and expected free energy — but if you flatten your state space to binary, you're losing half the signal before you start.

## Map to Three States

| Domain | −1 | 0 | +1 |
|--------|----|---|-----|
| Sensor reading | below threshold | at threshold | above threshold |
| Action selection | avoid | wait/observe | approach |
| Belief state | negative evidence | no evidence | positive evidence |
| Surprise level | impossible observation | unexpected (high entropy) | expected | 

## From Binary to Ternary

**Before: binary Shannon entropy**

```rust
// Two-state distribution
// H = -[p⋅log(p) + (1-p)⋅log(1-p)]
// Maximum entropy: 1.0 bit (when p = 0.5)
// The only uncertainty model: "either this or that"
```

**After: ternary Shannon entropy**

```rust
// Three-state distribution
// H = -[p(-1)⋅log(p(-1)) + p(0)⋅log(p(0)) + p(+1)⋅log(p(+1))]
// Maximum entropy: log₂(3) ≈ 1.585 bits
// Richer uncertainty: "either this or that or something else"
```

The extra 0.585 bits isn't a small difference — it's the difference between a binary decision and a ternary decision. A uniform ternary distribution has 58% more entropy than a uniform binary distribution. That extra uncertainty capacity is where the "I don't know yet" signal lives.

**Before: variational free energy with binary posterior**

```rust
// KL(posterior || prior) over {0, 1}
// If the posterior is uncertain, it lands at [0.5, 0.5]
// But [0.5, 0.5] says "both are equally likely" — not "I don't know"
```

**After: variational free energy with ternary posterior**

```rust
// KL(posterior || prior) over {-1, 0, +1}
// An uncertain posterior converges to [0, 1, 0]
// This says "I genuinely have no information" — not "both extremes are likely"
```

The distinction matters for active inference. An agent with binary beliefs that's uncertain about the state of the world thinks "it could be A or B" and picks the action that discriminates between them. An agent with ternary beliefs thinks "it could be A, B, or maybe neither" and picks an action that resolves the ambiguity — or decides that ambiguity is acceptable.

**0 is not nothing:** In the Free Energy Principle, the neutral belief state [0, 1, 0] has zero expected free energy — there's nothing to gain by resolving a belief that carries no information. The agent can conserve energy by not acting. In binary belief systems, the agent is always in a state of forced choice: [0.5, 0.5] drives exploration because both options seem equally plausible. Ternary beliefs allow the agent to recognize genuine ignorance and choose not to act.

```rust
// Binary: KL(q||p) forces a choice
q = [0.5, 0.5];  // uncertain, but both options are "live"

// Ternary: KL(q||p) allows abstention
q = [0.0, 1.0, 0.0];  // uncertain in a different way: "I have no model"
```

**The ternary conservation law** appears in expected free energy: the sum of pragmatic and epistemic values is bounded by the entropy of the predictive distribution. A ternary agent with high neutral mass has low expected free energy — it simply doesn't know enough to act yet. Binary agents never have this luxury.

## Why It Matters

Ternary belief states give agents the vocabulary to express genuine ignorance. The `0` state in a ternary distribution doesn't mean "50/50" — it means "no data." This changes active inference: agents can recognize when they don't know enough to act, instead of being forced into exploratory or exploitative behavior by binary uncertainty. The Free Energy Principle is richer in three states than two, and the math supports it seamlessly.
