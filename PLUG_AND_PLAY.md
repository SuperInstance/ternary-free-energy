# PLUG_AND_PLAY — Free Energy

> Free energy principles for ternary agent systems

## 🚀 Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
ternary-free-energy = { git = "https://github.com/SuperInstance/ternary-free-energy" }
```

Use in your code:

```rust
use ternary_free_energy::FreeEnergyMinimizer;

let mut fe = FreeEnergyMinimizer::new(3);
fe.observe(&[1, 0, -1]);
let action = fe.act();
```

## 📚 Available Documentation

| Document | Description |
|----------|-------------|
| `docs/FROM_BINARY.md` | Understanding ternary concepts as a binary programmer |
| `docs/MIGRATION.md` | Version migration guide |
| `docs/FUTURE-INTEGRATION.md` | Planned features and roadmap |

## 🔗 Integration

This crate is part of the [SuperInstance ternary fleet](https://github.com/SuperInstance). It uses the canonical `Ternary` type from `ternary-types` for cross-crate compatibility.

## 📄 License

MIT
