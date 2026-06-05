//! Free Energy Principle computations over ternary distributions {-1, 0, +1}.

pub type TernaryVal = i8;
pub const TERNARY_VALS: [TernaryVal; 3] = [-1, 0, 1];

/// Probability distribution over ternary values {-1, 0, +1}.
pub type TernaryDist = [f64; 3];

pub fn uniform() -> TernaryDist {
    [1.0 / 3.0; 3]
}

pub fn deterministic(v: TernaryVal) -> TernaryDist {
    let mut d = [0.0; 3];
    d[(v + 1) as usize] = 1.0;
    d
}

fn normalize_dist(d: &mut TernaryDist) {
    let sum: f64 = d.iter().sum();
    if sum > 1e-12 {
        for x in d.iter_mut() {
            *x /= sum;
        }
    }
}

/// Shannon entropy H(Q) for a ternary distribution.
pub struct TernaryEntropy;

impl TernaryEntropy {
    pub fn compute(dist: &TernaryDist) -> f64 {
        dist.iter()
            .filter(|&&p| p > 1e-12)
            .map(|&p| -p * p.ln())
            .sum()
    }

    /// Maximum entropy = ln(3) ≈ 1.0986.
    pub fn max_entropy() -> f64 {
        3.0_f64.ln()
    }

    /// Normalised entropy ∈ [0,1].
    pub fn normalised(dist: &TernaryDist) -> f64 {
        Self::compute(dist) / Self::max_entropy()
    }
}

/// Variational Free Energy F = KL[Q||P] – E_Q[log P(o|s)].
pub struct VariationalFreeEnergy;

impl VariationalFreeEnergy {
    /// KL[Q||P] = Σ Q(s) log(Q(s)/P(s)).
    pub fn kl_divergence(q: &TernaryDist, p: &TernaryDist) -> f64 {
        q.iter()
            .zip(p.iter())
            .filter(|(&qi, &pi)| qi > 1e-12 && pi > 1e-12)
            .map(|(&qi, &pi)| qi * (qi / pi).ln())
            .sum()
    }

    /// F = KL[Q||P] – expected_log_likelihood.
    pub fn compute(q: &TernaryDist, p: &TernaryDist, expected_log_likelihood: f64) -> f64 {
        Self::kl_divergence(q, p) - expected_log_likelihood
    }

    /// Approximate –log P(o) upper bound via KL.
    pub fn surprise_bound(q: &TernaryDist, p: &TernaryDist) -> f64 {
        Self::kl_divergence(q, p)
    }

    /// Expected log-likelihood E_Q[log P(o|s)] given likelihood table.
    pub fn expected_log_likelihood(q: &TernaryDist, log_likelihood: &TernaryDist) -> f64 {
        q.iter().zip(log_likelihood.iter()).map(|(&qi, &ll)| qi * ll).sum()
    }
}

/// Gradient-descent step on Q to minimise surprise.
pub struct SurpriseMinimization {
    pub learning_rate: f64,
}

impl SurpriseMinimization {
    pub fn new(learning_rate: f64) -> Self {
        Self { learning_rate }
    }

    /// Move Q toward P by one gradient step then renormalise.
    pub fn gradient_step(&self, q: &mut TernaryDist, p: &TernaryDist) {
        for i in 0..3 {
            q[i] += self.learning_rate * (p[i] - q[i]);
        }
        normalize_dist(q);
    }

    /// Run until KL < tol or max_steps reached; returns (steps_taken, converged).
    pub fn run_until_converged(
        &self,
        q: &mut TernaryDist,
        p: &TernaryDist,
        tol: f64,
        max_steps: usize,
    ) -> (usize, bool) {
        for step in 0..max_steps {
            if VariationalFreeEnergy::kl_divergence(q, p) < tol {
                return (step, true);
            }
            self.gradient_step(q, p);
        }
        (max_steps, VariationalFreeEnergy::kl_divergence(q, p) < tol)
    }

    pub fn converged(q: &TernaryDist, p: &TernaryDist, tol: f64) -> bool {
        VariationalFreeEnergy::kl_divergence(q, p) < tol
    }
}

/// Markov Blanket partition for a node in a directed ternary network.
#[derive(Debug, Clone, PartialEq)]
pub struct MarkovBlanket {
    pub target: usize,
    pub internal: Vec<usize>,
    pub blanket: Vec<usize>,
    pub external: Vec<usize>,
}

impl MarkovBlanket {
    /// Compute blanket = parents(target) ∪ children(target) ∪ co-parents(target).
    pub fn compute(
        n_nodes: usize,
        adjacency: &[Vec<bool>],
        target: usize,
    ) -> Self {
        use std::collections::BTreeSet;

        let mut blanket = BTreeSet::new();

        // parents of target
        for i in 0..n_nodes {
            if i != target && adjacency[i][target] {
                blanket.insert(i);
            }
        }

        // children of target + co-parents
        let children: Vec<usize> = (0..n_nodes)
            .filter(|&j| j != target && adjacency[target][j])
            .collect();
        for &c in &children {
            blanket.insert(c);
            for i in 0..n_nodes {
                if i != target && adjacency[i][c] {
                    blanket.insert(i);
                }
            }
        }
        blanket.remove(&target);

        let blanket_vec: Vec<usize> = blanket.iter().cloned().collect();
        let external: Vec<usize> = (0..n_nodes)
            .filter(|&i| i != target && !blanket_vec.contains(&i))
            .collect();

        Self {
            target,
            internal: vec![target],
            blanket: blanket_vec,
            external,
        }
    }

    /// True if node is in the blanket.
    pub fn in_blanket(&self, node: usize) -> bool {
        self.blanket.contains(&node)
    }

    /// Blanket size.
    pub fn size(&self) -> usize {
        self.blanket.len()
    }

    /// The blanket + internal partition is exhaustive over all nodes.
    pub fn covers_all(&self, n_nodes: usize) -> bool {
        let mut seen = vec![false; n_nodes];
        for &i in &self.internal {
            seen[i] = true;
        }
        for &i in &self.blanket {
            seen[i] = true;
        }
        for &i in &self.external {
            seen[i] = true;
        }
        seen.iter().all(|&s| s)
    }
}

/// Multi-level generative model with precision-weighted free energy.
pub struct HierarchicalGenerativeModel {
    pub levels: Vec<TernaryDist>,
    pub precision: Vec<f64>,
}

impl HierarchicalGenerativeModel {
    pub fn new(n_levels: usize) -> Self {
        Self {
            levels: vec![uniform(); n_levels],
            precision: vec![1.0; n_levels],
        }
    }

    /// Free energy contribution at level l = precision[l] * KL[Q_l || Q_{l-1}].
    pub fn free_energy_at_level(&self, level: usize) -> f64 {
        if level == 0 {
            return 0.0;
        }
        let kl = VariationalFreeEnergy::kl_divergence(
            &self.levels[level],
            &self.levels[level - 1],
        );
        self.precision[level] * kl
    }

    pub fn total_free_energy(&self) -> f64 {
        (1..self.levels.len())
            .map(|l| self.free_energy_at_level(l))
            .sum()
    }

    /// Gradient step: move level toward observation blend.
    pub fn update_level(&mut self, level: usize, obs_dist: &TernaryDist, lr: f64) {
        if level >= self.levels.len() {
            return;
        }
        let q = &mut self.levels[level];
        for i in 0..3 {
            q[i] += lr * (obs_dist[i] - q[i]);
        }
        normalize_dist(q);
    }

    /// Set the prior (level 0) belief.
    pub fn set_prior(&mut self, prior: TernaryDist) {
        if !self.levels.is_empty() {
            self.levels[0] = prior;
        }
    }

    pub fn n_levels(&self) -> usize {
        self.levels.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entropy_uniform_equals_ln3() {
        let u = uniform();
        let h = TernaryEntropy::compute(&u);
        assert!((h - 3.0_f64.ln()).abs() < 1e-10, "h={h}");
    }

    #[test]
    fn test_entropy_deterministic_is_zero() {
        let d = deterministic(1);
        let h = TernaryEntropy::compute(&d);
        assert!(h.abs() < 1e-10, "h={h}");
    }

    #[test]
    fn test_entropy_partial_is_between_zero_and_ln3() {
        let d: TernaryDist = [0.5, 0.3, 0.2];
        let h = TernaryEntropy::compute(&d);
        assert!(h > 0.0 && h < 3.0_f64.ln());
    }

    #[test]
    fn test_normalised_entropy_in_unit_interval() {
        let d: TernaryDist = [0.6, 0.3, 0.1];
        let hn = TernaryEntropy::normalised(&d);
        assert!(hn >= 0.0 && hn <= 1.0 + 1e-10);
    }

    #[test]
    fn test_kl_identical_is_zero() {
        let p: TernaryDist = [0.4, 0.35, 0.25];
        let kl = VariationalFreeEnergy::kl_divergence(&p, &p);
        assert!(kl.abs() < 1e-10, "kl={kl}");
    }

    #[test]
    fn test_kl_nonneg() {
        let q: TernaryDist = [0.6, 0.2, 0.2];
        let p: TernaryDist = [0.2, 0.5, 0.3];
        assert!(VariationalFreeEnergy::kl_divergence(&q, &p) >= 0.0);
    }

    #[test]
    fn test_vfe_compute_returns_finite() {
        let q: TernaryDist = [0.5, 0.3, 0.2];
        let p: TernaryDist = [0.33, 0.33, 0.34];
        let ell = -0.5;
        let f = VariationalFreeEnergy::compute(&q, &p, ell);
        assert!(f.is_finite());
    }

    #[test]
    fn test_surprise_minimization_converges() {
        let sm = SurpriseMinimization::new(0.2);
        let p: TernaryDist = [0.7, 0.2, 0.1];
        let mut q = uniform();
        let (_, converged) = sm.run_until_converged(&mut q, &p, 1e-4, 500);
        assert!(converged, "should converge in 500 steps");
    }

    #[test]
    fn test_surprise_minimization_reduces_kl() {
        let sm = SurpriseMinimization::new(0.1);
        let p: TernaryDist = [0.7, 0.2, 0.1];
        let mut q = uniform();
        let kl0 = VariationalFreeEnergy::kl_divergence(&q, &p);
        for _ in 0..20 {
            sm.gradient_step(&mut q, &p);
        }
        let kl1 = VariationalFreeEnergy::kl_divergence(&q, &p);
        assert!(kl1 < kl0);
    }

    #[test]
    fn test_markov_blanket_covers_all_nodes() {
        // simple 5-node chain: 0→1→2→3→4
        let n = 5;
        let mut adj = vec![vec![false; n]; n];
        for i in 0..4 {
            adj[i][i + 1] = true;
        }
        let mb = MarkovBlanket::compute(n, &adj, 2);
        assert!(mb.covers_all(n), "blanket+internal+external should cover all nodes");
    }

    #[test]
    fn test_markov_blanket_isolates_target() {
        let n = 5;
        let mut adj = vec![vec![false; n]; n];
        adj[0][2] = true; // parent
        adj[1][2] = true; // parent
        adj[2][3] = true; // child
        adj[2][4] = true; // child
        let mb = MarkovBlanket::compute(n, &adj, 2);
        // blanket should contain parents + children
        assert!(mb.in_blanket(0));
        assert!(mb.in_blanket(1));
        assert!(mb.in_blanket(3));
        assert!(mb.in_blanket(4));
    }

    #[test]
    fn test_hierarchical_model_total_free_energy_uniform() {
        let hm = HierarchicalGenerativeModel::new(3);
        // All levels uniform → KL=0
        let tfe = hm.total_free_energy();
        assert!(tfe.abs() < 1e-10, "tfe={tfe}");
    }

    #[test]
    fn test_hierarchical_model_update_normalizes() {
        let mut hm = HierarchicalGenerativeModel::new(3);
        let obs: TernaryDist = [0.8, 0.1, 0.1];
        hm.update_level(1, &obs, 0.5);
        let sum: f64 = hm.levels[1].iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }
}
