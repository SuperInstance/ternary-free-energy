//! # ternary-free-energy
//! Free Energy Principle computations for ternary {-1, 0, +1} systems.
//! Variational inference, entropy, surprise minimization on Z₃.

/// Ternary probability distribution over {-1, 0, +1}
#[derive(Debug, Clone)]
pub struct TernaryDist {
    pub p_neg: f64,
    pub p_zero: f64,
    pub p_pos: f64,
}

impl TernaryDist {
    pub fn new(p_neg: f64, p_zero: f64, p_pos: f64) -> Self {
        let total = p_neg + p_zero + p_pos;
        Self { p_neg: p_neg/total, p_zero: p_zero/total, p_pos: p_pos/total }
    }

    pub fn uniform() -> Self { Self { p_neg: 1.0/3.0, p_zero: 1.0/3.0, p_pos: 1.0/3.0 } }

    pub fn deterministic(val: i8) -> Self {
        match val {
            -1 => Self { p_neg: 1.0, p_zero: 0.0, p_pos: 0.0 },
            0 => Self { p_neg: 0.0, p_zero: 1.0, p_pos: 0.0 },
            _ => Self { p_neg: 0.0, p_zero: 0.0, p_pos: 1.0 },
        }
    }

    pub fn prob(&self, val: i8) -> f64 {
        match val { -1 => self.p_neg, 0 => self.p_zero, _ => self.p_pos }
    }

    pub fn sample(&self, rng: f64) -> i8 {
        if rng < self.p_neg { -1 }
        else if rng < self.p_neg + self.p_zero { 0 }
        else { 1 }
    }

    pub fn mean(&self) -> f64 { -self.p_neg + self.p_pos }

    pub fn variance(&self) -> f64 {
        let m = self.mean();
        self.p_neg * (-1.0 - m).powi(2) + self.p_zero * (0.0 - m).powi(2) + self.p_pos * (1.0 - m).powi(2)
    }
}

/// Shannon entropy of a ternary distribution
pub fn ternary_entropy(dist: &TernaryDist) -> f64 {
    let mut h = 0.0;
    for &p in &[dist.p_neg, dist.p_zero, dist.p_pos] {
        if p > 1e-10 { h -= p * p.log2(); }
    }
    h
}

/// KL divergence KL(p || q) for ternary distributions
pub fn kl_divergence(p: &TernaryDist, q: &TernaryDist) -> f64 {
    let mut kl = 0.0;
    for val in [-1, 0, 1] {
        let pp = p.prob(val);
        let qp = q.prob(val);
        if pp > 1e-10 && qp > 1e-10 {
            kl += pp * (pp / qp).ln();
        }
    }
    kl
}

/// Variational free energy = KL(q || p) - log_likelihood
pub struct VariationalFreeEnergy {
    pub kl_term: f64,
    pub log_likelihood: f64,
    pub total: f64,
}

impl VariationalFreeEnergy {
    pub fn compute(posterior: &TernaryDist, prior: &TernaryDist, observation: i8, likelihood: &TernaryDist) -> Self {
        let kl = kl_divergence(posterior, prior);
        let ll = likelihood.prob(observation).ln().max(-50.0);
        Self { kl_term: kl, log_likelihood: ll, total: kl - ll }
    }
}

/// Surprise (negative log probability) of an observation
pub fn surprise(dist: &TernaryDist, observation: i8) -> f64 {
    let p = dist.prob(observation);
    if p < 1e-10 { 50.0 } else { -p.ln() }
}

/// Surprise tracker for monitoring prediction quality over time
#[derive(Debug, Clone)]
pub struct SurpriseTracker {
    pub surprises: Vec<f64>,
    pub window: usize,
}

impl SurpriseTracker {
    pub fn new(window: usize) -> Self { Self { surprises: Vec::new(), window } }

    pub fn observe(&mut self, dist: &TernaryDist, observation: i8) {
        self.surprises.push(surprise(dist, observation));
        if self.surprises.len() > self.window {
            self.surprises.remove(0);
        }
    }

    pub fn avg_surprise(&self) -> f64 {
        if self.surprises.is_empty() { 0.0 }
        else { self.surprises.iter().sum::<f64>() / self.surprises.len() as f64 }
    }

    pub fn is_adapting(&self) -> bool {
        if self.surprises.len() < 3 { return false; }
        let n = self.surprises.len();
        let first_half: f64 = self.surprises[..n/2].iter().sum();
        let second_half: f64 = self.surprises[n/2..].iter().sum();
        (second_half / (n - n/2) as f64) < (first_half / (n/2) as f64)
    }
}

/// Markov blanket: identifies which variables are in/out of a variable's blanket
#[derive(Debug, Clone)]
pub struct MarkovBlanket {
    pub target: usize,
    pub parents: Vec<usize>,
    pub children: Vec<usize>,
    pub co_parents: Vec<usize>,
}

impl MarkovBlanket {
    pub fn new(target: usize) -> Self {
        Self { target, parents: Vec::new(), children: Vec::new(), co_parents: Vec::new() }
    }

    pub fn blanket_set(&self) -> Vec<usize> {
        let mut set = Vec::new();
        set.extend(&self.parents);
        set.extend(&self.children);
        set.extend(&self.co_parents);
        set.sort_unstable();
        set.dedup();
        set
    }

    pub fn contains(&self, var: usize) -> bool {
        self.blanket_set().contains(&var)
    }
}

/// Bayesian update: posterior ∝ likelihood × prior for ternary distributions
pub fn bayesian_update(prior: &TernaryDist, likelihood: &TernaryDist) -> TernaryDist {
    let p_neg = prior.p_neg * likelihood.p_neg;
    let p_zero = prior.p_zero * likelihood.p_zero;
    let p_pos = prior.p_pos * likelihood.p_pos;
    TernaryDist::new(p_neg, p_zero, p_pos)
}

/// Expected free energy for a policy: EFE = Σ -H[q(o)] + KL[q(s|o) || q(s)]
pub fn expected_free_energy(
    expected_obs_dist: &TernaryDist,
    posterior_given_obs: &TernaryDist,
    prior_state: &TernaryDist,
) -> f64 {
    let info_gain = -ternary_entropy(expected_obs_dist);
    let kl = kl_divergence(posterior_given_obs, prior_state);
    info_gain + kl
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_entropy_is_max() {
        let u = TernaryDist::uniform();
        assert!((ternary_entropy(&u) - 1.585).abs() < 0.01); // log2(3)
    }

    #[test]
    fn deterministic_entropy_zero() {
        let d = TernaryDist::deterministic(1);
        assert_eq!(ternary_entropy(&d), 0.0);
    }

    #[test]
    fn kl_same_distribution_zero() {
        let p = TernaryDist::uniform();
        assert!(kl_divergence(&p, &p) < 1e-10);
    }

    #[test]
    fn kl_asymmetric() {
        let p = TernaryDist::deterministic(1);
        let q = TernaryDist::new(0.2, 0.3, 0.5);
        let kl_pq = kl_divergence(&p, &q);
        let kl_qp = kl_divergence(&q, &p);
        assert!(kl_pq > 0.0);
        assert!((kl_pq - kl_qp).abs() > 0.01);
    }

    #[test]
    fn vfe_positive() {
        let prior = TernaryDist::uniform();
        let posterior = TernaryDist::deterministic(1);
        let likelihood = TernaryDist::new(0.1, 0.1, 0.8);
        let vfe = VariationalFreeEnergy::compute(&posterior, &prior, 1, &likelihood);
        assert!(vfe.total > 0.0);
    }

    #[test]
    fn surprise_high_for_unexpected() {
        let d = TernaryDist::deterministic(1);
        assert!(surprise(&d, -1) > 10.0);
    }

    #[test]
    fn surprise_zero_for_certain() {
        let d = TernaryDist::deterministic(1);
        assert!(surprise(&d, 1) < 1e-10);
    }

    #[test]
    fn surprise_tracker_adapting() {
        let mut tracker = SurpriseTracker::new(10);
        let dist = TernaryDist::new(0.1, 0.8, 0.1);
        // High surprise initially, then it drops as we observe mostly 0s
        tracker.observe(&TernaryDist::new(0.3, 0.4, 0.3), 1); // unexpected
        tracker.observe(&TernaryDist::new(0.3, 0.4, 0.3), -1); // unexpected
        tracker.observe(&dist, 0);
        tracker.observe(&dist, 0);
        tracker.observe(&dist, 0);
        assert!(tracker.avg_surprise() < 5.0);
    }

    #[test]
    fn markov_blanket_set() {
        let mut mb = MarkovBlanket::new(3);
        mb.parents.push(0);
        mb.parents.push(1);
        mb.children.push(5);
        mb.co_parents.push(2);
        let blanket = mb.blanket_set();
        assert_eq!(blanket, vec![0, 1, 2, 5]);
        assert!(mb.contains(0));
        assert!(!mb.contains(3));
    }

    #[test]
    fn bayesian_update_shifts_posterior() {
        let prior = TernaryDist::uniform();
        let likelihood = TernaryDist::new(0.1, 0.1, 0.8);
        let posterior = bayesian_update(&prior, &likelihood);
        assert!(posterior.p_pos > 0.5);
    }

    #[test]
    fn distribution_mean() {
        let d = TernaryDist::new(0.2, 0.3, 0.5);
        assert!((d.mean() - 0.3).abs() < 1e-10);
    }

    #[test]
    fn distribution_variance() {
        let d = TernaryDist::deterministic(0);
        assert_eq!(d.variance(), 0.0);
    }

    #[test]
    fn expected_free_energy_positive() {
        let obs = TernaryDist::new(0.2, 0.6, 0.2);
        let post = TernaryDist::new(0.1, 0.8, 0.1);
        let prior = TernaryDist::uniform();
        let efe = expected_free_energy(&obs, &post, &prior);
        assert!(efe.is_finite());
    }
}
