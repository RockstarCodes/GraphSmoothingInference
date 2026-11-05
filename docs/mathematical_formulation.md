# Mathematical formulation

## Variables
For node i ∈ V (|V| = n = 2,000): Y_i ∈ {0,1} benchmark label; A_i ∈ {0,1} recorded gender (R_i recorded region); X_i ∈ ℝ^275 remaining released columns; H_i ∈ ℝ^d learned embedding (Node2Vec or FairWalk — a learned representation, not a latent cause); E the observed edge set, E_F ⊆ E a prespecified forest.
Observed: E, X, A, R, Y_T for training nodes T. Latent at inference: Y_V∖T (validation, test and, under reduced label availability, dropped training nodes).

## Conditional Markov network
$$
P(\mathbf Y \mid \mathbf H, \mathbf X, E_F) = \frac{1}{Z(\mathbf H,\mathbf X)} \prod_{i\in V}\psi_i(Y_i;H_i,X_i)\prod_{(i,j)\in E_F}\psi_{ij}(Y_i,Y_j),
\qquad Z=\sum_{\mathbf y\in\{0,1\}^n}\prod_i\psi_i(y_i)\prod_{(i,j)\in E_F}\psi_{ij}(y_i,y_j).
$$
* Unary: ψ_i(1) = p_i, ψ_i(0) = 1 − p_i with p_i = σ(w^⊤ standardise(H_i) + b) from L2 logistic regression fitted on T (clipped to [10⁻⁶, 1−10⁻⁶]). In the main pipeline the unary uses H only; the feature-only and SGC baselines use X.
* Pairwise (attractive Potts / Ising): ψ_ij(y_i, y_j) = exp(λ_ij·1[y_i = y_j]), λ_ij ≥ 0.
* Evidence: for i ∈ T multiply by δ(Y_i = y_i^obs). The query is the posterior marginal q_i = P(Y_i = 1 | y_T, H, X, E_F) for i ∉ T. Decision D_i = 1[q_i ≥ τ], τ chosen on validation.

Because all potentials are strictly positive (after clipping), the distribution is positive and the Hammersley–Clifford equivalence between the factorisation and the local/global Markov properties holds on the forest.

## Markov blanket
In a pairwise MRF the Markov blanket of Y_i is its set of neighbours N_F(i) in E_F:
$$
P(Y_i=1\mid \mathbf y_{-i}) = \frac{p_i\,e^{\sum_{j\in N_F(i)}\lambda_{ij}y_j}}{p_i\,e^{\sum_j\lambda_{ij}y_j}+(1-p_i)\,e^{\sum_j\lambda_{ij}(1-y_j)}}.
$$
So the log-odds shift equals Σ_j λ_ij(2y_j − 1): each neighbour labelled 1 adds λ_ij, each labelled 0 subtracts λ_ij. If neighbours of group-1 nodes are disproportionately group 1 (demographic homophily) and group 1 has a higher base rate, observed neighbour labels shift group-1 marginals up and group-0 marginals down — the mechanism under test. Implemented in `PairwiseCMN.conditional_given_blanket`, tested against enumeration, and demonstrated on the real forest in `results/metrics/inference_verification_summary.json`.

## Exact sum-product on a forest
Root each tree component at its smallest node id; with log node potential ℓ_u(y) = log ψ_u(y) + log δ_evidence:
$$
\log m_{u\to v}(y_v)=\operatorname{logsumexp}_{y_u}\Big[\ell_u(y_u)+\lambda_{uv}1[y_u=y_v]+\sum_{w\in N(u)\setminus v}\log m_{w\to u}(y_u)\Big].
$$
Upward pass leaves→root, downward pass root→leaves; belief b_u ∝ exp(ℓ_u + Σ_w log m_{w→u}); log Z = Σ_components logsumexp(b_root). Exact on forests; O(|V| + |E_F|). Isolated nodes return their (evidence-reduced) normalised unary.

## Forest projection
Maximum spanning forest of E under w_ij = Jaccard(N[i], N[j]) of closed neighbourhoods (Kruskal; ties by (min, max) node id). Label-free, embedding-free, deterministic, identical across mitigation variants. Robustness: BFS forest, low-degree forest (w = −(deg_i + deg_j)), random-weight forest (seed 0). Loopy BP on the full graph is an optional exploratory reference only.

## Proposed attenuation (HCA) and comparators
$$
\lambda_{ij}= \lambda\,a_{ij},\quad a_{ij}=1-\eta\,1[A_i=A_j],\quad \eta\in[0,1].
$$
* λ: $\hat\lambda=\arg\min_{\lambda\in\Lambda}\frac1K\sum_k \mathrm{LL}\big(y_{T_k},\,q^{(\lambda)}_{T_k}\mid y_{T\setminus T_k}\big)$ with out-of-fold unaries on T (training folds only). This is the predictive term of the objective L = L_pred + α L_fair; it is fitted first because it has a probabilistic meaning (held-out marginal log-likelihood).
* η: the fairness term is the nondifferentiable validation ΔEOpp. Rather than a weighted sum with an arbitrary α, we use the constrained form $\hat\eta=\arg\min_\eta \Delta\mathrm{EOpp}_{val}(\eta)$ s.t. $\mathrm{F1}_{val}(\eta)\ge \mathrm{F1}_{val}(\text{uniform})-0.02$ by grid search (Lagrangian dual of the α-weighted problem over a finite grid). No smoothing penalty β is used: the single parameter η already bounds smoothing (0 ≤ λ_ij ≤ λ).
* Global-scalar ablation: a_ij = c, c chosen by the same rule on the same 11-point grid.
* No-pairwise ablation: λ_ij = 0 ⇒ q_i = p_i exactly (verified numerically to 5e-14).

**Use of the protected attribute.** HCA needs A_i for both endpoints at inference time, i.e. differential treatment of edges by group membership. The protected-attribute-free alternative with the same budget is the global scalar; H4 tests whether the group-aware rule earns its use of A. Group thresholds (decision-only mitigation) also use A at decision time. FairWalk uses A at representation time only.

## Fairness metrics
s_g = P(D=1|G=g), t_g = P(D=1|Y=1,G=g), f_g = P(D=1|Y=0,G=g); ΔDP = max s − min s; ΔEOpp = max t − min t; ΔEOdds = max(Δt, Δf); R_sel = min s / max s (undefined if max s = 0). Smoothing-attributable change: Δ_smooth(M) = M(uniform) − M(unary) on the same split, threshold rule and evaluation nodes.
