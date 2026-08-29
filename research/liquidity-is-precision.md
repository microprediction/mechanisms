# Liquidity is precision: shrinkage estimation as market aggregation

*Status: a worked generalization with a calibrated novelty assessment; §4
corrects one claim of the source thread. Reference implementation:
[`liquidity_fusion.py`](../mechanisms/liquidity_fusion.py), theorem-tested in
[`test_liquidity_fusion.py`](../tests/test_liquidity_fusion.py).*

The merge law (merging cost-function makers is infimal convolution; Proposition
6 of the
[composition paper](../papers/composition-and-the-algebra-of-mechanisms.md),
extended to fees in
[proportional-fees-and-the-order-book.md](proportional-fees-and-the-order-book.md))
has a statistical reading: liquidity is precision. Each maker's contributed
liquidity matrix acts as the precision of an independent Gaussian source, the
merged market quotes the posterior mean, and its price impact is the posterior
covariance. Shrinkage estimation then appears natively: a naive participant
quoting coordinates independently is a Ledoit-Wolf shrinkage target whose
intensity is its capital share, tuned by profit and loss rather than by
formula. One claim from the source discussion does not survive the algebra and
is corrected in §4: inventory shifts a maker's quote, never its weight.

## 1. The identification

Merging makers with costs $C_i$ gives the maker with cost
$C = C_1 \square \cdots \square C_n$, and conjugates add, so the aggregate
supply at price $p$ is $S(p) = \sum_i s_i(p)$ with slope

$$S'(p) \;=\; \sum_i \nabla^2 C_i^*(p),$$

exactly and at every price. Call $\Lambda_i(p) = \nabla^2 C_i^*(p)$, the
inverse Hessian of the cost at the corresponding point, maker $i$'s
*contributed liquidity*: liquidity adds under merging, and the merged market's
price impact is $(\sum_i \Lambda_i)^{-1}$. This additivity of price-space
Hessians across providers is worked out explicitly by Bhaskara, Frongillo,
Lindgren & Papireddygari (2023); what follows is its probabilistic reading.

For quadratic costs the statement is global, not just differential. Let maker
$i$ quote $m_i$ (its marginal price at current inventory) with constant
liquidity $\Lambda_i$. Supplies are $s_i(p) = \Lambda_i(p - m_i)$, and
clearing a net demand $\Delta$ gives

$$p^*(\Delta) \;=\; \Big(\sum_i \Lambda_i\Big)^{-1}\Big(\sum_i \Lambda_i m_i
\;+\; \Delta\Big).$$

At $\Delta = 0$ this is precision-weighted fusion: the market's consensus is
the Bayesian posterior mean of independent Gaussian sources with means $m_i$
and precisions $\Lambda_i$, and the posterior precision is the aggregate
liquidity. Equivalently, the merged market computes the logarithmic opinion
pool of Gaussian experts. Capital scales precision: for a perspective family
$C_b = b\,C_1(\cdot/b)$, contributed liquidity is linear in $b$, so a maker's
weight in the consensus is its capital.

The identification runs both directions. Where nobody quotes confidently,
aggregate liquidity is low and price impact is high: the market reports a wide
posterior in the direction it knows least about, which is exactly what a point
estimate of a covariance matrix fails to communicate.

## 2. Covariance as the estimand

Nothing above cares what the securities pay. Take them to pay realized second
moments (entries $X_j X_k$, centered as desired) and the price vector is a
covariance estimate, the merged market a covariance estimator, and each maker
a source with a correlation model and capital behind it. Variance is not
elicitable on its own but has elicitation complexity two, jointly with the
mean (Lambert, Pennock & Shoham 2008; Frongillo & Kash 2021), and
second-moment securities are the standard device for reading a market's
variance belief (Snowberg, Wolfers & Zitzewitz 2013), so the securities are
well-posed. A maker specialising in a subset of coordinates simply contributes
no liquidity off its subspace; the aggregate posterior is wide there, and
price impact says so.

## 3. Ledoit-Wolf is a market participant

Put one maker in the pool with a full correlation model, quoting the sample
estimate $S$ with capital $w_S$, and one naive maker quoting each entry
independently toward a structured target $F$ (identity, single-factor,
constant-correlation) with capital $w_F$. With common liquidity shape the
fused quote is

$$p^* \;=\; (1-\delta)\,S + \delta\,F, \qquad
\delta \;=\; \frac{w_F}{w_S + w_F},$$

the Ledoit-Wolf estimator with the shrinkage intensity read off as the naive
quoter's capital share rather than computed from a formula
([`test_liquidity_fusion.py`](../tests/test_liquidity_fusion.py)). Every
classical target is a participant with that prior, and the market runs the
whole ensemble at once: with $k$ targets the fused quote is the
capital-weighted multi-target shrinkage estimator.

The intensity then self-tunes by selection. Trading against informed flow, a
maker's expected loss per round is $\tfrac12 e_i^\top \Lambda_i e_i$ with
$e_i$ its quote error: each maker pays for its error in proportion to its own
confidence. If the correlations are real the correlated maker bleeds less,
compounds its capital share, and $\delta \to 0$; if they are spurious it
bleeds more and $\delta \to 1$. The tuning problem the shrinkage literature
solves with formulas or cross-validation is solved here by wealth dynamics.
For discrete beliefs this mechanism is established: a market of Kelly bettors
is Bayesian model averaging with wealths as posterior weights (Beygelzimer,
Langford & Pennock 2012), wealth updates are posterior updates in
machine-learning markets (Storkey, Millin & Geras 2012), wealth is the
learning rate in the stochastic-mirror-descent reading (Frongillo, Della
Penna & Reid 2012), and market selection for accurate beliefs is Blume &
Easley (2006). Defensive quoting is directional shrinkage: withdrawing liquidity from
directions involving one coordinate removes that maker's precision exactly
there, tilting the aggregate toward whatever the rest of the pool says.

## 4. What inventory does, and does not, do

The source discussion proposed a third layer: that as a book loads up, the
maker's contributed precision decays (for the log-cosh family, like
$\operatorname{sech}^2$), re-weighting the aggregate toward fresher sources.
The algebra says otherwise. For *any* fixed cost function the supply curve is

$$s_i(p) \;=\; (\nabla C_i)^{-1}(p) - q_i,$$

so inventory translates the curve and cannot change its slope at any price:
contributed liquidity at the consensus price is inventory-independent, for the
log-cosh family $b/(1-p^2)$ whatever the book. Loading shifts the maker's
quote (the mean of the source), never its weight (the precision). The
$\operatorname{sech}^2$ intuition identifies a different object: the Hessian
$\nabla^2 C_i$ at the maker's own inventory is the rate at which flow updates
its quote, its learning rate in the market-as-online-learning reading, not its
weight in the fusion. Within the algebra, capital is precision and inventory
is mean shift, and the two are orthogonal
([`test_liquidity_fusion.py`](../tests/test_liquidity_fusion.py)).

The corrected statement of "dynamic shrinkage" is therefore: fusion weights
move only when capital moves (selection, §3) or when the cost function itself
is changed between rounds, the province of liquidity-sensitive and adaptive
makers. A maker that wants its weight to decay as it warehouses risk must
shrink its posted liquidity explicitly; the fee machinery of the companion
note prices the flow but does not reweight the fusion.

## 5. Coverage and completion

A pool whose makers hold correlation models over subsets of coordinates
quotes a patchwork: precision lives where models and capital are, and the
aggregate precision matrix is sparse where nobody quotes cross-terms.
Defaulting the unquoted partial correlations to zero is the maximum-entropy
completion of the quoted entries, the classical covariance-selection
prescription, and it emerges from coverage rather than from a penalty term.
The market analogue of the graphical-lasso regulariser is an entry fee: edges
are quoted only where someone expects the precision to pay for itself.

## 6. Implementation

[`liquidity_fusion.py`](../mechanisms/liquidity_fusion.py) provides
`QuadraticMaker` (vector cost, prior quote, capital-scaled liquidity) and
`fuse` (clearing price, aggregate precision, fills). The tests verify: fusion
equals the Gaussian posterior and repricing makers to consensus clears; price
impact equals the inverse aggregate precision; a diagonal quoter yields the
Ledoit-Wolf blend with intensity equal to its capital share; informed flow
drains capital from the worse model so the intensity self-tunes; inventory
changes a quote but not a weight (for quadratic and log-cosh makers both);
non-positive-definite liquidity is rejected.

## 7. Prior art and novelty (calibrated)

Compiled from two web-verified searches (markets as estimators; shrinkage and
fusion). Claim by claim:

*Liquidity adds; merging is fusion (§1).* The convex half is published:
merging makers is infimal convolution with dual generating functions adding,
and the price-space Hessian additivity is worked out explicitly, in Bhaskara,
Frongillo, Lindgren & Papireddygari (2023), entirely in convex-analysis
language with no probabilistic reading. The probabilistic half is textbook in
other fields: precision addition for independent Gaussian sources is standard
estimation theory (Bar-Shalom et al. 2001), inverse-variance weighting of
forecasts is Bates & Granger (1969) with the dependent-source caveat in
Winkler (1981), and log opinion pools of Gaussians are precision-weighted.
That market equilibria implement opinion pools is also old: Pennock & Wellman
(1997) for CARA and log utilities, Storkey (2011) and Storkey, Millin & Geras
(2012) for product- and mixture-of-experts equilibria over discrete outcomes.
Das and coauthors designed a market maker that performs Gaussian MAP updates
(Brahma, Das & Magdon-Ismail 2012), a different object, and Sethi & Wortman
Vaughan (2016) show that with budgeted risk-averse traders the limit price is
*not* a function of beliefs alone, a caution about which trading models
support a clean fusion identity. Not found anywhere: the quadratic
instantiation stated as such, clearing price of merged cost-function makers =
precision-weighted mean of quotes, price impact = posterior covariance.
Verdict: the identity is a short step from published pieces and carries
expository rather than mathematical novelty; the step appears untaken.

*Shrinkage intensity as capital share, tuned by selection (§3).* The
Bayes-blend reading of shrinkage is classical: Haff (1980) and Frost &
Savarino (1986) derive sample-target blends as empirical-Bayes posterior
means, the inverse-Wishart posterior weighs prior and sample by pseudo-counts
(capital-like masses), and Ledoit & Wolf (2004) give their own geometric
Bayesian interpretation. Multi-target shrinkage exists with weights from MSE
formulas or quadratic programs (Lancewicki & Aladjem 2014; Bartz et al. 2014;
George 1986 for multiple targets in the mean problem). Economic-performance
tuning of intensity exists: DeMiguel, Martin-Utrera & Nogales (2013)
calibrate to portfolio criteria, Ban, El Karoui & Lim (2018) tune by
performance-based cross-validation, Caldeira et al. (2017) combine covariance
forecasts by economic criteria, Kelly, Malamud, Pourmohammadi & Trojani
(2023) optimize a portfolio of shrinkage portfolios. Wealth-as-weight is
established for discrete beliefs (Beygelzimer, Langford & Pennock 2012;
Storkey, Millin & Geras 2012; Blume & Easley 2006; Evstigneev, Hens &
Schenk-Hoppé on the Kelly rule). Not found: shrinkage intensity identified
with the capital ratio of quoting agents, or multi-target weights evolved by
survival rather than computed. Verdict: likely novel as a synthesis, narrow
gap; every ingredient is published, and the framing must cite the
economic-calibration literature as the nearest neighbour.

*Covariance as the estimand (§2).* Elicitability is settled (variance not
elicitable alone, Lambert, Pennock & Shoham 2008; complexity two, Frongillo &
Kash 2021; second-moment securities, Snowberg, Wolfers & Zitzewitz 2013).
No incentive-designed market for a covariance *matrix* with positive
semidefinite consistency was found in the prediction-market literature;
correlation trades in practice (correlation swaps, CBOE implied-correlation
indices) as derivative pricing without elicitation guarantees. Verdict: the
matrix-level design is open, which cuts both ways: unoccupied, and unbuilt.

*Inventory is mean shift, not weight (§4).* The correction is elementary
(one line from the supply-curve formula) and is this note's own; the
learning-rate object it distinguishes is published (wealth as learning rate,
Frongillo, Della Penna & Reid 2012; inverse liquidity as step size, Nueve &
Waggoner 2025).

*Coverage and completion (§5).* Known mathematics, new framing at most. Zero
partial correlations on unspecified entries characterize the
determinant-maximizing completion (Grone, Johnson, Sá & Wolkowicz 1984),
which is Dempster's (1972) covariance selection; the L1 route is the
graphical lasso (Friedman, Hastie & Tibshirani 2008); max-det completion is
applied to missing financial correlations by Georgescu, Higham & Peters
(2018). Deriving the sparsity pattern from market coverage was not found.

Overall: the mathematics is classical or a short step from Bhaskara et al.
(2023); the candidate contribution is the identification (liquidity is
precision, capital is mass, selection is tuning) and the correction in §4.
Positioning must cite Bhaskara et al. (2023), Storkey, Millin & Geras (2012),
Beygelzimer, Langford & Pennock (2012), Haff (1980) and the multi-target and
economic-calibration shrinkage literature as the nearest neighbours.

## 8. Open questions

- *Fees in the fusion.* With per-maker fees the supply curves carry dead
  zones (the companion note), so the fused estimate is a lasso-like
  selection: sources inside their band contribute nothing. What estimator
  does the fee-bearing pool compute, and is the fee the exact market analogue
  of the graphical-lasso penalty conjectured in §5?
- *Selection dynamics, formally.* §3 argues drift, not convergence. Under
  what flow model does the capital share converge to the oracle shrinkage
  intensity, and at what rate relative to the Ledoit-Wolf formula's risk?
- *Non-quadratic exactness.* The fusion identity is exact for quadratic
  costs and differential in general. Is there a family for which the
  clearing price is an exact generalized mean of quotes (the log-cosh family
  clears in arctanh coordinates, suggesting a Kolmogorov-mean form)?
- *Covariance intersection.* Precision addition assumes independent sources;
  makers trained on overlapping data are not. Does a covariance-intersection
  style fusion correspond to any modification of the merge operator?

## References

*Merging makers, markets as aggregation and learning.*
- Pennock, D. M. & Wellman, M. P. (1997). "Representing Aggregate Belief
  through the Competitive Equilibrium of a Securities Market." *UAI*.
- Storkey, A. J. (2011). "Machine Learning Markets." *AISTATS*, PMLR 15.
- Storkey, A. J., Millin, J. & Geras, K. (2012). "Isoelastic Agents and
  Wealth Updates in Machine Learning Markets." *ICML*.
- Barbu, A. & Lay, N. (2012). "An Introduction to Artificial Prediction
  Markets for Classification." *JMLR* 13.
- Frongillo, R., Della Penna, N. & Reid, M. D. (2012). "Interpreting
  Prediction Markets: A Stochastic Approach." *NeurIPS*.
- Beygelzimer, A., Langford, J. & Pennock, D. M. (2012). "Learning
  Performance of Prediction Markets with Kelly Bettors." *AAMAS*.
  arXiv:1201.6655.
- Hu, J. & Storkey, A. (2014). "Multi-period Trading Prediction Markets with
  Connections to Machine Learning." *ICML*. arXiv:1403.0648.
- Sethi, R. & Wortman Vaughan, J. (2016). "Belief Aggregation with Automated
  Market Makers." *Computational Economics* 48(1).
- Brahma, A., Chakraborty, M., Das, S., Lavoie, A. & Magdon-Ismail, M.
  (2012). "A Bayesian Market Maker." *EC*.
- Bhaskara, A., Frongillo, R., Lindgren, E. & Papireddygari, M. (2023). "A
  General Theory of Liquidity Provisioning for Prediction Markets."
  arXiv:2311.08725.
- Nueve, E. & Waggoner, B. (2025). "Smooth Quadratic Prediction Markets."
  *NeurIPS*. arXiv:2505.02959.
- Blume, L. & Easley, D. (2006). "If You're So Smart, Why Aren't You Rich?
  Belief Selection in Complete and Incomplete Markets." *Econometrica* 74(4).
- Evstigneev, I. V., Hens, T. & Schenk-Hoppé, K. R. (2011). "Survival and
  Evolutionary Stability of the Kelly Rule." SSRN 1468523.

*Covariance shrinkage.*
- Stein, C. (1956). "Inadmissibility of the Usual Estimator for the Mean of a
  Multivariate Normal Distribution." *Berkeley Symposium* 1.
- Haff, L. R. (1980). "Empirical Bayes Estimation of the Multivariate Normal
  Covariance Matrix." *Annals of Statistics* 8(3), 586–597.
- Frost, P. A. & Savarino, J. E. (1986). "An Empirical Bayes Approach to
  Efficient Portfolio Selection." *JFQA* 21(3), 293–305.
- Ledoit, O. & Wolf, M. (2004). "A Well-Conditioned Estimator for
  Large-Dimensional Covariance Matrices." *J. Multivariate Analysis* 88(2);
  (2004). "Honey, I Shrunk the Sample Covariance Matrix." *JPM* 30(4).
- Schäfer, J. & Strimmer, K. (2005). "A Shrinkage Approach to Large-Scale
  Covariance Matrix Estimation." *Stat. Appl. Genet. Mol. Biol.* 4(1).
- George, E. I. (1986). "Minimax Multiple Shrinkage Estimation." *Annals of
  Statistics* 14(1), 188–205.
- Lancewicki, T. & Aladjem, M. (2014). "Multi-Target Shrinkage Estimation for
  Covariance Matrices." *IEEE Trans. Signal Processing* 62(24), 6380–6390.
- Bartz, D., Höhne, J. & Müller, K.-R. (2014). "Multi-Target Shrinkage."
  arXiv:1412.2041.
- Engle, R. F., Ledoit, O. & Wolf, M. (2019). "Large Dynamic Covariance
  Matrices." *JBES* 37(2), 363–375.

*Economic calibration of estimators.*
- DeMiguel, V., Garlappi, L. & Uppal, R. (2009). "Optimal versus Naive
  Diversification: How Inefficient Is the 1/N Portfolio Strategy?" *RFS*
  22(5), 1915–1953.
- DeMiguel, V., Martin-Utrera, A. & Nogales, F. J. (2013). "Size Matters:
  Optimal Calibration of Shrinkage Estimators for Portfolio Selection."
  *J. Banking & Finance* 37(8), 3018–3034.
- Ban, G.-Y., El Karoui, N. & Lim, A. E. B. (2018). "Machine Learning and
  Portfolio Optimization." *Management Science* 64(3), 1136–1154.
- Caldeira, J. F., Moura, G. V., Nogales, F. J. & Santos, A. A. P. (2017).
  "Combining Multivariate Volatility Forecasts: An Economic-Based Approach."
  *J. Financial Econometrics* 15(2), 247–285.
- Kelly, B., Malamud, S., Pourmohammadi, M. & Trojani, F. (2023). "Universal
  Portfolio Shrinkage." NBER w32004.

*Fusion and elicitation.*
- Bates, J. M. & Granger, C. W. J. (1969). "The Combination of Forecasts."
  *OR Quarterly* 20(4), 451–468.
- Winkler, R. L. (1981). "Combining Probability Distributions from Dependent
  Information Sources." *Management Science* 27(4), 479–488.
- Julier, S. J. & Uhlmann, J. K. (1997). "A Non-divergent Estimation
  Algorithm in the Presence of Unknown Correlations." *Proc. American
  Control Conference*, 2369–2373.
- Bar-Shalom, Y., Li, X. R. & Kirubarajan, T. (2001). *Estimation with
  Applications to Tracking and Navigation.* Wiley.
- Lambert, N. S., Pennock, D. M. & Shoham, Y. (2008). "Eliciting Properties
  of Probability Distributions." *EC*, 129–138.
- Frongillo, R. & Kash, I. A. (2021). "Elicitation Complexity of Statistical
  Properties." *Biometrika* 108(4), 857–879.
- Snowberg, E., Wolfers, J. & Zitzewitz, E. (2013). "Prediction Markets for
  Economic Forecasting." *Handbook of Economic Forecasting* 2A; NBER w18222.

*Precision sparsity and completion.*
- Dempster, A. P. (1972). "Covariance Selection." *Biometrics* 28(1),
  157–175.
- Grone, R., Johnson, C. R., Sá, E. M. & Wolkowicz, H. (1984). "Positive
  Definite Completions of Partial Hermitian Matrices." *Linear Algebra
  Appl.* 58, 109–124.
- Friedman, J., Hastie, T. & Tibshirani, R. (2008). "Sparse Inverse
  Covariance Estimation with the Graphical Lasso." *Biostatistics* 9(3),
  432–441.
- Georgescu, D. I., Higham, N. J. & Peters, G. W. (2018). "Explicit Solutions
  to Correlation Matrix Completion Problems, with an Application to Risk
  Management and Insurance." *Royal Society Open Science* 5(3).
