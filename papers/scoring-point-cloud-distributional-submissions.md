# Scoring Point-Cloud Distributional Submissions

### The deconvolution incentive, jittered outcomes, and the heat ladder

Peter Cotton · *Working draft v0.2* · 2026

> **Status.** An evolving working note, not a finished paper. Theorems 1–3 are
> proved below at working-draft rigor in the population-law setting: the
> cloud is idealised by its sampling law, and the smoothing/jitter kernel is
> fixed in advance. Finite clouds and participant-endogenous bandwidths are
> outside the proved results (§7). The mechanism and the closed forms are
> implemented and unit-tested in
> [`mechanisms/nearest_the_pin.py`](../mechanisms/nearest_the_pin.py)
> (`mollified_log_score`, and tests that verify Theorem 1 numerically in both
> directions). The prior-art audit lives in
> [`research/mollified-scoring-and-the-heat-ladder.md`](../research/mollified-scoring-and-the-heat-ladder.md).
> *v0.2:* sharpened hypotheses following referee feedback — conditional
> truthfulness gap (Thm. 1), integrability assumptions (Thm. 2), exogenous
> bandwidth requirement, standard de Bruijn regularity in place of a tail
> overclaim (Thm. 3), and an explicit level-vs-difference payment analysis
> with a new proposition that band scores are strictly proper.
> Drafted with AI assistance; errors are mine.

---

## Abstract

Forecasting contests often accept a cloud of Monte-Carlo samples, smooth it
into a kernel density estimate, and score the result at the realised outcome.
This design is improper: the optimal cloud is drawn from the deconvolution of
the forecaster's belief by the smoothing kernel, which for a Gaussian belief
means shaving the squared bandwidth off the variance (Theorem 1). Jittering
the outcome with the same kernel restores strict propriety, because
convolution by a kernel whose characteristic function has dense support is
injective on probability measures (Theorem 2). Repeating the repaired score
across smoothing scales decomposes the log-score edge, via de Bruijn's
identity, into Fisher-divergence payments for shape plus a coarse-scale
payment for mass (Theorem 3), yielding a budget-balanced multiscale pool. All
results are population-level, with the kernel fixed in advance.

---

## 1. The problem

A contest accepts from each participant $i$ a cloud
$x^{(i)}_1,\dots,x^{(i)}_m\in\mathbb R^d$, smooths it into a density
$\hat q_i = \tfrac1m\sum_j \varphi_h(\cdot-x^{(i)}_j)$, and — when the outcome
$z$ is revealed — rewards $\log \hat q_i(z)$ (or splits a pot in proportion to
$\hat q_i(z)$, the exponentiated form). This is the reward rule of the
[nearest-the-pin parimutuel](nearest-the-pin-parimutuel.md) and of
monteprediction-style contests, and the standard evaluation protocol for
sample-based generative models.

The question this paper answers: *what cloud should a rational participant
submit?* The answer is not "samples from your belief," and the failure has
a closed form, a one-line repair, and, once repaired, a multi-scale
structure with its own uses.

Throughout, we idealise the cloud by its sampling law $\rho$ ($m\to\infty$;
finite $m$ is Open Problem 1), so the mechanism observes $T_h\rho := \rho*\varphi_h$,
where $\varphi_h$ is a density on $\mathbb R^d$, symmetric, strictly positive
(the Gaussian $N(0,h^2 I)$ is canonical). The truth is $p^*$; all expectations
below are assumed finite (for Gaussian kernels and beliefs with finite second
moments, $\log T_h\rho$ has at-worst-quadratic tails, so this is mild).

## 2. The deconvolution incentive

**Theorem 1 (improperness of raw-outcome KDE scoring).** *Let
$S_{\mathrm{raw}}(\rho,z)=\log(T_h\rho)(z)$. Then*

$$\mathbb E_{z\sim p^*}\,S_{\mathrm{raw}}(\rho,z)
   \;=\; -H(p^*) \;-\; \mathrm{KL}\!\big(p^*\,\Vert\, T_h\rho\big),$$

*so the report solves $\min_{q\in T_h\mathcal P}\mathrm{KL}(p^*\Vert q)$, where
$\mathcal P$ is the set of probability laws. Consequently:*

*(i) If the deconvolution $\rho^\dagger := T_h^{-1}p^*$ exists in
$\mathcal P$, it is the unique optimal report (uniqueness by the injectivity
of $T_h$, Lemma 1), it differs from the truth, and truthful reporting is
strictly suboptimal with truthfulness gap*

$$\Delta \;=\; \mathrm{KL}\!\big(p^*\Vert p^**\varphi_h\big)\;>\;0.$$

*(ii) If $p^*\notin T_h\mathcal P$, the optimum is the KL projection of $p^*$
onto the convex class $T_h\mathcal P$, and $\Delta$ above is only the loss of
the truthful report relative to the unattainable ideal $q=p^*$; the gap to the
attainable optimum is $\Delta-\inf_{\rho\in\mathcal P}
\mathrm{KL}(p^*\Vert T_h\rho)$, which can vanish in degenerate cases (for a
point-mass belief the truthful cloud is optimal).*

*(iii) Gaussian closed form: $p^*=N(\mu,\Sigma)$, $\varphi_h=N(0,h^2I)$. The
deconvolution exists iff $\Sigma-h^2I\succeq 0$, and then
$\rho^\dagger=N(\mu,\Sigma-h^2I)$. In one dimension with $\tau^2>h^2$: shave
$h^2$ off the variance, with gap
$\Delta=\tfrac12[\log(1+h^2/\tau^2)-h^2/(\tau^2+h^2)]\approx h^4/(4\tau^4)$.*

**Proof.** The identity is the standard cross-entropy decomposition,
$\int p^*\log q = -H(p^*)-\mathrm{KL}(p^*\Vert q)$, applied to $q=T_h\rho$;
maximizing over $\rho$ is minimizing KL over the convex image class
$T_h\mathcal P$. (i) If $p^*\in T_h\mathcal P$ the KL term can be driven to
zero, its unique minimum, and only by $q=p^*$; Lemma 1 lifts uniqueness of $q$
to uniqueness of $\rho$; the truthful report attains
$\mathrm{KL}(p^*\Vert p^*_h)$ where $p^*_h:=p^**\varphi_h$, and $p^*_h\neq p^*$
always: taking characteristic functions,
$\hat p^*(\omega)\,(1-\hat\varphi_h(\omega))=0$ for all $\omega$; for the
Gaussian kernel $\hat\varphi_h(\omega)=e^{-h^2|\omega|^2/2}<1$ off
$\omega=0$, so $\hat p^*$ would have to vanish off the origin, impossible for
a characteristic function (continuity and $\hat p^*(0)=1$). (ii) KL is jointly
lower semicontinuous and strictly convex in its second argument where finite,
and $T_h\mathcal P$ is convex, so the attainable optimum is the KL projection
when attained; the point-mass example ($p^*=\delta_\mu$: the truthful cloud
maximises $(T_h\rho)(\mu)$) shows the gap to the attainable optimum can be
zero. (iii) Gaussians:
convolution adds covariances; the KL between $N(0,\tau^2)$ and
$N(0,\tau^2+h^2)$ is the stated expression; Taylor expansion gives
$h^4/(4\tau^4)$. $\blacksquare$

**Remarks.** (a) When the deconvolution fails to exist ($\Sigma-h^2I\not\succeq
0$, or a rough $p^*$), the optimum is the KL projection of $p^*$ onto
$T_h\mathcal P$ and can degenerate toward atomic clouds — exactly the
point-mass exploit of Theis, van den Oord & Bethge (2016), who observed the
improperness (without the deconvolution characterization). (b) The gap is
fourth order in $h$, which is why the flaw survives casual inspection — but it
is a *slope*, and any optimising submitter, human or fitted, walks down it.
(c) The phenomenon is the log-score/bandwidth sibling of the "fair scores"
findings for the ensemble CRPS (Fricker, Ferro & Stephenson 2013; Ferro 2014),
where the analogous cheat is derived along the ensemble-size axis.

## 3. The repair: jitter the pin

**Definition (mollified log score).** With $\varepsilon\sim N(0,I)$,

$$S_h(\rho,z)\;=\;\mathbb E_\varepsilon\big[\log (T_h\rho)(z+h\varepsilon)\big]
   \;=\;\big(\varphi_h * \log T_h\rho\big)(z).$$

Operationally: *settle at a jittered outcome* $z'=z+h\varepsilon$ — or use the
right-hand form, which integrates the jitter analytically and makes settlement
deterministic. The two have the same expectation, hence identical incentive
properties.

**Lemma 1 (injectivity criterion).** *Convolution $T_\varphi$ is injective on
probability measures iff the zero set of the characteristic function has empty
interior,*

$$\operatorname{int}\{\omega : \hat\varphi(\omega)=0\} \;=\; \emptyset$$

*(equivalently, $\{\hat\varphi\neq 0\}$ is dense in $\mathbb R^d$).*

**Proof.** ($\Leftarrow$) $T_\varphi\rho=T_\varphi\rho'$ gives
$(\hat\rho-\hat\rho')\hat\varphi\equiv0$, so $\hat\rho=\hat\rho'$ on a dense
set, hence everywhere by continuity of characteristic functions, hence
$\rho=\rho'$ by the uniqueness theorem. ($\Rightarrow$) If $\hat\varphi$
vanishes on an open ball $B$ (with $0\notin B$, since $\hat\varphi(0)=1$),
take $\psi\in C_c^\infty(B)$, $\psi\neq0$, and set
$\hat g(\omega)=\psi(\omega)+\overline{\psi(-\omega)}$, so that $g$ is a real
Schwartz function (a finite signed density) with total mass
$\int g=\hat g(0)=0$ and $\hat g$ supported where $\hat\varphi=0$. Choose a
base density $p$ with tails heavier than Schwartz decay (Cauchy), so
$|\epsilon g|\le p$ pointwise for small $\epsilon>0$. Then
$q=p+\epsilon g$ is a probability density distinct from $p$, and
$\widehat{T_\varphi q}=\hat\varphi\,(\hat p+\epsilon\hat g)
=\hat\varphi\,\hat p$: two distinct laws with identical smoothings.
$\blacksquare$

This is Wiener-flavoured (Wiener's Tauberian theorem is the $L^1$ statement);
for probability measures the continuity of characteristic functions buys the
dense-support version. Practical reading: Gaussian and Laplace jitter are
injective (nonvanishing characteristic functions); the uniform kernel is
also fine (sinc zeros are isolated, hence the nonvanishing set is dense);
band-limited kernels fail (e.g. Fejér-type kernels, whose characteristic
function vanishes on a half-line — two beliefs agreeing on low frequencies
become indistinguishable).

**Theorem 2 (kernel-channel properness).** *Let $S$ be a scoring rule, $K$ a
Markov kernel with push-forward $T_K$ on laws, and define*

$$S_K(\rho,z)\;=\;\mathbb E_{z'\sim K(z,\cdot)}\big[S(T_K\rho,\,z')\big].$$

*Assume (a) $\mathbb E_{w\sim T_K p}\,|S(T_K\rho, w)|<\infty$ for the laws
$p,\rho$ under comparison (or adopt an extended-expectation convention), and
(b) $T_K\mathcal P$ lies within the report class on which $S$ is defined. If
$S$ is proper, $S_K$ is proper. If $S$ is strictly proper on
$T_K\mathcal P$, then $S_K$ is strictly proper on $\mathcal P$ iff
$T_K$ is injective on $\mathcal P$. In particular the mollified log score
$S_h$ is strictly proper for the pre-smoothing law whenever the zero set of
$\hat\varphi$ has empty interior — for Gaussian jitter, always.*

**Proof.** By Fubini,
$\mathbb E_{z\sim p^*}S_K(\rho,z)=\mathbb E_{w\sim T_Kp^*}S(T_K\rho,w)$: the
expected score of the report $T_K\rho$ for the outcome law $T_Kp^*$, under
$S$. Properness of $S$ maximizes this at $T_K\rho=T_Kp^*$, which the truthful
$\rho=p^*$ achieves — properness. Strict properness of $S$ on the image class
forces $T_K\rho=T_Kp^*$ at any maximizer; injectivity of $T_K$ lifts this to
$\rho=p^*$, and conversely if $T_K$ is not injective two distinct reports tie.
For $S=\log$ and $K$ = convolution by $\varphi_h$: the log score is strictly
proper on densities, $T_K\rho=\rho*\varphi_h$ is a density, and Lemma 1 gives
injectivity. $\blacksquare$

**How much to jitter? Exactly as much as you smooth.** The jitter is not a
free parameter; it is pinned, kernel for kernel, to the mechanism's own
smoothing. Jittering with s.d. $j$ while smoothing with bandwidth $h$ pairs the
jittered truth $p^**\varphi_j$ with the smoothed report $\rho*\varphi_h$, and
in the all-Gaussian setting the optimal report variance is

$$v^\ast \;=\; \tau^2 + j^2 - h^2 .$$

At $j=0$ this is Theorem 1's shave. At $j=h$, and only at $j=h$, the optimum
is the truth. At $j>h$ the mechanism pays padding by $j^2-h^2$. The formula
holds where admissible ($v^\ast\ge0$); otherwise the Gaussian-family optimum
sits on the boundary (the KL-projection case of Theorem 1(ii)). Under-jitter
rewards sharpening, over-jitter rewards blurring, and matching the bandwidth is
the unique fixed point (the general statement is Theorem 2 with the *same*
kernel on both sides; a mismatched pair elicits
$\arg\min_\rho \mathrm{KL}(p^**\varphi_j\,\Vert\,\rho*\varphi_h)$, the
$j$-blurred belief deconvolved by $h$).

**The bandwidth must be exogenous.** Theorem 2 assumes a *fixed* channel: the
smoothing/jitter kernel must not depend on the submission being scored. A
data-driven bandwidth computed *from the participant's own cloud* (Scott's
rule on the submission, as reference KDE implementations default to) puts
$h(\rho)$ under the participant's control, the score becomes
$\mathbb E_\varepsilon\log(T_{h(\rho)}\rho)(z+h(\rho)\varepsilon)$, and strict
properness no longer follows from Theorem 2. In a contest, freeze the
bandwidth before submissions are observed (from the outcome history, a
reference climatology, or a posted rule) and jitter with that frozen kernel.
The incentives of participant-endogenous bandwidths are an open problem (§7).

**Attribution.** The properness of the convolved score against a
noised outcome is not new: Bröcker & Smith (2007, §5) prove it for a general
observation-noise channel, and Ferro (2017, Prop. 3) works out exactly the
white-noise/Gaussian case. What both left open is strictness. Bröcker &
Smith, verbatim: "If $S$ is strictly proper though, $\bar S$ is not
necessarily strictly proper, because if $\bar q(z)=\bar p(z)$, this does not
necessarily mean equality of $p(x)$ and $q(x)$." Lemma 1 closes exactly that
gap. The discrete-outcome analog is the label-noise literature's forward loss
correction with an invertible transition matrix (Patrini et al. 2017, Thm. 2;
van Rooyen & Williamson 2018). Verification treats outcome noise as a
nuisance to be endured, and Ferro explicitly doubts the "efficacy" of
perturbing observations; read as mechanism design, jitter matched to the
mechanism's own smoothing is what makes the point-cloud game strictly
proper.

## 4. The heat ladder

Let $p_t := p * N(0,tI)$, the heat flow ($\partial_t p_t=\tfrac12\Delta p_t$).
To keep one time variable, write $u = h^2 + t$ for the *total* smoothing scale:
the rung at flow time $t$ is the §3 score with kernel
$\varphi_{\sqrt u}$ — for clouds this is free to compute, since it is *the
same cloud* scored with bandwidth $\sqrt u$ against a pin jittered by the same
kernel. Below, subscripts $t$ refer to the flow started from the already
$h$-smoothed laws, so every rung the mechanism runs has $u \ge h^2 > 0$.

**Theorem 3 (scale decomposition of the log-score edge).** *Let $p^*,\rho$
have finite second moments, and suppose the standard relative de Bruijn
regularity conditions hold on $[s,T]$: $\mathrm{KL}(p^*_t\Vert\rho_t)$ finite
along the flow, the relative Fisher divergence integrable on the interval, and
enough decay to justify the integrations by parts below. Then for
$0\le s<T$,*

$$\frac{d}{dt}\,\mathrm{KL}\!\big(p^*_t\Vert\rho_t\big)
   \;=\;-\tfrac12\,D_F\!\big(p^*_t\Vert\rho_t\big),
\qquad
\mathrm{KL}(p^*_s\Vert\rho_s)
 = \mathrm{KL}(p^*_T\Vert\rho_T)
 + \tfrac12\!\int_s^T\! D_F(p^*_t\Vert\rho_t)\,dt,$$

*where $D_F(p\Vert q)=\int p\,\lVert\nabla\log p-\nabla\log q\rVert^2$ is the
Fisher divergence.*

**Proof.** Write $p=p^*_t$, $q=\rho_t$, both solving $\partial_t
u=\tfrac12\Delta u$. Then

$$\frac{d}{dt}\int p\log\frac pq
 \;=\;\int (\partial_t p)\log\frac pq \;+\;\underbrace{\int \partial_t p}_{=0}
 \;-\;\int \frac pq\,\partial_t q .$$

First term: $\tfrac12\int \Delta p\,\log(p/q)
= -\tfrac12\int\nabla p\cdot\nabla\log(p/q)
= -\tfrac12\int p\,\nabla\log p\cdot(\nabla\log p-\nabla\log q)$.
Third term: $-\tfrac12\int \frac pq\,\Delta q
= \tfrac12\int \nabla\!\Big(\frac pq\Big)\cdot\nabla q
= \tfrac12\int p\,(\nabla\log p-\nabla\log q)\cdot\nabla\log q$.
Summing,

$$\frac{d}{dt}\,\mathrm{KL}
 = -\tfrac12\int p\,\lVert\nabla\log p-\nabla\log q\rVert^2
 = -\tfrac12 D_F(p\Vert q),$$

and integrating over $[s,T]$ gives the display. After smoothing by
$u\ge h^2>0$ both densities are positive and smooth, which is necessary but
not by itself sufficient: Gaussian convolution does *not* replace heavy tails
by Gaussian tails, so the vanishing of boundary terms is an assumption (the
stated regularity), automatic for the Gaussian and compactly supported cases
used in the examples. $\blacksquare$

The differential identity is de Bruijn's, in relative form (Stam 1959; Barron
1986; Lyu 2009); its integral form prices the likelihood of diffusion models
(Song, Durkan, Murray & Ermon 2021). Note the bandwidth floor does real work: at $t=0$ an empirical
cloud has no density and the identity is vacuous; every rung the mechanism
actually runs starts at $t\ge h^2$, where everything is smooth.

**The heat-ladder pool.** Fix scales $0=t_0<t_1<\dots<t_K=T$ and weights
$w_k\ge0$. Participant $i$ stakes $s_i$ and submits one cloud. Two payment
schedules must be distinguished, because they buy different things.

*Level schedule.* Rung $k$ splits $w_k\sum_i s_i$ by a budget-balanced rule
driven by the rung score $S^{(k)}_i := S_{\sqrt{h^2+t_k}}(\rho_i,z)$ — the
stake-weighted additive form
$\Delta W_i^{(k)}=w_k\,s_i\,(S_i^{(k)}-\bar S^{(k)})$ with $\bar S^{(k)}$ the
stake-weighted mean (Lambert et al. 2008). Each rung is strictly proper
(Theorem 2), so any nonnegative-weighted sum is. But the expected-regret
decomposition carries *cumulative* weights on the Fisher bands, not the rung
weights themselves: writing $\mathrm{KL}_k := \mathrm{KL}(p^*_{t_k}\Vert
\rho_{t_k})$,

$$\sum_{k=0}^{K} w_k\,\mathrm{KL}_k
 \;=\; \Big(\sum_{k=0}^{K} w_k\Big)\mathrm{KL}_K
 \;+\; \tfrac12\sum_{\ell=0}^{K-1}\Big(\sum_{k=0}^{\ell} w_k\Big)
   \int_{t_\ell}^{t_{\ell+1}} D_F(p^*_t\Vert\rho_t)\,dt .$$

*Difference schedule.* To pay a Fisher band directly, use the score
*differences* as the payment driver.

**Proposition (band scores are strictly proper).** *The difference score
$S^{(k)} - S^{(k+1)}$ has expected regret
$\mathrm{KL}_k - \mathrm{KL}_{k+1}
= \tfrac12\int_{t_k}^{t_{k+1}} D_F(p^*_t\Vert\rho_t)\,dt \ge 0$, with equality
iff $\rho = p^*$. It is therefore itself a strictly proper score, even though
a difference of proper scores is not proper in general.*

**Proof.** The expected value of $S^{(k)}$ under $p^*$ is
$-H(p^*_{t_k}) - \mathrm{KL}_k$; the entropy terms are report-independent, so
the regret of the difference is $\mathrm{KL}_k - \mathrm{KL}_{k+1}$, which is
the band integral by Theorem 3. It vanishes iff $D_F(p^*_t\Vert\rho_t)=0$ on
the band, i.e. $\nabla\log\rho_t=\nabla\log p^*_t$ $p^*_t$-a.e.; after Gaussian
smoothing both densities are everywhere positive, so the log-ratio is constant,
and both integrating to one forces $\rho_t=p^*_t$, hence $\rho=p^*$ by
Lemma 1. $\blacksquare$

**Corollary.** *(i) Each rung (or band), hence the tower, is budget-balanced
in the additive stake-weighted form. (ii) Each level score is strictly proper
for the cloud law (Theorem 2) and each band score is strictly proper (the
Proposition), so truthful submission is optimal in the small-stake,
risk-neutral limit; the multiplicative pot split requires a log-wealth (Kelly)
model stated separately. (iii) Band payments purchase (integrated) Fisher
divergence, which is Hyvärinen-scored shape, invariant to the normalization
of the submission; the top level pays $\mathrm{KL}(p^*_T\Vert\rho_T)$, which
at mode-connecting scales carries the between-mode* mass *that score matching
is blind to (Wenliang & Kanagawa 2020; Zhang et al. 2022; Koehler, Heckett &
Risteski 2023). Under the level schedule the band weights are the cumulative
sums above.*

Practicalities: rung scores are positively correlated (one cloud,
re-smoothed), so discriminative value concentrates in a few well-separated
scales; $K$ of order 3–5, geometrically spaced, mirrors the noise ladders of
annealed score matching (Song & Ermon 2019).

## 5. Historical note: the jitter was intuition first

The microprediction platform (Cotton 2022; launched 2019) added a small amount
of noise to submissions and to the ground truth before settling its cloud-based
lotteries. This was merely intuitive, a fairness-and-anti-gaming instinct
about discreteness and ties, with no incentive theorem attached; the platform
paper recorded the practice in one line and moved on. The verification literature, meanwhile, treated outcome
noise as a defect: something to be modelled away (Saetra et al. 2004; Candille
& Talagrand 2008), with Ferro (2017) explicitly doubting the value of
perturbing observations.

Theorem 2 is the missing theorem: symmetrized jitter is what makes a
smoothed-submission game strictly proper for the submitted law. The theory
also sharpens the intuition into design guidance it could not supply on its
own: the jitter distribution matters. Kernels whose
characteristic function has dense support (Gaussian, Laplace, even uniform)
preserve strict properness; band-limited kernels do not (Lemma 1). Intuition
chose jitter; the theorem chooses *which* jitter.

## 6. Related work

Improperness of sample-based scoring: Theis, van den Oord & Bethge (2016)
(KDE log-likelihood "an improper scoring function"); the fair-scores line for
the ensemble CRPS (Bröcker 2012; Fricker, Ferro & Stephenson 2013; Ferro
2014); the estimator view of KDE log scores (Krüger, Lerch, Thorarinsdottir &
Gneiting 2021); discrete impossibility and randomized repair (Kimpara,
Frongillo & Waggoner 2023). Convolved scores under observation error: Bröcker
& Smith (2007); Ferro (2017); Bessac & Naveau (2021). Noisy-channel learning:
Patrini et al. (2017); van Rooyen & Williamson (2018); surrogate scoring rules
(Liu, Wang & Chen 2022). Transform-properness: Allen, Ginsbourger & Ziegel
(2023, Prop. 4 published / Prop. 3 arXiv v1); Pic, Dombry, Naveau & Taillardat
(2025, Prop. 1) — deterministic injective transforms; Theorem 2 is the
Markov-kernel extension. The identity: Stam (1959); Barron (1986); Lyu (2009);
Song et al. (2021). Local scores: Hyvärinen (2005); Parry, Dawid & Lauritzen
(2012). Nearest mechanism neighbours, each missing a leg: Lang, Leutbecher &
Maciel (2025) — a Gaussian scale-ladder of proper scores as a *training loss*;
Dudík, Wang, Pennock & Rothschild (2021) — a multi-resolution *market* by
partition refinement, subsidized rather than budget-balanced; Lambert et al.
(2008, 2015) and the microprediction pool — self-funding cloud wagering at a
single scale. A full audit with verdicts is in the companion
[research note](../research/mollified-scoring-and-the-heat-ladder.md).

## 7. Open problems

1. **Fair rungs.** The finite-$m$ correction making each rung's expected score
   optimized by *sampling* from the belief (the log-score/KDE analog of
   Ferro's fair CRPS), and its interaction with the jitter. All results here
   are population-level; the finite-cloud game is not covered by the theorems.
2. **Endogenous bandwidth.** If the bandwidth is computed from the
   participant's own submission (Scott's rule on the cloud), the channel is no
   longer fixed and Theorem 2 does not apply; the participant controls both
   $\rho$ and $h(\rho)$. Characterize the optimal joint deviation, and whether
   any self-referential bandwidth rule preserves truthfulness.
3. **Optimal scale weights.** For wealth-concentration objectives, is there a
   closed-form optimal $w(t)$ — and does it recover the likelihood weighting
   of Song et al. (2021)?
4. **Anisotropic rungs.** Reference-covariance flows in place of isotropic
   heat, preserving rung-wise strict properness (cf. the
   [anisotropic sliced scores note](../research/anisotropic-sliced-scores.md)).
5. **The Kelly interpolation.** Between $b\to0$ (linear pot split,
   mode-seeking) and $b=1$ (log score, truthful), characterize the rung-wise
   optimal misreport as a function of the stake fraction.

## References

- Allen, S., Ginsbourger, D. & Ziegel, J. (2023). "Evaluating Forecasts for
  High-Impact Events Using Transformed Kernel Scores." *SIAM/ASA JUQ* 11(3),
  906–940.
- Barron, A. R. (1986). "Entropy and the Central Limit Theorem." *Ann. Probab.*
  14(1), 336–342.
- Bessac, J. & Naveau, P. (2021). "Forecast Score Distributions with Imperfect
  Observations." *ASCMO* 7, 53–71.
- Bröcker, J. (2012). "Evaluating Raw Ensembles with the Continuous Ranked
  Probability Score." *QJRMS* 138(667), 1611–1617.
- Bröcker, J. & Smith, L. A. (2007). "Scoring Probabilistic Forecasts: The
  Importance of Being Proper." *Weather and Forecasting* 22(2), 382–388.
- Candille, G. & Talagrand, O. (2008). "Impact of Observational Error on the
  Validation of Ensemble Prediction Systems." *QJRMS* 134(633), 959–971.
- Cotton, P. (2022). *Microprediction: Building an Open AI Network.* MIT Press.
- Dudík, M., Wang, X., Pennock, D. M. & Rothschild, D. M. (2021). "Log-time
  Prediction Markets for Interval Securities." *AAMAS*.
- Ferro, C. A. T. (2014). "Fair Scores for Ensemble Forecasts." *QJRMS*
  140(683), 1917–1923.
- Ferro, C. A. T. (2017). "Measuring Forecast Performance in the Presence of
  Observation Error." *QJRMS* 143(708), 2665–2676.
- Fricker, T. E., Ferro, C. A. T. & Stephenson, D. B. (2013). "Three
  Recommendations for Evaluating Climate Predictions." *Meteorol. Appl.*
  20(2), 246–255.
- Hyvärinen, A. (2005). "Estimation of Non-Normalized Statistical Models by
  Score Matching." *JMLR* 6, 695–709.
- Kimpara, D., Frongillo, R. & Waggoner, B. (2023). "Proper Losses for Discrete
  Generative Models." *ICML*, PMLR 202.
- Koehler, F., Heckett, A. & Risteski, A. (2023). "Statistical Efficiency of
  Score Matching: The View from Isoperimetry." *ICLR*.
- Krüger, F., Lerch, S., Thorarinsdottir, T. & Gneiting, T. (2021). "Predictive
  Inference Based on Markov Chain Monte Carlo Output." *Int. Stat. Rev.* 89(2),
  274–301.
- Lambert, N. S., Langford, J., Wortman, J., Chen, Y., Reeves, D. M., Shoham,
  Y. & Pennock, D. M. (2008). "Self-Financed Wagering Mechanisms for
  Forecasting." *EC*, 170–179; (2015) *J. Econ. Theory* 156, 389–416.
- Lang, S., Leutbecher, M. & Maciel, P. (2025). "A Multi-Scale Loss Formulation
  for Learning a Probabilistic Model with Proper Score Optimisation."
  arXiv:2506.10868.
- Liu, Y., Wang, J. & Chen, Y. (2022). "Surrogate Scoring Rules." *ACM TEAC*
  10(3).
- Lyu, S. (2009). "Interpretation and Generalization of Score Matching." *UAI*.
- Parry, M., Dawid, A. P. & Lauritzen, S. (2012). "Proper Local Scoring Rules."
  *Ann. Statist.* 40(1), 561–592.
- Patrini, G., Rozza, A., Menon, A. K., Nock, R. & Qu, L. (2017). "Making Deep
  Neural Networks Robust to Label Noise: A Loss Correction Approach." *CVPR*.
- Pic, R., Dombry, C., Naveau, P. & Taillardat, M. (2025). "Proper Scoring
  Rules for Multivariate Probabilistic Forecasts based on Aggregation and
  Transformation." *ASCMO* 11, 23–58.
- Saetra, Ø., Hersbach, H., Bidlot, J.-R. & Richardson, D. S. (2004). "Effects
  of Observation Errors on the Statistics for Ensemble Spread and Reliability."
  *Mon. Wea. Rev.* 132(6), 1487–1501.
- Song, Y., Durkan, C., Murray, I. & Ermon, S. (2021). "Maximum Likelihood
  Training of Score-Based Diffusion Models." *NeurIPS*.
- Song, Y. & Ermon, S. (2019). "Generative Modeling by Estimating Gradients of
  the Data Distribution." *NeurIPS*.
- Stam, A. J. (1959). "Some Inequalities Satisfied by the Quantities of
  Information of Fisher and Shannon." *Information and Control* 2(2), 101–112.
- Theis, L., van den Oord, A. & Bethge, M. (2016). "A Note on the Evaluation of
  Generative Models." *ICLR*.
- van Rooyen, B. & Williamson, R. C. (2018). "A Theory of Learning with
  Corrupted Labels." *JMLR* 18(228), 1–50.
- Wenliang, L. K. & Kanagawa, H. (2020). "Blindness of Score-Based Methods to
  Isolated Components and Mixing Proportions." arXiv:2008.10087.
- Zhang, M., Key, O., Hayes, P., Barber, D., Paige, B. & Briol, F.-X. (2022).
  "Towards Healing the Blindness of Score Matching." arXiv:2209.07396.
