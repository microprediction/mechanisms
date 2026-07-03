# Scoring Point-Cloud Distributional Submissions

### The nearest-the-pin parimutuel, jittered outcomes, and the heat ladder

Peter Cotton · *Working draft v0.3* · 2026

---

## Abstract

A parimutuel over a continuum splits the pot in proportion to the probability
density each participant placed at the realised outcome. In practice the
submission is a cloud of Monte-Carlo samples, smoothed into a kernel density
estimate and scored at the pin. We give the mechanism and its incentive
theory, and show that the smoothing step is where the theory bites: scoring a
KDE at the raw outcome is improper, with the optimal cloud drawn from the
deconvolution of the forecaster's belief by the kernel (Theorem 1). Jittering
the outcome with the same kernel restores strict propriety, because
convolution by a kernel whose characteristic function has dense support is
injective on probability measures (Theorem 2). Repeating the repaired score
across smoothing scales decomposes the log-score edge, via de Bruijn's
identity, into Fisher-divergence payments for shape plus a coarse-scale
payment for mass (Theorem 3). In high dimensions the pool runs instead on
random one-dimensional projections, which recover the multivariate energy
score exactly. All incentive results are population-level, with the kernel
fixed in advance.

---

## 1. The problem

A classical parimutuel operates over a finite partition of outcomes. Bettors
stake on outcomes; the pot is divided among backers of the realised outcome in
proportion to stake; the implied probabilities are the pool fractions, and the
operator bears no risk. Modern forecasting is rarely categorical: the object
of interest is a full predictive distribution over a continuous, often
multivariate quantity, and the natural generalisation replaces "a ticket on
outcome $j$" with "probability mass placed near the point $z$".

Operationally, a contest accepts from each participant $i$ a cloud
$x^{(i)}_1,\dots,x^{(i)}_m\in\mathbb R^d$, smooths it into a density
$\hat q_i = \tfrac1m\sum_j \varphi_h(\cdot-x^{(i)}_j)$, and, when the outcome
$z$ is revealed, rewards $\log \hat q_i(z)$ or splits a pot in proportion to
$\hat q_i(z)$. This is the reward rule of
[monteprediction.com](https://monteprediction.com), described there as

> "a splitting of the pot in proportion to the density that you ascribe to the
> truth $z$, [which] also depends on the density that others ascribe to $z$,"

and it is also the standard evaluation protocol for sample-based generative
models. Not every contest smooths: in the MidOne contests at CrunchDAO
[@crunchdao_midone], participants supplied densities directly, through a
shared convention for density specifications [@cotton_density], and the
incentive analysed in §3 does not arise. This paper concerns cloud
submissions.

Two questions organise what follows. What does the pool built on this rule
look like, and is it truthful (§2)? And, more delicately: *what cloud should a
rational participant submit?* The answer to the second is *not* "samples from
your belief," and the failure has a closed form (§3), a one-line repair (§4),
and, once repaired, a multi-scale structure (§5).

Throughout, we idealise the cloud by its sampling law $\rho$ ($m\to\infty$;
finite $m$ is Open Problem 1), so the mechanism observes
$T_h\rho := \rho*\varphi_h$, where $\varphi_h$ is a density on $\mathbb R^d$,
symmetric, strictly positive (the Gaussian $N(0,h^2 I)$ is canonical). The
truth is $p^*$; all expectations below are assumed finite (for Gaussian
kernels and beliefs with finite second moments, $\log T_h\rho$ has
at-worst-quadratic tails, so this is mild).

## 2. The pool: a nearest-the-pin parimutuel

Why a *density* split, and not some other functional? Because the parimutuel
already has a sharp incentive theory in the discrete case, and it is exactly
the one we want. Consider $n$ outcomes with true probabilities $p_k$, and a
parimutuel in which a player allocates a unit stake as a distribution
$b = (b_1, \dots, b_n)$ over outcomes. If the rest of the pool's stake
fractions are $r_k$ and the player is small, a unit bet on $k$ returns $1/r_k$
when $k$ occurs. A player maximising the expected log growth of wealth solves

$$\max_{b \in \Delta}\; \sum_k p_k \log\frac{b_k}{r_k},$$

whose unique maximiser is $b_k = p_k$, *bet your beliefs*
[@kelly1956newinterpretation; @breiman1961optimal]. The growth rate at the
optimum is the Kullback–Leibler divergence $D(p \,\|\, r)$: the player profits
exactly to the extent their belief beats the crowd's implied distribution.
Log-wealth maximisation and truthful reporting coincide, and the quantity
being maximised is the logarithmic score of the report against the realised
outcome, net of the crowd.

**Mechanism.** Each of $n$ participants holds wealth $W_i$ and submits a
predictive density $q_i$ over $\mathbb{R}^d$. Each risks a stake $s_i = b\,W_i$
for a fixed fraction $b \in (0,1)$. The outcome $z$ is revealed. The collected
pot $S = \sum_i s_i$ is redistributed in proportion to $s_i\, q_i(z)$, stake
times the density placed at $z$, so participant $i$'s wealth change is

$$\boxed{\;\Delta W_i \;=\; S\,\frac{s_i\,q_i(z)}{\sum_j s_j\,q_j(z)} \;-\; s_i\;}
\tag{NTP}$$

**Self-funding.** $\sum_i \Delta W_i = S - S = 0$: the pool is a pure transfer,
the operator bears no risk, exactly as in the racetrack tote. (Implemented and
tested in [`pot_split`](../mechanisms/nearest_the_pin.py).)

**Truthfulness, at the density level.** Fixing the field's reports and stakes,
the aggregate density at $z$ is $Q(z) = \sum_j s_j q_j(z)$. A small
participant's expected log-wealth growth from reporting $q$ is, to first order
in $b$,

$$\mathbb{E}_{z \sim p}\!\left[\log\!\Big(1 + b\big(\tfrac{S}{s_i}\tfrac{s_i q(z)}{Q(z)} - 1\big)\Big)\right]
 \;\approx\; b\,\Big(\mathbb{E}_{z\sim p}\!\big[\tfrac{S\,q(z)}{Q(z)}\big] - 1\Big),$$

and the report $q$ maximising $\mathbb{E}_{z\sim p}[\,q(z)/Q(z)\,]$ subject to
$\int q = 1$ is, by the same Gibbs argument as the discrete case, $q = p$: the
true density. We verify the incentive numerically in
[`test_nearest_the_pin.py`](../tests/test_nearest_the_pin.py): a truthful
reporter out-grows a biased one against a truthful field.

**Relationship to other mechanisms.**

- It is the continuous, density-weighted limit of the discrete parimutuel
  above, and a density-pot-split generalisation of Pennock's dynamic
  parimutuel market (DPM) [@pennock2004dynamic]: the DPM prices shares on
  discrete outcomes via a cost function; NTP prices density mass on a
  continuum.
- Its truthfulness rests on the logarithmic score / log-wealth growth, the
  same object that, applied *sequentially*, gives Hanson's LMSR
  [@hanson2007logarithmic]. NTP is the *pooled* reading; LMSR is the
  *sequential* reading.
- The score it implicitly applies to a sample cloud is a strictly proper
  scoring rule for the predictive density; §6 makes this precise and connects
  it to the energy score.

The truthfulness claim above concerns the reported *density* $q$. In practice
$q$ is constructed from a sample cloud by KDE, so the report space is the
cloud and the mechanism applies $q = \rho * \varphi_h$. The truthful object
and the submitted object come apart, and the next two sections are about the
gap.

## 3. The deconvolution incentive

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
zero. (iii) Gaussians: convolution adds covariances; the KL between
$N(0,\tau^2)$ and $N(0,\tau^2+h^2)$ is the stated expression; Taylor expansion
gives $h^4/(4\tau^4)$. $\blacksquare$

**Remarks.** (a) When the deconvolution fails to exist ($\Sigma-h^2I\not\succeq
0$, or a rough $p^*$), the optimum is the KL projection of $p^*$ onto
$T_h\mathcal P$ and can degenerate toward atomic clouds — exactly the
point-mass exploit of @theis2016note, who observed the improperness (without
the deconvolution characterization). (b) The gap is fourth order in $h$, which
is why the flaw survives casual inspection — but it is a *slope*, and any
optimising submitter, human or fitted, walks down it. (c) The phenomenon is
the log-score/bandwidth sibling of the "fair scores" findings for the ensemble
CRPS [@fricker2013three; @ferro2014fair], where the analogous cheat is derived
along the ensemble-size axis.

## 4. The repair: jitter the pin

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
injective (nonvanishing characteristic functions); the uniform kernel is also
fine (sinc zeros are isolated, hence the nonvanishing set is dense);
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
$T_K\mathcal P$, then $S_K$ is strictly proper on $\mathcal P$ **iff**
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
The incentives of participant-endogenous bandwidths are an open problem
(§10).

**Attribution.** The properness of the convolved score against a
noised outcome is not new: @brocker2007proper [§5] prove it for a general
observation-noise channel, and @ferro2017observation [Prop. 3] works out exactly the
white-noise/Gaussian case. What both left open is strictness. Bröcker &
Smith, verbatim: "If $S$ is strictly proper though, $\bar S$ is not
necessarily strictly proper, because if $\bar q(z)=\bar p(z)$, this does not
necessarily mean equality of $p(x)$ and $q(x)$." Lemma 1 closes exactly that
gap. The discrete-outcome analog is the label-noise literature's forward loss
correction with an invertible transition matrix [@patrini2017making, Thm. 2; @vanrooyen2018theory]. Verification treats outcome noise as a
nuisance to be endured, and Ferro explicitly doubts the "efficacy" of
perturbing observations; read as mechanism design, jitter matched to the
mechanism's own smoothing is what makes the point-cloud game strictly
proper.

## 5. The heat ladder

Let $p_t := p * N(0,tI)$, the heat flow ($\partial_t p_t=\tfrac12\Delta p_t$).
To keep one time variable, write $u = h^2 + t$ for the *total* smoothing scale:
the rung at flow time $t$ is the §4 score with kernel
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

**Proof.** Write $p=p^*_t$, $q=\rho_t$, both solutions of the heat
equation. Then

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

The differential identity is de Bruijn's, in relative form [@stam1959some; @barron1986entropy; @lyu2009interpretation]; its integral form prices the likelihood of diffusion models
[@song2021maximum]. Note the bandwidth floor does real work: at $t=0$ an empirical
cloud has no density and the identity is vacuous; every rung the mechanism
actually runs starts at $t\ge h^2$, where everything is smooth.

**The heat-ladder pool.** Fix scales $0=t_0<t_1<\dots<t_K=T$ and weights
$w_k\ge0$. Participant $i$ stakes $s_i$ and submits one cloud. Two payment
schedules must be distinguished, because they buy different things.

*Level schedule.* Rung $k$ splits $w_k\sum_i s_i$ by a budget-balanced rule
driven by the rung score $S^{(k)}_i := S_{\sqrt{h^2+t_k}}(\rho_i,z)$ — the
stake-weighted additive form
$\Delta W_i^{(k)}=w_k\,s_i\,(S_i^{(k)}-\bar S^{(k)})$ with $\bar S^{(k)}$ the
stake-weighted mean [@lambert2008selffinanced; @lambert2015axiomatic]. Each
rung is strictly proper (Theorem 2), so any nonnegative-weighted sum is. But
the expected-regret decomposition carries *cumulative* weights on the Fisher
bands, not the rung weights themselves: writing
$\mathrm{KL}_k := \mathrm{KL}(p^*_{t_k}\Vert \rho_{t_k})$,

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
risk-neutral limit; the multiplicative pot split of §2 requires a log-wealth
(Kelly) model stated separately. (iii) Band payments purchase (integrated)
Fisher divergence, which is Hyvärinen-scored shape, invariant to the
normalization of the submission; the top level pays
$\mathrm{KL}(p^*_T\Vert\rho_T)$, which at mode-connecting scales carries the
between-mode mass that score matching is blind to [@wenliang2020blindness;
@zhang2022healing; @koehler2023statistical]. Under the level schedule the band
weights are the cumulative sums above.*

Practicalities: rung scores are positively correlated (one cloud,
re-smoothed), so discriminative value concentrates in a few well-separated
scales; $K$ of order 3–5, geometrically spaced, mirrors the noise ladders of
annealed score matching [@song2019generative].

## 6. The projection route

**The high-dimensional obstruction.** In $d = 11$ dimensions, recovering a
participant's density $q_i(z)$ from a finite sample cloud by KDE is fragile:
the bandwidth, the curse of dimensionality, and the heavy tails of financial
returns all bite. monteprediction's contest has participants submit *a
million* scenarios precisely because dense coverage is needed to pin down
$q_i(z)$. And the incentive analysis of §3 gets *worse* with dimension:
Scott's-rule bandwidths grow toward the signal scale as $d$ rises, so the
optimal shave $h^2$ becomes first-order rather than a subtle correction.

**Slicing.** The projection (sliced) version sidesteps the $d$-dimensional
density. Draw random unit directions $u \in S^{d-1}$, project every
participant's cloud and the outcome onto each $u$, and score the resulting
*one-dimensional* forecasts, where density estimation, the CRPS, and quantiles
are all easy and robust; then average over directions. This is exactly aligned
with how the energy score decomposes. For a uniformly random $u$ on the sphere
and any $x \in \mathbb{R}^d$,

$$\mathbb{E}_u\,|\langle u, x\rangle| \;=\; c_d\,\|x\|,
\qquad c_d = \frac{\Gamma(d/2)}{\sqrt\pi\,\Gamma((d+1)/2)},$$

so $\|x\| = c_d^{-1}\,\mathbb{E}_u|\langle u, x\rangle|$. Substituting into the
energy score $\mathrm{ES}(P, y) = \mathbb{E}\|X - y\| - \tfrac12\mathbb{E}\|X-X'\|$
gives the projection identity

$$\boxed{\;\mathrm{ES}(P, y) \;=\; c_d^{-1}\;\mathbb{E}_u\big[\,\mathrm{CRPS}(P_u,\ \langle u, y\rangle)\,\big]\;}
\tag{PROJ}$$

where $P_u$ is the law of the projected sample $\langle u, X\rangle$. The
multivariate energy score *is* the average over random directions of the
one-dimensional CRPS, and the energy score is strictly proper for the full
distribution [@gneiting2007strictly]. The sliced quantity is therefore a
proper score that needs only 1-D evaluations. We verify (PROJ) numerically: in
[`energy_score_via_projection`](../mechanisms/nearest_the_pin.py) the sliced
estimate matches the exact multivariate energy score within a few percent at a
few thousand directions, and equals the CRPS exactly in 1-D
([`test_nearest_the_pin.py`](../tests/test_nearest_the_pin.py)).

**The projection-scored nearest-the-pin.** Replace $q_i(z)$ in (NTP) by a
projection-based skill score: for each direction $u$, the 1-D CRPS (or 1-D
density) of participant $i$'s projected cloud at $\langle u, z\rangle$; average
over $u$ to get a per-participant score; split the pot in proportion to it. The
pool keeps its self-funding and truthfulness properties (the energy score is
proper, so truthful reporting is still optimal) while becoming computable and
stable in high dimensions. This is, in spirit, the projection version at
[monteprediction](https://www.monteprediction.com): score the
eleven-dimensional cloud through its one-dimensional shadows. Because no
kernel smoothing is applied, the deconvolution incentive of §3 does not arise;
the finite-$m$ analog is the fairness correction of @ferro2014fair, applied
slice-wise in one dimension.

**Link to the random-projections literature.** Slicing a high-dimensional
problem into random 1-D projections is a recurring, theoretically-backed
device:

- Johnson–Lindenstrauss [@johnson1984extensions]: random projections
  approximately preserve pairwise Euclidean distances, which is precisely the
  quantity the energy score / energy distance is built from, so a modest
  number of directions preserves the score.
- Sliced Wasserstein distances [@rabin2012wasserstein; @bonneel2015sliced]:
  average 1-D optimal-transport costs over random projections, a now-standard,
  cheap surrogate for the multivariate Wasserstein distance — the
  optimal-transport cousin of (PROJ).
- Sliced score matching [@song2020sliced]: estimate high-dimensional score
  functions through random projections, for the same computational reasons.
- Energy distance [@szekely2013energy] is itself an integral of squared
  characteristic-function differences and, via the identity above, of absolute
  projected differences; the projection representation is intrinsic, not a
  heuristic.

The projection version is not an approximation bolted onto the mechanism; it
is the *native* high-dimensional form of a density-pot-split parimutuel whose
proper score (the energy score) is, by construction, an average over random
projections.

## 7. Two routes to high-dimensional scoring

Step back from the pool to the statistical problem underneath: how do you
score a joint distributional forecast in $\mathbb{R}^d$ when $d$ is large
relative to the data you can condition on? This is exactly the regime
($p > n$) in which the naïve held-out Gaussian likelihood, the density-based
score, becomes unreliable, because the estimated covariance is rank-deficient
and its inverse is nonsense.

The portfolio/spatial-statistics literature answers this from the density
side. *Two Sides of Schur Damping* [@cotton2026schur] and the underlying Schur
complementary allocation [@cotton2024schur] observe that a Gaussian density
factorises through a Vecchia / conditional pseudo-likelihood
[@vecchia1988estimation]
$\prod_k \mathcal{N}(y_k; b_k^\top y_c, S_k)$ whose conditional covariances
$S_k$ are *Schur complements*, and that the reliable score in the undersampled
regime is a damped version of this factorisation, the Schur pseudo-likelihood,
with a closed-form James–Stein reliability damping $\gamma^\star$. When the
raw joint density is untrustworthy, score it through a structured,
positive-definiteness-preserving factorisation rather than the full inverse
covariance. (This is the basis of the `precise` library's covariance
assessors; see the [schur](https://github.com/microprediction/schur) project.)

The nearest-the-pin parimutuel needs precisely such a score: it must turn each
participant's joint forecast into a number at $z$. There are then two routes,
and they are the two sides of the same coin:

| route | how the joint forecast is scored | regime it suits |
|-------|----------------------------------|-----------------|
| density | structured / damped joint density — Schur pseudo-likelihood, Vecchia factorisation | a parametric or covariance-shaped forecast; $p > n$ handled by damping |
| projection | average 1-D CRPS over random directions — the sliced energy score (PROJ) | a free-form sample cloud; high $d$ handled by slicing |

The organising claim of the high-dimensional story: the density route
(Schur/Vecchia) and the projection route (energy/sliced) are alternative ways
to make a joint forecast scoreable in high dimensions, and the pool can be run
with either. A speculative synthesis, flagged as conjecture, is a
*Schur-damped* projection score, in which the directions $u$ are not isotropic
but shaped by a damped estimate of a reference covariance (project more often
along the well-estimated directions), interpolating between the two columns
with a single reliability dial $\gamma$ exactly as in the Schur work. Its
analysis is open, and a properness boundary is worked out in the
[anisotropic sliced scores note](../research/anisotropic-sliced-scores.md):
strict properness survives only if the anisotropy comes from a reference
covariance fixed in advance, not from the forecast under test.

## 8. Microprediction, and a historical note

The nearest-the-pin parimutuel is one of the mechanisms of the microprediction
vision [@cotton2022microprediction]: a web-scale network of autonomous
forecasters continuously submitting *distributional* predictions and being
paid by self-funding, truth-eliciting pools. Two further pieces close the
loop:

- Calibration via Z-streams. The crowd's aggregate density $Q$ induces, for a
  scalar quantity, $z$-scores $\Phi^{-1}(F(x))$; if the market is calibrated
  these are standard normal over time, and any departure (fat tails, skew,
  autocorrelation) is an exploitable, self-correcting anomaly. The pool's
  payouts push the aggregate back toward calibration.
- Aggregation. The crowd density $Q$ is itself a forecast, a wealth-weighted
  pool of the participants' densities: a (log-)opinion pool whose weights are
  endogenously set by past accuracy.

The microprediction platform, launched in 2019, added a small amount of noise
to submissions and to the ground truth before settling its cloud-based
lotteries. This was *merely intuitive*, a fairness-and-anti-gaming instinct
about discreteness and ties, with no incentive theorem attached; the platform
paper recorded the practice in one line and moved on. The verification
literature, meanwhile, treated outcome noise as a defect: something to be
modelled away [@saetra2004effects; @candille2008impact], with
@ferro2017observation doubting the value of perturbing observations.

Theorem 2 is the missing theorem: symmetrized jitter is what makes a
smoothed-submission game strictly proper for the submitted law. The theory
also sharpens the intuition into design guidance it could not supply on its
own: the jitter distribution matters. Kernels whose characteristic function
has dense support (Gaussian, Laplace, even uniform) preserve strict
properness; band-limited kernels do not (Lemma 1). Intuition chose jitter; the
theorem chooses *which* jitter.

## 9. Related work

Improperness of sample-based scoring: @theis2016note, who call the KDE
log-likelihood "an improper scoring function"; the fair-scores line for the ensemble CRPS
[@brocker2012raw; @fricker2013three; @ferro2014fair], with the over-dispersion
direction for fitted ensembles in @siegert2019ensemble; the estimator view of
KDE log scores [@kruger2021predictive]; ensemble dressing
[@brocker2008dressing]; discrete impossibility and randomized repair
[@kimpara2023proper]; sample elicitation via variational divergences
[@wei2021sample]. Convolved scores under observation error:
[@brocker2007proper; @ferro2017observation; @bessac2021forecast].
Noisy-channel learning: [@patrini2017making; @vanrooyen2018theory]; surrogate
scoring rules [@liu2022surrogate]. Transform-properness: @allen2023transformed
[Prop. 4 published, Prop. 3 in arXiv v1] and @pic2025proper [Prop. 1] cover
deterministic injective transforms; Theorem 2 is the Markov-kernel extension.
The identity: [@stam1959some; @barron1986entropy; @lyu2009interpretation;
@song2021maximum], with the denoising connection in @vincent2011connection.
Local scores: [@hyvarinen2005score; @parry2012proper]. Nearest mechanism
neighbours, each missing a leg: @lang2025multiscale, a Gaussian scale-ladder
of proper scores as a training loss; @dudik2021logtime, a multi-resolution
market by partition refinement, subsidized rather than budget-balanced; and
self-funding cloud wagering at a single scale [@lambert2008selffinanced;
@lambert2015axiomatic]. A full audit with verdicts is in the companion
[research note](../research/mollified-scoring-and-the-heat-ladder.md).

## 10. Open problems

1. **Fair rungs.** The finite-$m$ correction making each rung's expected score
   optimized by *sampling* from the belief, the log-score/KDE analog of
   the fair CRPS of @ferro2014fair, and its interaction with the jitter. All
   results here are population-level; the finite-cloud game is not covered by
   the theorems.
2. **Endogenous bandwidth.** If the bandwidth is computed from the
   participant's own submission (Scott's rule on the cloud), the channel is no
   longer fixed and Theorem 2 does not apply; the participant controls both
   $\rho$ and $h(\rho)$. Characterize the optimal joint deviation, and whether
   any self-referential bandwidth rule preserves truthfulness.
3. **Finite-$b$ and finite-$n$ truthfulness.** §2's argument is first-order in
   the wealth fraction $b$ and assumes a small participant. What is the exact
   equilibrium for finite $b$ and finitely many strategic participants?
4. **Choice of score.** Density pot-split (jittered, §4) vs. projection
   (sliced energy, §6) are both proper but reward different aspects of a
   forecast. Which yields better calibration and faster wealth concentration
   on skilled forecasters?
5. **Optimal scale weights.** For wealth-concentration objectives, is there a
   closed-form optimal $w(t)$ in the ladder of §5, and does it recover the
   likelihood weighting of @song2021maximum?
6. **Schur-damped projections.** Does anisotropic, covariance-shaped slicing
   with a reliability dial $\gamma$ (§7) dominate isotropic slicing in the
   $p>n$ regime, and does it inherit the closed-form $\gamma^\star$?
7. **Variance of the sliced estimator.** How many directions are needed for
   the sliced score of §6 to rank participants correctly, as a function of $d$
   and the cloud size? (A Johnson–Lindenstrauss-style bound.)

## References

::: {#refs}
:::

*Implementation: [`mechanisms/nearest_the_pin.py`](../mechanisms/nearest_the_pin.py).
Tests (self-funding, truthfulness, the projection identity, and Theorem 1 in
both directions):
[`tests/test_nearest_the_pin.py`](../tests/test_nearest_the_pin.py).*
