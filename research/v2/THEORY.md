# Formal targets and design contracts

These propositions are basic consequences of the stated assumptions. They are useful for specifying and testing the system, not presented as independently novel mathematical results.

## 1. Two estimands, never silently interchanged

Let D be an ordered set of source blocks, q a question, A an answer model, and C a compressor. For a fixed answer token sequence a, define

`g_C(D; a) = log p_A(a | q, C(q,D))`

`tau_C(i; a) = g_C(D; a) - g_C(D without i; a)`.

**Own-pipeline audit fidelity** compares an estimator to tau_C for that same C and fixed a. **Reference-effect preservation** compares tau_C to tau_raw, evaluating both on the same token sequence a from the raw-context system. The latter is an empirical property of a changed answering system, not guaranteed by correct caching. Quality is independently evaluated against benchmark answers.

## 2. Fixed-routing compression

Choose a source universe U, stable source IDs, fixed routing pi, fixed component budgets b_g, tokenizer framing, immutable compressor weights, and decoding policy. These are explicit policy inputs shared by the factual and intervened branches. Budgets are not redistributed after deletion. Routing in the initial prototype is source-ID based; a learned extension must assign each source independently, using a fixed trained map rather than refitting clusters after deletion.

For each component g, compute `m_g = c_g(q, {d_i : pi(i)=g})`. The memory is a token-space concatenation of independently framed, nonempty components. An empty component emits no content or frame. The answerer always reruns on the resulting memory; its hidden states after changed text are not assumed reusable.

### Proposition A: source-local deletion commutation

If every component contains exactly one source and c_g is deterministic under the declared component seed, deleting source i removes only its framed component. The token stream obtained from cached unaffected components is identical to recomputing C on the intervened source set under the same U, pi, and b_g.

**Proof.** Every remaining source has identical content, q, source ID, component budget, framing, seed, and model parameters in both branches. Therefore each remaining component has identical tokens. Their stable order is unchanged. The removed component is empty in both constructions. Concatenation gives identical input tokens. Applying the same deterministic answer scorer gives identical log likelihoods and deletion effects. For nondeterministic numerical execution, empirical receipt and scoring checks are required rather than an unconditional bitwise claim.

### Proposition B: grouped incremental replay

For a changed source set S with fixed routing, only components in `pi(S)` need recomputation. If routing is a pointwise content-dependent map, replacements can affect both the old and new assigned components; those memberships must be updated explicitly.

**Proof.** Every component outside the affected union has identical members and contents. Its inputs and decoding policy are identical, so the argument of Proposition A applies to that component. Recompute all affected components and concatenate in stable component order. This equals full recomputation. An empty affected component requires no model generation.

The prototype implements deletion and same-ID replacements with fixed ID routing. It does not implement learned content-dependent routing or claim a speedup over an optimized implementation of the same incremental dependency graph.

### Failure conditions

Global reranking, cross-source-dependent routing, budget redistribution, source renumbering, a changed tokenizer, changed compressor weights, or a shared random stream advanced in different orders can invalidate reuse. A frozen factual plan that depends on D cannot be called an exogenous constant unless freezing it is explicitly part of the intervention estimand. The implementation fingerprints policy and source inputs and validates independently recomputed branches.

## 3. Factual accuracy cannot certify effect preservation

For a binary answer event, let the raw and compressed systems both give event probability 0.9 on factual D. On D without i, let the raw system give 0.1 and the compressed system 0.9. Factual distributions agree exactly, but probability intervention effects are 0.8 and 0. This construction demonstrates that factual agreement alone does not identify counterfactual agreement. It does not show that every compressor or attribution method must fail.

## 4. A classical transfer bound

Suppose the full continuation distributions obey `KL(P_D || Q_C(D)) <= epsilon_0` and `KL(P_D-i || Q_C(D-i)) <= epsilon_i`. For any fixed answer-prefix event E, Pinsker's inequality and the triangle inequality imply

`|tau_raw^prob(i;E) - tau_C^prob(i;E)| <= sqrt(epsilon_0/2) + sqrt(epsilon_i/2)`.

**Proof.** Each event probability difference is at most its total-variation distance, bounded by the corresponding Pinsker term. Subtract the two intervention contrasts and apply the triangle inequality. This bound concerns probability contrasts, not unbounded log-probability contrasts. Measuring KL on a few teacher-forced prefixes is not a certificate of the assumed full-continuation KL bounds.

## 5. Candidate learned objective

A source-separable soft compressor emits a fixed number of slots per source from a frozen HF encoder, learned attention pooling, and a learned projection into the frozen answerer's embedding space. These are standard building blocks. Source-local architecture provides the caching contract; it does not supply reference fidelity for free.

For a fixed answer span and paired factual/intervened teacher and student log probabilities, use

`L = L_QA + lambda_f * L_factual_distill + lambda_cf * mean_i[(tau_student(i)-tau_teacher(i))^2]`.

The implementation normalizes span log probabilities by target length during the canary. The factual-only control has identical architecture, initialization, example order, teacher computation, and student counterfactual forward passes, with lambda_cf=0. This isolates the objective rather than extra model capacity or additional perturbation compute. A separate unpaired perturbation-distillation baseline is required for the eventual paper.

The initial training run is a finite-gradient/optimization canary, not a claim of held-out improvement. The candidate objective needs comparison on untouched questions, multiple seeds, model families, and a second natural benchmark before any paper-level efficacy claim.
