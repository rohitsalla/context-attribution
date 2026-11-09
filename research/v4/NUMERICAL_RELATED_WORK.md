# Numerical controls: prior work and claim boundary

Primary sources checked 8 November 2026. Numerical sensitivity is not a new contribution claimed by this project.

- [PyTorch 2.10 numerical-accuracy notes](https://docs.pytorch.org/docs/2.10/notes/numerical_accuracy.html) explain why mathematically equivalent computations need not be bitwise identical, including batched/sliced computation and reduced precision. Our tolerances are experimental validation criteria, not guarantees from PyTorch.
- [Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference](https://arxiv.org/abs/2506.09501), updated October 2025, studies precision and system-configuration effects and proposes LayerCast. Do not present inference precision sensitivity or FP32 computation as an invention here.
- [Same Probe, Different Numbers](https://arxiv.org/abs/2609.31796) distinguishes changed-token effects from arithmetic changes in probe transfer. Its probes are largely stable; cite its actual findings rather than using its title to imply general interpretability failure. Our statistic is a conditional source-deletion response, not an activation-probe verdict.
- [BlockLock](https://arxiv.org/abs/2611.06977), submitted 4 November 2026, fixes protected-request batch geometry under a stated deployment contract. Shape/geometry control and bitwise restoration have direct precedent. Its reported scope is not a guarantee for our H200 configuration or training gradients.

The project-specific questions are whether a compressed representation transfers a declared raw-model intervention response, and which controls make that comparison interpretable. A correct repeat of the same program input is different from validating that two intended intervention constructions have equivalent attention connectivity and sufficiently close arithmetic results. Both requirements belong in the experiment contract.

Native compact-position deletion results remain measurements of the specified executed implementation. The failed positional probe cannot yet support a semantic mechanism claim. Explicit CPU mask-construction checks match on two layouts (16 and 1,500 tokens); the selected full-model FP32 CPU check and full CUDA precision diagnostic are separate experiments. A CPU pass would not certify CUDA numerical behavior or all questions.
