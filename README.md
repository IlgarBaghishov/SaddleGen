# SaddleGen

Generative AI for **transition-state discovery** in periodic materials.

Given only a reactant geometry, SaddleGen proposes plausible saddle-point structures — no product state required. The target use case is reaction discovery for batteries, catalysts, and bulk materials, where current workflows (NEB, Dimer) require an a-priori guess of the reaction's end state.

## Status

Early-stage. No working code yet — the repository currently holds design documentation only. See [`CLAUDE.md`](CLAUDE.md) for the full methods specification.

## Approach (in brief)

- **Flow matching** over atomic Cartesian coordinates (straight-line Optimal Transport from a perturbed reactant to a saddle).
- **Multimodality via shell sampling.** Multiple saddles share a reactant; to avoid mode-averaging, flow trajectories start from points on a small sphere around the reactant. Different starting points route to different saddles via local-environment features.
- **Velocity field** = pretrained **UMA** (fairchem) as a frozen (or fine-tuned) backbone + a small equivariant `VelocityHead` trained on top.
- **Training data:** ~30 million (reactant, saddle, product) triplets from NEB + Dimer searches across Materials Project, ICSD, Alexandria, OC20, and OC22.

## Built on

- [fairchem](https://github.com/facebookresearch/fairchem) (Meta FAIR) — UMA backbone, `AtomicData`, PBC-aware graph building, ASE Calculator integration.
- [ASE](https://wiki.fysik.dtu.dk/ase/) — atomic simulation environment.
- [PyTorch](https://pytorch.org) + [HuggingFace `accelerate`](https://huggingface.co/docs/accelerate) — training loop, multi-GPU / multi-node scaling.

## Author

Ilgar Baghishov, Henkelman group, University of Texas at Austin.

## License

TBD.
