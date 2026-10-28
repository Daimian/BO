# BO

Bayesian-optimization-driven active learning for the automated construction of the effective
Hamiltonian of BaTiO₃.

Parameterizing an effective Hamiltonian has traditionally required hand-built distorted structures
that satisfy symmetry constraints. Here the distorted configurations are proposed by Bayesian
optimization (TuRBO trust regions with BoTorch), evaluated with VASP, and the Hamiltonian
coefficients are refitted by linear regression after each step. Fewer than 30 distorted
configurations were enough to fit the Hamiltonian for BaTiO₃, with Monte Carlo transition temperatures within 10% of experiment.

This is the research code used for the paper below, kept as it was run on the cluster.

## Paper

M. Dai et al., *Active learning-based automated construction of Hamiltonian for structural phase
transitions: a case study on BaTiO₃*, J. Phys.: Condens. Matter (2025).
[doi:10.1088/1361-648X/ad882a](https://doi.org/10.1088/1361-648X/ad882a)

## Contents

| Path | Content |
| --- | --- |
| `main.py` | active-learning loop: refit coefficients → BO proposes a configuration → VASP evaluates it |
| `bo.py` | TuRBO state, candidate generation and acquisition (UCB) |
| `vasp.py` | wraps a VASP calculation of a distorted BaTiO₃ cell as a BoTorch test problem |
| `BaTiO3/` | reference structure (`POSCAR.ori`), VASP input files, and the recorded run in `Data/` |

## Running

```bash
conda create --name BO python=3.11.5
conda activate BO
conda install -c pytorch pytorch torchvision torchaudio
conda install -c conda-forge botorch pymatgen
```

The scripts submit VASP calculations through Slurm (`sbatch job-vasp.sh`) and expect a cluster
environment: set `root_path`, `path` and `ssh_root` at the top of `main.py` to your own working directories before running `python main.py`.

## License

[MIT](LICENSE)
