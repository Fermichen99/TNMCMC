# Tensor Network Markov Chain Monte Carlo: Efficient Sampling of Three-Dimensional Spin Glasses and Beyond

[![Language](https://img.shields.io/badge/Language-Julia-blue.svg)](https://julialang.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains a Julia implementation for simulating the 3D Edwards-Anderson (EA) spin glass model. The simulation is powered by the Tensor Network Monte Carlo (TNMCMC) algorithm, a sophisticated method for studying complex statistical physics systems.

## Overview

Spin glasses are a classic and challenging subject in condensed matter physics, characterized by quenched disorder and frustration. This project investigates the properties of the 3D EA model by leveraging TNMCMC, a powerful hybrid algorithm that combines the efficient state representation of Tensor Networks with the robust sampling capabilities of Monte Carlo methods.

The code is designed to be configurable and reproducible, allowing for systematic studies of various physical parameters across multiple disorder realizations.

## Features

-   **Model**: 3D Edwards-Anderson (EA) Ising spin glass.
-   **Algorithm**: An efficient Tensor Network Monte Carlo (TNMCMC) update scheme.
-   **Configurability**: Easily configure all simulation parameters (lattice size, temperature, disorder, etc.) via command-line arguments.
-   **Reproducibility**: Support for random number seeds to ensure reproducible simulation results.

## Installation

This project is written in the Julia programming language.

1.  **Install Julia**: If you do not have Julia installed, download and install the latest stable version from the [official Julia website](https://julialang.org/downloads/).

2.  **Clone the Repository**:
    ```bash
    git clone https://github.com/Fermichen99/TNMCMC.git
    ```

3.  **Install Dependencies**: The project relies on several Julia packages. The modern way to handle dependencies in Julia is through a `Project.toml` file. You can create one and instantiate the environment.

    **Standard Libraries (No installation needed):**
    The following dependencies are part of Julia's standard library and do not need to be added manually:
    - `Dates`
    - `Logging`
    - `Base.Threads`
    - `LinearAlgebra`

    **External Packages (Installation required):**
    You need to add `JLD2` and `OMEinsum`. In the Julia REPL, navigate to the project directory and use the package manager:
    ```julia
    import Pkg
    Pkg.add("FileIO")
    Pkg.add("Random")
    Pkg.add("JLD2")
    Pkg.add("OMEinsum")
    Pkg.add("ArgParse") # Likely needed for command-line parsing
    ```

## Usage

The simulation is launched from the main script `3Drunargs.jl`. All parameters are passed as command-line arguments.

### Example Command

```bash
julia 3Drunargs.jl --Type FBC --P 0.5 --zchunk 1 --ychunk 4 --xchunk 4 --Lz 4 --L 4 --Dimension 3 --Beta 1.5 --chi 4 --Ndisorder 10 --Nsample 100 --Ntherm 100 --step 1 --Nreplic 2 --seed 1 --verbose 1
```

### Command-Line Arguments

| Argument | Description | Example Value |
| :--- | :--- | :--- |
| `--Type` | Type of boundary conditions. `FBC` for Free Boundary Conditions. | `FBC` |
| `--P` | Disorder probability. The coupling $J_{ij}$ is set to `-1` with probability `P`. | `0.5` |
| `--zchunk` | The size of the block updated by TNMCMC in the Z-direction. | `1` |
| `--ychunk` | The size of the block updated by TNMCMC in the Y-direction. | `4` |
| `--xchunk` | The size of the block updated by TNMCMC in the X-direction. | `4` |
| `--Lz` | The size of the lattice in the Z-direction. | `4` |
| `--L` | The size of the lattice in the X and Y directions (assumes $L_x = L_y = L$). | `4` |
| `--Dimension` | The dimension of the model. | `3` |
| `--Beta` | The inverse temperature, $\beta = 1/T$. | `1.5` |
| `--chi` | The tensor network bond dimension used for truncation. | `4` |
| `--Ndisorder` | The total number of disorder realizations to simulate. | `10` |
| `--Nsample` | The number of Monte Carlo samples to collect after thermalization. | `100` |
| `--Ntherm` | The number of Monte Carlo thermalization steps for each realization. | `100` |
| `--step` | The sampling numbers of each block update. | `1` |
| `--Nreplic` | The number of replicas used in Replica. | `2` |
| `--seed` | The seed for the random number generator for reproducibility. | `1` |
| `--verbose` | Whether to print detailed information during simulation (1 for print). | `1` |


## File Structure

```
.
├── 3Drunargs.jl      # Main script, entry point for simulations, parses arguments.
├── Simu.jl           # Controls the main simulation workflow.
├── Para.jl           # Defines and manages simulation parameters.
├── Spin.jl           # Functions related to spin configurations.
├── Metro.jl          # Implements the core Metropolis-Hastings sampling algorithm.
├── Tensor.jl         # Implements the core TNMCMC sampling algorithm.
├── Stat.jl           # Module for statistical analysis and calculation of observables.
├── DicDat.jl         # Data type.
```

## Methodology

The core of this project is the **Tensor Network Monte Carlo (TNMCMC)** algorithm. This is a hybrid method that leverages the strengths of two fields:
-   **Tensor Networks (TNs)** are used to globally update the model, effectively overcoming the critical slowing down found in traditional local-update Monte Carlo methods.
-   **Monte Carlo (MC)** methods are used to sample the vast configuration space of the spin glass.

This combination allows for a more powerful investigation of the low-temperature physics of frustrated systems like the 3D EA model.

## Output Structure

The simulation generates two main directories in the location where the script is executed: `spinwbond` for storing the disorder configurations, and `dat` for storing the simulation results.

---

### 1\. Disorder Configurations (`spinwbond`)

This directory stores the generated random bond couplings ($J_{ij}$) for each disorder realization, allowing for reproducibility and re-analysis.

* **Directory Path:**

    ```
    spinwbond/L=<L>/seed=<seed>/
    ```

* **Example:**

    ```
    spinwbond/L=4/seed=1/
    ```

* **File Format:**
    Files are named `bond_L<L>_n<idis>.jld2`, where `<L>` is the lattice size and `<idis>` is the disorder realization index (from 1 to `Ndisorder`).

* **Data Structure:** Each `.jld2` file contains a single 4D array representing the bonds. The structure is `bond[z, y, x, d]`, where `d` is an index from 1 to 6 corresponding to the bond direction relative to the site `(z, y, x)`.

* **Bond Direction Mapping:** The 6 bond directions stored at the 4th index correspond to the connection between site `(z,y,x)` and its neighbours as follows:

    1.  `bond[z, y, x, 1]` $\leftrightarrow$ Bond with site `(z-1, y, x)`
    2.  `bond[z, y, x, 2]` $\leftrightarrow$ Bond with site `(z, y, x-1)`
    3.  `bond[z, y, x, 3]` $\leftrightarrow$ Bond with site `(z, y+1, x)`
    4.  `bond[z, y, x, 4]` $\leftrightarrow$ Bond with site `(z+1, y, x)`
    5.  `bond[z, y, x, 5]` $\leftrightarrow$ Bond with site `(z, y, x+1)`
    6.  `bond[z, y, x, 6]` $\leftrightarrow$ Bond with site `(z, y-1, x)`

---

### 2\. Simulation Results (`dat`)

This directory stores the raw output data from the simulation, organized by the key physical parameters.

* **Directory Path:**
    ```
    dat/L=<L>/Beta=<Beta>/Zchunk=<Zchunk>/chi=<chi>/seed=<seed>/
    ```
* **Example:**
    ```
    dat/L=4/Beta=1.50000/Zchunk=1/chi=4/seed=1/
    ```
* **Output Files:** Inside the final directory, you will find the following files:

| Filename      | Description                                                                          |
| :------------ | :----------------------------------------------------------------------------------- |
| `3DAcc`       | Stores the acceptance rates for the Monte Carlo updates.                             |
| `3DEne1`      | The energy of the first replica ($E_1$) recorded at each sampling step.                |
| `3DEne2`      | The energy of the second replica ($E_2$) recorded at each sampling step.               |
| `3DMag1`      | The magnetization of the first replica ($M_1$) recorded at each sampling step.         |
| `3DMag2`      | The magnetization of the second replica ($M_2$) recorded at each sampling step.        |
| `3DQvrt`      | The spin overlap order parameter, $q = \frac{1}{N}\sum_i \sigma_i^{(1)}\sigma_i^{(2)}$. |
| `3Dtime`      | The wall-clock time in seconds recorded for each sampling step.                        |
| `logfile.log` | A log file containing real-time progress updates of the simulation (in percent).       |



## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.