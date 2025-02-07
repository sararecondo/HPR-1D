# Guide to Setting Up the Environment and Running the Notebooks

## Prerequisites

1. **Set up WSL with Ubuntu in Visual Studio Code**
2. **Install conda on Ubuntu**

## Create a new conda environment

To create and activate a new conda environment in Ubuntu, follow these steps:

```bash
conda create -n fenicsx -c conda-forge python=3.10 fenics-dolfinx mpich petsc=*=complex* 
conda activate fenicsx
```

## Install project dependencies

Navigate to the project directory and run the following command:

```bash
pip install -r requirements.txt
```

## Install Latex

A requirement for running the notebooks is Latex. To install it, run:

```bash
sudo apt-get update
sudo apt-get install texlive-full
```

## Additional documentation

For more information, refer to the [FEniCS documentation][1].



[1]: https://docs.fenicsproject.org/dolfinx/v0.9.0/python/index.html