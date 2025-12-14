# Conda environment setup

Create a conda env and activate:
```bash
conda create -n test python=3.12
conda activate test
```

# Installation Environment
```bash
pip install -r requirements.txt
```
You may need to install MOSEK according to **[MOSEK installation guide](https://docs.mosek.com/10.2/install/index.html)**.

# Run the code
For training process, you can run
```bash
python train.py {method} {environment}
```
to train the model.

Method can be **DKO** **mamba** **MLP**;
Environment can be **cartpole** and **cartpole_V**.

For the control process, you can run
```bash
python control.py {method} {environment}
```
to evaluate the control performance of the given model.

Method can be **DKO** **mamba** **MLP**;
Environment can be **cartpole** and **cartpole_V**.

# MamKO wastewater training shortcuts

To launch the MamKO wastewater runs with the recommended settings, use the helper script:

```bash
# Standard (simulated wastewater) training
scripts/train_mamko.sh

# Hybrid training (real + simulated influent data)
scripts/train_mamko.sh hybrid

# Override defaults
EPOCHS=500 BATCH_SIZE=128 scripts/train_mamko.sh
```

# Wastewater workflow

For details on preparing the wastewater influent datasets, choosing configuration flags (including hybrid mode), and running training or resilience evaluation against the wastewater environment, see [`docs/wastewater.md`](docs/wastewater.md).


