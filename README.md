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


