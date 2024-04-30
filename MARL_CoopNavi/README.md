##  MARL Coop Navi

#### Instructions on launching *CureFuzz* for Multi-Agent Reinforcement Learning models for Coop-Navi

#### Setting up environment:

```bash
conda create -n MARL python=3.5.4
conda env update --name MARL --file environment_MARL.yml
conda activate MARL

cd ./maddpg
pip install -e .
cd ../multiagent-particle-envs
pip install -e .
cd ../maddpg/experiments/
```

----

#### Fuzz testing:

Check the default path of the model is correct in `./maddpg/experiments/test_cure.py`. 

Run `python testing.py` to start fuzz testing.

