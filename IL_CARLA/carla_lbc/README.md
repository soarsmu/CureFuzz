##  IL CARLA

#### Instructions on launching *CureFuzz* for  Imitation Learning models for CARLA

----

#### Notes:
The core code of *CureFuzz* is in `./fuzz/fuzz.py`. 

The RL model we evaluate is borrowed from this awesome repository: https://github.com/dotchen/LearningByCheating, which is under MIT license.

Part of the `PythonAPI` and the map rendering code is borrowed from the official [CARLA](https://github.com/carla-simulator/carla) repo, which is under MIT license.

----

#### Setting up environment:

We recommend to setup environment for *RL_CARLA* first, as the *CARLA* can be reused.

Run the following:
```bash
# Setup CARLA
cd PythonAPI/carla/dist
wget http://www.cs.utexas.edu/~dchen/lbc_release/egg/carla-0.9.6-py3.5-linux-x86_64.egg

# Setup environment
conda create -n carlaIL python=3.5.6

easy_install carla-0.9.6-py3.5-linux-x86_64.egg

cd ../../..

conda env update --name carlaIL --file environment_carlail.yml
conda activate carlaIL

# Download models
mkdir -p ckpts/image
cd ckpts/image
wget http://www.cs.utexas.edu/~dchen/lbc_release/ckpts/image/model-10.th
wget http://www.cs.utexas.edu/~dchen/lbc_release/ckpts/image/config.json
cd ../..
mkdir -p ckpts/priveleged
cd ckpts/priveleged
wget http://www.cs.utexas.edu/~dchen/lbc_release/ckpts/privileged/model-128.th
wget http://www.cs.utexas.edu/~dchen/lbc_release/ckpts/privileged/config.json
cd ../..
```

----

#### Fuzz testing:
First run `../../RL_CARLA/carla_RL_IAs/CarlaUE4.sh -fps=10 -benchmark -carla-port=3000` to start the CARLA environment.

Run `python ./benchmark_agent.py --suite=town2 --model-path=ckpts/image/model-10.th` to start fuzzing.