##  ACAS Xu (DNN)

----

#### Setting up environment:

Run the following:
```bash
conda create -n acas python=3.7.9
conda env update --name acas --file environment_ACAS.yml
conda activate acas
```

----

#### Notes:
The core code of *CureFuzz* is in `./fuzz/cure_fuzz.py`. 

`./simulate.py` is the main simulation code for ACAS Xu.

All models including ACAS Xu models and models after repair are inside `./models`.

----

#### Fuzz testing:
Run `python simulate.py` to start fuzz testing.

We set the default fuzzing time to 12 hours with 10000 initial seeds by `python simulate.py --terminate 12 --seed_size 10000`.


