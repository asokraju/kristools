# kristools
Tests running DDPG on a simple buck converter environment.

Use `config.yml` to set the parameters of the desired experiment. Running `test.py` as a script will then output the results to a named and dated directory in the results folder.

## Using Singularity
The Singularity definition file `dcbf.def` can be used to create a Singularity container for running tests. To use it, first install `singularity` then do
```bash
# singularity build dcbf.sif dcbf.def
```
To run the container with `tensorflow` access to NVIDIA GPUs, do
```bash
$ singularity run --nv dcbf.def
```
