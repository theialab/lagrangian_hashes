# Official code for Lagrangian Hashes

[[Project Page]](https://theialab.github.io/laghashes/)
[[Arxiv]](https://arxiv.org/abs/2409.05334)

## Cloning the Repository
The repository contains submodules, thus please check it out with
```bash
# clone the repo with submodules.
git clone --recursive https://github.com/theialab/lagrangian_hashes.git
```

## Installation

**Dependence**: Please install [Pytorch](https://pytorch.org/get-started/previous-versions/) first. This code is tested with Pytorch-2.1.0 with CUDA 12.1

Then please install nerfacc and lagrangian hashes code by running the setup.py file
```
python -m pip install .
cd laghash
python -m pip install -e .
cd ..
```


Then to install other libraries(including tcnn), use the requirements.txt
```
pip install -r requirements.txt
```

## Experiments 

Before running the example scripts, please check which dataset is needed, and download the dataset first. You could use `dataset.data_root` to specify the path or modify the config yaml.

### NeRF-Synthetic (blender) dataset

``` bash
python examples/train_laghash_nerf_occ.py --config-name "synthetic_occ.yaml" dataset.scene="chair"
```

### Tanks and Temples (masked) dataset

``` bash
python examples/train_laghash_nerf_occ.py --config-name "tanks_temples_occ.yaml" dataset.scene="Family"
```




## Citation

```bibtex
@inproceedings{govindarajan2024laghashes,
  title     = {Lagrangian Hashing for Compressed Neural Field Representations},
  author    = {Shrisudhan Govindarajan, Zeno Sambugaro, Ahan Shabhanov, Towaki Takikawa, Weiwei Sun, Daniel Rebain, Nicola Conci, Kwang Moo  Yi, Andrea Tagliasacchi},
  booktitle = {ECCV},
  year      = {2024},
}
```
