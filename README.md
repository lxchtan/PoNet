## Introduction

This is a code repository for our paper ***[PoNet: Pooling Network for Efficient Token Mixing in Long Sequences](https://arxiv.org/abs/2110.02442)***.
The full source code will be released soon.

## Requirements

If you want to clone with the checkpoint, `git-lfs` is needed. And then use the code

```shell
git lfs install
git clone https://github.com/lxchtan/PoNet
```

If you want to clone without large files,

```shell
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/lxchtan/PoNet
```

##### Python environment

The requirements package is in `requirements.txt`.

`Torch-scatter` can be gotten from the link `https://github.com/rusty1s/pytorch_scatter`.

If you are using nvidia's GPU and CUDA version supports 10.2, you can use the following code to create the desired virtual python environment:

```shell
conda create -n ponet python=3.8
conda activate ponet
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts
pip install -r requirements.txt
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.8.2+cu102.html
```

## Instruction

Please refer to the shell file under the `run_shell` folder.

## TODO

- [ ] release the source code
  - [x] [2021.10.19] Pre-train Tasks 
  - [x] [2021.10.19] GLUE Tasks
  - [x] [2022.03.15] LRA Tasks
  - [ ] Long-Text Tasks
- [x] [2021.10.19] release the pretrained checkpoints

## Cite

```bibtex
@inproceedings{DBLP:journals/corr/abs-2110-02442,
  author    = {Chao{-}Hong Tan and
               Qian Chen and
               Wen Wang and
               Qinglin Zhang and
               Siqi Zheng and
               Zhen{-}Hua Ling},
  title     = {{PoNet}: Pooling Network for Efficient Token Mixing in Long Sequences},
  booktitle = {10th International Conference on Learning Representations, {ICLR} 2022,
               Virtual Event, April 25-29, 2022},
  publisher = {OpenReview.net},
  year      = {2022},
  url       = {https://openreview.net/forum?id=9jInD9JjicF},
}
```
