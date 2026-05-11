# SEMamba Apple Silicon

This repository provides an Apple Silicon-compatible version of SEMamba for Mac computers with Apple M-series chips.

The original SEMamba repository is available here:

https://github.com/RoyChao19477/SEMamba.git

## Installation

```python3
conda env create -f environment.yml
conda activate semamba_mac

python -m pip install -r requirements.txt
python -m pip install -e ./mamba_install_mps --no-deps
python -m pip install transformers==4.49.0
```

## Notes

This version is intended for running SEMamba on Apple Silicon Macs using PyTorch MPS.

For the original implementation, training code, and full project details, please refer to the original repository above.


## Citation:
If you find the paper useful in your research, please cite:  
```
@article{chao2024investigation,
  title={An Investigation of Incorporating Mamba for Speech Enhancement},
  author={Chao, Rong and Cheng, Wen-Huang and La Quatra, Moreno and Siniscalchi, Sabato Marco and Yang, Chao-Han Huck and Fu, Szu-Wei and Tsao, Yu},
  journal={arXiv preprint arXiv:2405.06573},
  year={2024}
}
```
