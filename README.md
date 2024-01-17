# IndUDA
IJCAI2023-Independent Feature Decomposition and Instance Alignment for Unsupervised Domain Adaptation

## Data preparation
Please download corresponding dataset([Office-Home](https://www.hemanthdv.org/officeHomeDataset.html), [Office-31](https://faculty.cc.gatech.edu/~judy/domainadapt/)) to \experiments\dataset.

## Training
Example on Office-Home:
```
python tools/train.py --cfg ./experiments/config/Office-home/cp.yaml --method INN --exp_name home_c2p
```

## Testing
Example on Office-Home:
```
python tools/test.py --cfg config_filg --weights weights_file --exp_name home_c2p
```

## Citing
```
@inproceedings{ijcai2023p91,
  title     = {Independent Feature Decomposition and Instance Alignment for Unsupervised Domain Adaptation},
  author    = {He, Qichen and Xiao, Siying and Ye, Mao and Zhu, Xiatian and Neri, Ferrante and Hou, Dongde},
  booktitle = {Proceedings of the Thirty-Second International Joint Conference on
               Artificial Intelligence, {IJCAI-23}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Edith Elkind},
  pages     = {819--827},
  year      = {2023},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2023/91},
  url       = {https://doi.org/10.24963/ijcai.2023/91},
}

```
