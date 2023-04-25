# IndUDA
IJCAI2023-Independent Feature Decomposition and Instance Alignment for Unsupervised Domain Adaptation

## Data preparation
Please download corresponding dataset([Office-Home](https://www.hemanthdv.org/officeHomeDataset.html), [Office-31]([https://www.hemanthdv.org/officeHomeDataset.html](https://faculty.cc.gatech.edu/~judy/domainadapt/)) ) to \experiments\dataset.

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
