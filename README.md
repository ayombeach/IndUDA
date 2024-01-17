# IndUDA
IJCAI2023-Independent Feature Decomposition and Instance Alignment for Unsupervised Domain Adaptation

## Abstract
Existing Unsupervised Domain Adaptation (UDA) methods typically attempt to perform knowledge transfer in a domain-invariant space explicitly or implicitly. In practice, however, the obtained features are often mixed with domain-specific information which causes performance degradation. To overcome this fundamental limitation, this article presents a novel independent feature decomposition and instance alignment method (IndUDA in short). Specifically, based on an invertible flow, we project the base feature into a decomposed latent space with domain-invariant and domain specific dimensions. To drive semantic decomposition independently, we then swap the domain invariant part across source and target domain samples with the same category and require their inverted features are consistent in class-level with the original features. By treating domain-specific information as noise, we replace it by Gaussian noise and further regularize source model training by instance alignment, i.e., requiring the base features close to the corresponding reconstructed features, respectively. Extensive experiment results demonstrate that our method achieves state-of-the-art performance on popular UDA benchmarks.

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

## Citation
If you use this toolbox or benchmark in your research, please cite this paper.
```
@inproceedings{ijcai2023p91,
  title     = {Independent Feature Decomposition and Instance Alignment for Unsupervised Domain Adaptation},
  author    = {He, Qichen and Xiao, Siying and Ye, Mao and Zhu, Xiatian and Neri, Ferrante and Hou, Dongde},
  booktitle = {Proceedings of the Thirty-Second International Joint Conference on
               Artificial Intelligence, {IJCAI-23}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  pages     = {819--827},
  year      = {2023}
}

```

## Acknowledgment
We would like to thank School of CSE, University of Electronic Science and Technology of China, Surrey Institute for People-Centred Artifcial Intelligence and NICE Research Group, University of Surrey, and Advanced Research Institute, Southwest University of Political Science&Law for providing such an excellent ML research platform.
