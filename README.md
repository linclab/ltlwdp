Code accompanying "Learning to live with Dale's principle: ANNs with separate excitatory and inhibitory units" by Cornford, Kalajdzievski, Leite, Lamarquette, Kullmann and Richards. 

The "exp_configs" directory contains config files for different experiments and models, "ariamis" contains code. Run with:

```
python run_exp.py --config exp_configs/test_mnist_mlp.yaml
python run_exp.py --config exp_configs/test_mnist_DANN.yaml
python run_exp.py --config exp_configs/test_mnist_DANN_no_correction.yaml
python run_exp.py --config exp_configs/test_mnist_columnEI.yaml
python run_exp.py --config exp_configs/test_mnist_dngd_config.yaml
```

Requirements:
```
orion==0.1.8
torch==1.5.0
torchvision==0.6.0
```

Code for convolutional models in Appendix B is in the "conv" directory. 
