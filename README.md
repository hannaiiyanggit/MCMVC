# View Labels Are Important: A Multifacet Complementarity Study of Deep Multi-view Clustering
Here we provide the code and data of this paper.
## Requirements
pytorch >= 1.2.0
numpy >= 1.19.1
scikit-learn >= 0.23.2
munkres >= 1.1.4
## Configuration
The hyper-parameters are defined in configure.py.
## Datasets
Scene-15 and LandUse21 can be found in MCMVC/data/, while NoisyMNIST can be downloaded at [Google Drive](https://drive.google.com/file/d/1b__tkQMHRrYtcCNi_LxnVVTwB-TWdj93/view).
## Usage
The code supports:
* an example implementation of the model
* an example implementation with different missing rates
* an example implementation with two different functions for instance-level contrastive learning as in the paper
> python main.py --dataset 1 --devices 0 --print_num 50 --test_time 5 --instance_loss 0 --missing_rate 0.0

We use 0,1,2,3 to represent different datasets (Caltech101-20, Scene-15, LandUse21, and NoisyMNIST), and we use 0,1 to use different loss functions (mse loss and infoNCE loss).

