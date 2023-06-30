# GBV-GSL
This is the code of paper: Gated Bi-View Graph Structure Learning.

# Environment Settings
```
python==3.7.15
numpy==1.21.5
scikit_learn==1.0.2
scipy==1.7.3
torch==1.21.1
torch_geometric==2.2.0
torch_sparse==0.6.16
```
GPU: GeForce RTX A100 \
CPU: Intel(R) Xeon(R) Platinum 8268 CPU @ 2.90GHz

# Usage
Go into ./code/, and then run the following command:
```
python main.py wine --gpu=0
```
where "wine" can be replaced by "breast_cancer", "digits", "polblogs", "texas", "cornell" or "wisconsin". \
If you futher want to reproduce the results under attacks, please use the following command:
```
python main.py breast_cancer --gpu=0 --dele --flag=1 --ratio=0.05
```
where datasets∈["breast_cancer", "polblogs", "texas"], --dele can be replaced by --add, --flag∈[1, 2, 3], 1 or 2 means the 1st or 2nd view is attacked, and 3 means both of them are attacked. --ratio is in [0.05, 0.1, 0.15] under --dele, while in [0.25, 0.5, 0.75] under --add

# Contact
If you have any questions, please feel free to contact me with {wangxinyi@njust.edu.cn}
