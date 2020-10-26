## TransTQA data and code

This repository contains the code package for the EMNLP'20 paper:

[A Technical Question Answering System with Transfer Learning](paper/TransTQA.pdf), [Wenhao Yu](https://wyu97.github.io/) (ND), [Lingfei Wu](https://sites.google.com/a/email.wm.edu/teddy-lfwu/) (IBM), Yu Deng (IBM), Ruchi Mahindru (IBM), Qingkai Zeng (ND), Sinem Guven (IBM), [Meng Jiang](http://meng-jiang.com/) (ND).

- Our demonstration website is avaiable at http://159.89.9.22:8080/

- We also provide a video for the paper at https://vimeo.com/431118548

## Environment settings
A detailed dependencies list can be found in `requirements.txt` and can be installed by:
```
pip install -r requirements.txt
```

## Run the code
For pre-training the model (we use [askubuntu](https://askubuntu.com/) for technical domain QA pre-training as default):
```
./script/run_pretrain.sh
```

For model transfer learning (we have two target datasets: [stackunix](https://unix.stackexchange.com/) and [techqa](https://leaderboard.techqa.us-east.containers.appdomain.cloud/) (ACL 2020)):
```
./script/run_transfer.sh
```
***Note*** that you should specify the path of pre-trained model and dataset.

## Citation
If you find this repository useful in your research, please cite our paper:

```
@inproceedings{yu2020technical,
  title={A Technical Question Answering System with Transfer Learning},
  author={Yu, Wenhao and Wu, Lingfei and Deng, Yu and Mahindru, Ruchi and Zeng, Qingkai and Guven, Sinem and Jiang, Meng},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2020}
}
```
