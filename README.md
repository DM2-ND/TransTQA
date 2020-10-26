## TransTQA data and code

This repository contains the code package for the EMNLP'20 paper:
>[A Technical Question Answering System with Transfer Learning](paper/TransTQA.pdf) <br>

#### Authors: [Wenhao Yu](https://wyu97.github.io/), Lingfei Wu, Yu Deng, Ruchi Mahindru, Qingkai Zeng, Sinem Guven, [Meng Jiang](http://meng-jiang.com/).

Our demonstration website is avaiable at http://159.89.9.22:8080/
We also provide a video for at https://vimeo.com/431118548

## Environment settings
A detailed dependencies list can be found in `requirements.txt` and can be installed by:
```
pip install -r requirements.txt
```

## Run the code
For pretraining the model:
```
./script/run_pretrain.sh
```

For model transfer learning:
```
./script/run_transfer.sh
```

## Cite
If you find this repository useful in your research, please cite our paper:

```
@inproceedings{yu2020technical,
  title={A Technical Question Answering System with Transfer Learning},
  author={Yu, Wenhao and Wu, Lingfei and Deng, Yu and Mahindru, Ruchi and Zeng, Qingkai and Guven, Sinem and Jiang, Meng},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2020}
}
```
