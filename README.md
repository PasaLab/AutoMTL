# AutoMTL

## About

This repository is the official implementation of paper *Automatic Multi-Task Learning Framework with Neural Architecture Search in Recommendations*

Multi-task learning (MTL) aims to make full use of knowledge contained in multiple tasks to enhance overall performance and efficiency. 
The main challenge for MTL models is negative transfer. 
Existing MTL models, mainly built on the Mixture-of-Experts (MoE) structure, seek enhancements in performance through feature selection and expert sharing mode design. However, one expert sharing mode may not be universally applicable due to the complex correlations and diverse demands among various tasks. 
Additionally, homogeneous expert architectures in such models further limit their performance.
AutoMTL leverages neural architecture search (NAS) to design optimal expert architectures and sharing modes. 
The Dual-level Expert Sharing mode and Architecture Navigator (DESAN) search space of AutoMTL can not only efficiently explore expert sharing modes and feature selection schemes but also focus on the architectures of expert subnetworks. 
The Progressively Discretizing Differentiable Architecture Search (PD-DARTS) algorithm can efficiently explore the search sapce.

## Requirements

To install requirements:
```bash
pip install -r requirements.txt
```

## Configurations

Download corresponding datasets from

- IJCAI-2015: https://tianchi.aliyun.com/dataset/dataDetail?dataId=472
- UserBehavior-2017: https://tianchi.aliyun.com/dataset/dataDetail?dataId=6493
- KuaiRand-Pure: https://kuairand.com/
- QB-Video: https://static.qblv.qq.com/qblv/h5/algo-frontend/tenrec_dataset.html
- AliCCP: https://tianchi.aliyun.com/dataset/408

Preprocess the datasets by code or jupyter notbooks in `src/datasets/preprocesses/`.

Edit configuration files according to these in `configs/`.

## Architecture Search

Search with AutoMTL: 
```bash
python src/run_nas.py --dataset_name UserBehavior --device_ids=0
```

It will return the training and test performance of searched architecture.

## Valid Searched Architectures

- Set the architecture file path and checkpoint path in `src/test_nas.py`.
- Run script: `src/test_nas.py --dataset_name UserBehavior --device_ids=0`


## Citation

```
@inproceedings{jiang2024automatic,
  title={Automatic Multi-Task Learning Framework with Neural Architecture Search in Recommendations},
  author={Jiang, Shen and Zhu, Guanghui and Wang, Yue and Yuan, Chunfeng and Huang, Yihua},
  booktitle={Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery \& Data Mining},
  year={2024}
}
```

## License

The codes and models in this repo are released under the GNU GPLv3 license.