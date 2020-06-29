

# CompReSS: Compressing Representations for Self-Supervised Learning


<p align="center">
  <img src="https://user-images.githubusercontent.com/62820830/85651404-e48b8880-b675-11ea-9432-754546efad1a.png" width="85%">
</p>

This repository is the official implementation of [https://arxiv.org/
](https://arxiv.org/). 

Project webpage. [https://umbcvision.github.io/CompReSS/
](https://umbcvision.github.io/CompReSS/). 

```
@Article{abbasi2020compress,
  author  = {Soroush Abbasi Koohpayegani and Ajinkya Tejankar and Vipin Pillai and Hamed Pirsiavash},
  title   = {CompReSS: Compressing Representations for Self-Supervised Learning},
  journal = {arXiv preprint arXiv:},
  year    = {2020},
}
```

[comment]: <> (📋Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials)

## Requirements

Install PyTorch and ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet). We used Python 3.7.4 for our experiments.

To install requirements:

PyTorch:
```setup
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

To run NN and Cluster Alignment, you require to install FAISS. 

FAISS: 
```setup
# CPU version only
conda install faiss-cpu -c pytorch

# GPU version
conda install faiss-gpu cudatoolkit=10.0 -c pytorch # For CUDA10
```




[comment]: <>  (📋Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...)

## Training

Our code is based on unofficial implementation of Moco from [here](https://github.com/HobbitLong/CMC). 


To train the student(s) using cached teachers in the paper :


We converted TensorFlow simCLRv1 ResNet50x4(download it [here](https://github.com/google-research/simclr)) to PyTorch. You can download pretrained simCLR ResNet50x4 PyTorch model from [here](https://drive.google.com/file/d/1fZ2gfHRjVSFz9Hf2PHsPUao9ZKmUXg4z/view?usp=sharing).

First, run this command to calculate and store cache features.  


```train
python train_student.py \
    --cache_teacher \ 
    --teacher <path_to_pretrained_model or cached_features> \
    --student_arch mobilenet \
    --checkpoint_path <path_to_checkpoint_folder> \
    --data <path_to_imagenet_data>
```



To train the student(s) using pretrained teachers in the paper :


Download pretrained Moco ResNet50 model from [here](https://github.com/facebookresearch/moco).

```train
python train_student.py \
    --teacher_arch resnet50 \ 
    --teacher <path_to_pretrained_model or cached_features> \
    --student_arch mobilenet \
    --checkpoint_path <path_to_checkpoint_folder> \
    --data <path_to_imagenet_data>
```


To train the student(s) without Momentum framework execute train_student_without_momentum.py instead of train_student.py

[comment]: <> (📋Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.)
## Evaluation

To evaluate NN models on ImageNet, run:

```eval
python eval_knn.py \
    -a alexnet \
    --weights <path_to_pretrained_model> \
    --save <path_to_save_folder> \
    <path_to_imagenet_data>
```


To evaluate Linear Classifier models on ImageNet, run:

```eval
python save_var_mean.py --x_root <NN_evaluation_save_folder>

python eval_multiple_linear.py \
    --arch alexnet \
    --weights <path_to_pretrained_model> \
    --mean_paths <NN_evaluation_save_folder>/var_mean.pth.tar \
    --save <path_to_save_folder> \
    <path_to_imagenet_data>
```

To evaluate cluster alignment models on ImageNet, run:

```eval
python eval_cluster_alignment.py  \
    --batch-size 256 \ 
    --weights <path_to_pretrained_model> \
    --model resnet18  \
    --save <path_to_save_folder> \
    <path_to_imagenet_data> \ 
    --visualization \ 
    --confusion_matrix
```




## Results

<p align="center">
  <img src="https://user-images.githubusercontent.com/62820830/85651425-f2d9a480-b675-11ea-99e8-8fcf32d0a266.png" width="100%">
</p>



Our model achieves the following performance :


| Model name         | Teacher | Top 1 Linear Classifier Accuracy | Top 1 NN Accuracy | Pre-trained |
| ------------------ | --------- |----------------------------------| ----------------- | ----------------- |
| CompReSS(Resnet50) | SimCLR ResNet50x4 |               71.6%              |        63.4%        | [Pre-trained Resnet50](https://drive.google.com/file/d/15rzzSkcedEuCE7Cm8yLXopA5PqHUQscb/view?usp=sharing) |
| CompReSS(Mobilenet)| MocoV2 ResNet50 |               63.0%              |        54.4%        | [Pre-trained Mobilenet](https://drive.google.com/file/d/1gNkO48iREh6M6uuLd8TGqaOm3ChWiAdc/view?usp=sharing) |
| CompReSS(Resnet18) | MocoV2 ResNet50 |               61.7%              |        53.4%        | [Pre-trained Resnet18](https://drive.google.com/file/d/1L-RCmD4gMeicxJhIeqNKU09_sH8R3bwS/view?usp=sharing) | 
| CompReSS(Alexnet)  | SimCLR ResNet50x4 |               57.6%              |        52.3%        | [Pre-trained Alexnet](https://drive.google.com/file/d/1wiEdfk6unXKtYRL1faIMoZMXnShaxBMU/view?usp=sharing) |





## License

This project is under the MIT license.


