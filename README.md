# Flower Species Image Classifier

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

## Project Structure

The repository contains the following files:

- `Image Classifier Project.ipynb`: A Jupyter notebook where the image classifier is built using transfer learning with ResNet-18. This notebook includes visualizations of the training process, such as loss and accuracy metrics over time, to provide insights into the model's performance.

- `train.py`: A command-line script to train a new network on a dataset and save the model as a checkpoint.

- `predict.py`: A command-line script to use a trained network for predicting the class of an input image.

## Usage

### Train

To train a new network on a dataset, use the `train.py` command-line script. This script will print out the training loss, validation loss, and validation accuracy as the network trains.

#### Basic Usage:

```bash
python train.py data_dir --save_dir save_directory
```

#### Options:

- Set directory to save checkpoints:

```bash
python train.py data_dir --save_dir save_directory
```

- Choose architecture:

```bash
python train.py data_dir --arch "resnet18"
```

- Set hyperparameters:

```bash
python train.py data_dir --learning_rate 0.01 --hidden_units 250 --epochs 3
```

- Use GPU for training:

```bash
python train.py data_dir --gpu
```
