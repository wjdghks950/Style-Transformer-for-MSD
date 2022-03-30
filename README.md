# Style Transformer for Expert-Laymen Style Transfer (on MSD Dataset)

This repository contains the codes for the paper [Style Transformer: Unpaired Text Style Transfer without Disentangled Latent Representation](https://arxiv.org/abs/1905.05621).

The code has been revised to incorporate and train on the expert-laymen style transfer task dataset ([MSD dataset](https://aclanthology.org/2020.acl-main.100.pdf)).

The following are the revisions made to the existing code base:
1. Code refactoring to deal with each data instance and feature on a class-level (e.g., `MSDFeature`, `MSDExample`).
2. Fine-tuned BERT-base model (`bert-base-uncased`) on the MSD training dataset to measure the **Style Accuracy** of the style-transferred sentence. This replaces the `fasttext` classifier originally used in the baseline code.
3. Replaced `kenlm` with the `torch.exp(loss)` to calculate the perplexity of the generated sentence.

## Requirements

pytorch >= 0.4.0
torchtext >= 0.4.0
nltk

## Usage

The hyperparameters for the Style Transformer can be found in ''main.py''.

The most of them are listed below:

```
    data_path : the path of the datasets
    log_dir : where to save the logging info
    save_path = where to save the checkpoing
    
    discriminator_method : the type of discriminator ('Multi' or 'Cond')
    min_freq : the minimun frequency for building vocabulary
    max_length : the maximun sentence length 
    embed_size : the dimention of the token embedding
    d_model : the dimention of Transformer d_model parameter
    h : the number of Transformer attention head
    num_layers : the number of Transformer layer
    batch_size : the training batch size
    lr_F : the learning rate for the Style Transformer
    lr_D : the learning rate for the discriminator
    L2 : the L2 norm regularization factor
    iter_D : the number of the discriminator update step pre training interation
    iter_F : the number of the Style Transformer update step pre training interation
    dropout : the dropout factor for the whole model

    log_steps : the number of steps to log model info
    eval_steps : the number of steps to evaluate model info

    slf_factor : the weight factor for the self reconstruction loss
    cyc_factor : the weight factor for the cycle reconstruction loss
    adv_factor : the weight factor for the style controlling loss
```

You can adjust them in the Config class from the ''main.py''.

If you want to run the original Style Transformer model on Yelp and IMDB, use the following command:

```shell
$ ./train.sh 0
```
(0 simply stands for the CUDA gpu id (if there is one in your system))

If you want to run the revised, MSD-version of Style Transformer model on MSD dataset, use the following command:

```shell
$ ./msd_train.sh 0
```


## Outputs

Update: You can find the outputs of our model in the "outputs" folder.