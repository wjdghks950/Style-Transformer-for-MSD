# Style Transformer for Expert-Laymen Style Transfer (MSD Dataset)

This repository contains the codes for **Style Transformer for Expert-Laymen Style Transfer**. The Style Transfer task requires mapping a source language (e.g., expert) to a target language (e.g., laymen).

An illustration is given in the ([MSD dataset](https://aclanthology.org/2020.acl-main.100.pdf)), where a sentence composed of expert-level terminologies is converted to an easily understandable, "laymen" sentence.

This repository builds upon [Style Transformer: Unpaired Text Style Transfer without Disentangled Latent Representation](https://arxiv.org/abs/1905.05621), where the Transformer's self-attention-powered style transfer model is proposed.

Our work processes the expert style transfer dataset (i.e., MSD dataset) to convert expert language into the corresponding laymen language using the Style Transformer model.

This repository is composed of the following:
1. Implementation of `MSDFeatures` and `MSDExample` classes to integrate MSD dataset into the codebase.
2. A separately fine-tuned BERT-base model (`bert-base-uncased`) on the MSD training dataset to measure the **Style Accuracy** of the style-transferred sentence. This replaces the `fasttext` classifier originally used in the baseline code.
3. Replaced `kenlm` with the `torch.exp(loss)` to calculate the perplexity of the generated sentence.

## Requirements
```
	pytorch >= 0.4.0
	torchtext >= 0.4.0
	nltk
```

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
	$ ./train.sh [gpu_id]
```

If you want to run the MSD-version of Style Transformer model on MSD dataset, use the following command:

```
	$ ./msd_train.sh 0
```


## Outputs

Update: You can find the outputs of our model in the "outputs" folder.

