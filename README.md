# TensorflowGan

Conditional Generative Adverserial Network trained to approximate the distribution of MNIST Dataset. Also used some improvements from "Improved Techniques for Training GANs" (Salimans et al., 2016).

# Get started

First clone this repository in a folder of your choice and switch to the created folder

    git clone https://github.com/matthias-bitzer/TensorflowGan.git
    cd TensorflowGan
    
The dataset is directly loaded from the tensorflow API. Make sure you have tensorflow, numpy, matplotlib and open-cv installed and perform

    python gan_model.py
    
to start the training.
