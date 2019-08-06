# Siamese Neural Network Architecture for EEG Classification

### Citation
The paper can be found [here].

Please cite this work as follows:

```
@article{calhas2019use,
  title={On the use of Pairwise Distance Learning for Brain Signal Classification with Limited Observations},
  author={Calhas, David and Romero, Enrique and Henriques, Rui},
  journal={arXiv preprint arXiv:1906.02076},
  year={2019}
}


```

### Requirements:
* [Python - 3.7]
* [tensorflow-gpu]
* [keras]
* [GPyOpt]
* [sklearn]
* [numpy]
* [scipy]
* [random]
* [xgboost]

### Instructions to run:
Before running please make sure you have all of the requirements  installed and a Nvidia GPU.
The following scripts should be ran first: 
* **snn_hyperparameter_optimization.py**
* **cnnclf_hyperparameter_optimization.py**

These scripts compute the suboptimal hyperparameters obtained from a Bayesian Optimization Search.

Following, to obtain the results presented in the paper, please run the following scripts: 
* **fft_loo_validation.py**
* **cnnclf_loo_validation.py**
* **snn_loo_validation.py**

Please note that the **snn_hyperparameter_optimization.py** and **cnnclf_hyperparameter_optimization.py** scripts take quite a while to run, so we left the optimized hyperparameters already computed. If you want to run them again, please insert the optimized values obtained in the **cnnclf_loo_validation.py** and **snn_loo_validation.py** scripts in the lines below a comment: 

```
#change these values according to the ones obtained ...
```

### Reproducibility/Comparison Baseline
If you want to compare this model with your own in a different dataset, please modify **eeg.py** file to what best suits you. The authors thank you for your interest in our work. Please cite our work when using it :)

[Python - 3.7]: https://www.python.org/downloads/release/python-370/
[tensorflow-gpu]: https://www.tensorflow.org/install/gpu
[keras]: https://keras.io/
[GPyOpt]: https://sheffieldml.github.io/GPyOpt/
[sklearn]: https://pypi.org/project/scikit-learn/
[numpy]: https://pypi.org/project/numpy/
[scipy]: https://pypi.org/project/scipy/
[random]: https://pypi.org/project/random2/
[xgboost]: https://pypi.org/project/xgboost/
[here]: https://arxiv.org/pdf/1906.02076.pdf
