Programs for training a CNN model using a UFC outcome dataset

CNN.py - model training using a CPU. Running the training will generate a plot of the resulting weights following training  
CNN_distribute.py - model for distributing training across multiple GPUs if available

Running either python file will train the model and output the accuracy on unseen data (~= 65% accuracy)

Steps

1. To download the training data and create the virtual environment use make install (10 minutes)
```
make install
```
2. Train and return the accuracy of the model on unseen data. Also prints resulting training weights if CNN.py is ran
```
UFC_ENV/bin/python CNN.py
```
3. Deactivate virtual environment and remove anciliary directories
```
make clean
```


