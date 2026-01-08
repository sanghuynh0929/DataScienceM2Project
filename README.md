# DataScienceM2Project

### 1. Prerequisites and Installation

The notebook relies on several standard data science and machine learning libraries. Ensure you have them installed in your environment. You can install the necessary packages using `pip`:

```bash
pip install numpy pandas matplotlib seaborn torch scikit-learn statsmodels scipy

```

After installation, the entire notebook can be run. 

### 2. Dataset

The data is **automatically downloaded** from this GitHub repository within the notebook. It uses the `freMTPL2freq` dataset (a French Motor Third-Party Liability insurance frequency dataset).
The code also includes an automated pipeline to handle variable preprocessing, which differs slightly when dealing with different models (i.e. model with only numerical features/model with entity embedding/model with one-hot embedding)

### 3. Model Architecture: `LocalGLMNet`

The `LocalGLMNet` class defines a neural network architecture designed to mimic and enhance a standard Generalized Linear Model (GLM).

* **Hidden Layers:** It contains a series of fully connected layers with `tanh` activation that extract non-linear features from the input.
* **Local GLM Layer:** The final output of the network creates "local" coefficients for each input feature, then a skip connection is added.
* **Poisson Regression:** It is specifically designed for count data by applying an exponential activation to the output and multiplying it by the `exposure`.

### 4. Training Function: `train_model`

The `train_model` function is a general-purpose utility to handle the optimization process for all PyTorch models in this notebook.

* **Loss Function:** It uses `PoissonNLLLoss` (Poisson Negative Log-Likelihood Loss), which is standard for frequency modeling.
* **Training Loop:** It iterates through the specified number of epochs, performing backpropagation using the provided optimizer (here we use `NAdam`).
* **Validation & Checkpointing:** After each epoch, it calculates the loss on the validation set. If the validation loss improves, it saves the model's state to a `checkpoint_path` to ensure the best version is preserved.
* **Performance Tracking:** It returns a history of training and validation losses, which are visualized in the notebook to monitor convergence. The model with the best validation loss will be reloaded to be evaluated in the test set.
