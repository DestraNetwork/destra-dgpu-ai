# Destra-DGPU-AI - Alpha 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


Destra-DGPU-AI is a Python package for connecting to the Destra GPU Network to train AI models. This guide provides instructions for installing the package, setting up the environment, and integrating the library into your training scripts.


## Prerequisites

- Ubuntu system
- Stable internet connection
- Sepolia RPC Endpoint
- Destra GPU Registry - to connect to head node ([deployed on testnet](https://sepolia.etherscan.io/address/0x9B1B198C5C671F8B5a67721cC4Fff5E9F020D505))


## Setup Instructions

### 1. Clone the GitHub Repository

First, clone the GitHub repository to your local machine.

```sh
git clone https://github.com/DestraNetwork/destra-dgpu-ai.git
cd destra-dgpu-ai
```



### 2. Install Python 3.9.6

Ensure you have Python 3.9.6 installed. You can either install from source or use `pyenv` to manage your Python versions.

#### Install from Source (Recommended)

1. **Install Required Build Tools**:

    ```sh
    sudo apt-get update
    sudo apt-get install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev curl libbz2-dev
    ```

2. **Download and Extract Python Source Code**:

    ```sh
    cd /usr/src
    sudo wget https://www.python.org/ftp/python/3.9.6/Python-3.9.6.tgz
    sudo tar xzf Python-3.9.6.tgz
    ```

3. **Build and Install Python**:

    ```sh
    cd Python-3.9.6
    sudo ./configure --enable-optimizations
    sudo make altinstall
    ```

#### Or Using `pyenv`

1. **Install `pyenv`**:

    ```sh
    curl https://pyenv.run | bash
    ```

    Add the following lines to your shell configuration file (e.g., `~/.bashrc`, `~/.zshrc`):

    ```sh
    export PATH="$HOME/.pyenv/bin:$PATH"
    eval "$(pyenv init --path)"
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"
    ```

    Restart your shell or source the configuration file:

    ```sh
    source ~/.bashrc
    source ~/.zshrc
    ```

2. **Install Python 3.9.6**:

    ```sh
    pyenv install 3.9.6
    pyenv global 3.9.6
    ```

### 3. Set Up a Virtual Environment

Create and activate a virtual environment for your project.

```sh
python3.9 -m venv dgpu-env
source dgpu-env/bin/activate
```

### 4. Install the Destra-DGPU Package

Install the `destra-dgpu-ai` package from the provided wheel file.

```sh
cd ~/destra-dgpu-ai
pip install destra-dgpu-ai/destra_dgpu_ai-0.1.0-cp39-cp39-linux_x86_644.whl
```

### 5.  Integrating Destra-DGPU-AI in Your Training Script

#### Example Training Script

Below is a pseudocode example that illustrates how to use the Destra-DGPU-AI library to connect to the Destra GPU network and train an AI model.

```python
# Import libraries
from destra_dgpu_ai import destra_ray, destra_ray_init
import torch
from some_model_library import Model, Dataset

# Initialize Destra GPU network
destra_ray_init("0x9B1B198C5C671F8B5a67721cC4Fff5E9F020D505", "https://sepolia.infura.io/v3/<YOUR_INFURA_KEY>")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset and model
dataset = Dataset.load("your_dataset")
model = Model.load("your_model").to(device)

# Tokenize and prepare dataset
def preprocess(data):
    # Tokenize data
    return tokenized_data

tokenized_dataset = preprocess(dataset)

# Put large objects in Ray object store (Optional)
model_ref = destra_ray.put(model)
dataset_ref = destra_ray.put(tokenized_dataset)

# Define training function
@destra_ray.remote
def train_model(model_ref, dataset_ref, hyperparameters):
    model = destra_ray.get(model_ref)
    dataset = destra_ray.get(dataset_ref)
    
    # Training logic here
    for epoch in range(hyperparameters['epochs']):
        for batch in dataset:
            # Train model
            pass
    
    return model

# Define hyperparameters
hyperparameters = {
    'epochs': 10,
    'learning_rate': 0.001,
    'batch_size': 32
}

# Start training
trained_model_ref = train_model.remote(model_ref, dataset_ref, hyperparameters)
trained_model = destra_ray.get(trained_model_ref)

# Save trained model
trained_model.save("path_to_save_trained_model")
```

### 5. Run Your Script

Run your script to start training your AI model using the Destra GPU network.

```sh
python your_training_script.py
```

## Troubleshooting

If you encounter any issues, ensure that:

- The necessary dependencies are installed.
- The correct Python version is being used.
- Your firewall rules are allowing your script to connect to the Destra GPU Network.

If you still face any issues, contact us on our telegram channel.