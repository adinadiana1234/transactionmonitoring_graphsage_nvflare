# Automated Federated Learning Setup for GraphSAGE on Elliptic++ Dataset with Terraform and NVFlare

## ORM Stack to deploy a Compute with GPU Shape and create and train a GNN transaction classification model using GraphSAGE and Elliptic++ transaction dataset

The transactional example in NVFlare's Graph Neural Network (GNN) tutorial demonstrates how to apply federated learning to a GNN model, specifically using GraphSAGE for tasks like transactional fraud detection.
https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/gnn

Here’s a high-level breakdown of what happens:

1. ##Model Creation## : The example shows how to set up and initialize a GNN model (GraphSAGE) using PyTorch Geometric. This model learns representations of nodes in a graph, making it well-suited for tasks like classifying transactions based on their connections.

2. Federated Learning Setup: Federated learning is employed here, meaning multiple clients (e.g., banks or organizations) collaborate to train a shared model without sharing their private data. Each client trains the model locally on their data, and only the model updates are shared with a central server. The server aggregates these updates to improve the global model.

3. Commands Involved:

- Model Definition: The GNN model is defined in the configuration files or Python scripts provided in the example. The model architecture (e.g., number of layers, hidden units) is specified in these scripts.
- Training: Commands like nvflare simulator or nvflare job submit initiate training in a federated setting. Each site (client) trains the GNN model locally, and results are aggregated by the central server.

4. Inference Model: After training, the resulting model can be used for inference—making predictions on new, unseen transactions by analyzing their graph-based connections. This step is also included in the workflow to evaluate the model’s effectiveness after training.

In short, this example guides you through setting up a GNN model, applying federated learning to train it across multiple clients, and using the trained model to make predictions about transactions.

## Cloudinit.sh Script Actions

The cloudinit.sh script automates the setup of the environment required to run the GNN transactional classification model using the Elliptic++ dataset. Below is a breakdown of the actions performed by the script:

1. Environment Setup
Adds the required API keys and public SSH keys for authentication.

2. System Update and Package Installation
Installs essential system packages like dnf-utils, zip, unzip, and gcc for compiling and managing software.
Configures Docker repositories and installs Docker to enable containerized environments.
Enables Docker services to start automatically.
Installs the NVIDIA container toolkit to allow for GPU support within Docker containers.

3. Python Environment Setup
Installs python3-pip to manage Python packages and installs/upgrades Python-related tools such as pip, wheel, and oci-cli.
Installs specific Python libraries like langchain and six, which are needed for various components of the project.

4. Storage Configuration
Uses Oracle's oci-growfs tool to automatically grow the filesystem to take advantage of all available space.

5. Transaction Monitoring Setup
Installs Python 3.9, git, and additional tools like wget, unzip, and firewalld (for managing firewall rules).
Sets Python 3.9 as the default Python version.
Installs data science packages such as jupyter, numpy, pandas, matplotlib, and tqdm to support machine learning workflows.

6. NVFlare Setup
Installs NVFlare, NVIDIA's federated learning framework.
Clones the NVFlare GitHub repository and switches to the 2.4 branch to ensure the correct version for the example.
Installs all Python dependencies needed to run the Graph Neural Network (GNN) example using pip and the requirements.txt file from the repository.

7. PyG Package Installation
Installs additional PyTorch Geometric (PyG) packages required to implement the GraphSAGE model, ensuring compatibility with PyTorch 2.4.

8. Dataset Setup
Creates a directory to store the Elliptic++ dataset and downloads it from Oracle's Object Storage.
Unzips the dataset and moves it to the correct directory structure for NVFlare.

9. NVFlare Configuration
Configures NVFlare using predefined job templates and sets up the necessary environment to run federated learning with the GraphSAGE model.

10. Model Training
Runs the GraphSAGE financial classification script locally for each client using the graphsage_finance_local.py script to simulate training on different clients.

11. Create and Run Federated Learning Job
Creates a new NVFlare job that defines the GraphSAGE model training process in a federated setting.
Specifies configurations such as the number of training rounds, the model's architecture, and key metrics like validation_auc.

12. NVFlare Simulator Execution
Runs the NVFlare simulator with two clients to test the federated learning job in a simulated environment.

13. Firewall Configuration for TensorBoard
Configures firewall settings to open port 6006 to allow external access to TensorBoard, which will be used to monitor training progress.

14. TensorBoard Startup
Starts TensorBoard, allowing users to visualize and track the performance of the GNN model during training.

15. Jupyter Notebook Setup
Deploys a Jupyter notebook server inside a Docker container, exposing it on port 8888.
Clones a repository with preconfigured transaction monitoring notebooks to provide a ready-to-use environment for experimentation.

16. Firewall Configuration for Jupyter
Configures the firewall to open port 8888 for external access to the Jupyter notebook interface.


## Next

What ca you do with the model obtained?
- After training your Graph Neural Network (GNN) model (like GraphSAGE) on transactional data, during inference, you would input a new transaction (or a batch of transactions) into the model.
- The model would then analyze the transaction's features (and potentially its connections in the graph, like related transactions or accounts) and output a prediction—for example, a score or label that indicates whether the transaction is fraudulent or not.

## Inference Process for Transaction Fraud Detection:
1. Prepare New Transaction Data: You need to format the new transaction (or a batch of transactions) in the same way you prepared the data for training (e.g., as node features in a graph for GNNs).

2. Load the Model and Input Data: Load the trained model (as we discussed in the previous response) and pass the new transaction data into it.

3. Make a Prediction: The model will output a result based on the input data. For a fraud detection model, this could be:

- A probability or confidence score (e.g., 0.95 means 95% chance the transaction is fraudulent).
- A binary label (e.g., 0 = not fraudulent, 1 = fraudulent).

To test the trained model :

1. Load the Required Libraries
Ensure you have all necessary libraries installed, such as PyTorch and PyTorch Geometric (since you trained a GraphSAGE model).

If you need to install them:

```bash
pip install torch torch-geometric
```

2. Load the Trained Model
In your Python script or interactive environment, load the model file (FL_global_model.pt) that was saved after training.

```python
import torch

# Load the trained model from the file
model_path = 'finance_fl_workspace/simulate_job/app_server/FL_global_model.pt'
model = torch.load(model_path)

# Set the model to evaluation mode (important for inference)
model.eval()
```

3. Prepare New Transaction Data

You need to prepare your new transaction data in the format expected by the model. Since you're using a GNN model like GraphSAGE, the input is typically a graph with nodes representing transactions and edges representing relationships between transactions (e.g., same account, related transactions, etc.).

4. Run inference

Once you have the data in the right format, you can pass it through the model to make predictions.

