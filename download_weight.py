from huggingface_hub import snapshot_download

# Download entire model
snapshot_download(
    repo_id="nvidia/Nemotron-Mini-4B-Instruct",
    local_dir="./model_weights"
)

# Or download specific files
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="nvidia/Nemotron-Mini-4B-Instruct",
    filename="pytorch_model.bin",
    local_dir="./weights"
)