- Loading hf models, safetensor file, state_dict
```python
# download model files
from huggingface_hub import snapshot_download
snapshot_download(repo_id="facebook/dinov2-small", local_dir="./model")

# load model state_dict
from safetensors.torch import load_file

filepath = "./model/model.safetensors"
state_dict = load_file(filepath)
#...ensure jax impl keys matches state_dict keys
```

- model weights conversion
Check [convert_dino.py](./dino_v2_jax/convert_dino.py) for an example