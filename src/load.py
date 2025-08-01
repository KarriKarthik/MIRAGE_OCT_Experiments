github_repo_url = "https://github.com/j-morano/MIRAGE.git"
"""
# Other commands =====
!git clone {github_repo_url}
!mv "MIRAGE" "MIRAGE_Updated" # Rename the folder
!mkdir /content/MIRAGE_Updated/update_code_package
%%writefile /content/MIRAGE_Updated/hf/__init__.py # Had to be included to be considered as package
# =============
"""

from huggingface_hub import PyTorchModelHubMixin
from MIRAGE_Updated.hf.mirage_hf import MIRAGEWrapper

class MIRAGEhf(MIRAGEWrapper, PyTorchModelHubMixin):
    def __init__(
        self,
        input_size=512,
        patch_size=32,
        modalities='bscan-slo',
        size='base',
    ):
        super().__init__(
            input_size=input_size,
            patch_size=patch_size,
            modalities=modalities,
            size=size,
        )

# For the MIRAGE model based on ViT-Base
model = MIRAGEhf.from_pretrained("j-morano/MIRAGE-Base")
# # For the MIRAGE model based on ViT-Large
# model = MIRAGEhf.from_pretrained("j-morano/MIRAGE-Large")
print("Model loaded successfully.")
