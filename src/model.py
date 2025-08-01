import torch.nn as nn

# Remove the 'slo' adapter from the ModuleDict
del model.model.input_adapters['slo']


# Redefine the head 
model.head = nn.Sequential(
    nn.Linear(768, 256),
    nn.GELU(approximate='none'),
    nn.BatchNorm1d(256),
    nn.Dropout(p=0.5),
    nn.Linear(256, 128),
    nn.GELU(approximate='none'),
    nn.BatchNorm1d(128),
    nn.Dropout(p=0.5),
    nn.Linear(128, 4) # Output raw scores
)

# Freeze the original model's weights 
# This stops gradients from being calculated for the base model
for param in model.model.parameters():
    param.requires_grad = False

# Xavier initialization to the new head (linear layers)
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

model.head.apply(initialize_weights)

# Verify updated model defination
for name, param in model.named_parameters():
    print(f"{name} | Trainable: {param.requires_grad}")
