import torch
import math
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from transformers.models.roberta.modeling_roberta import RobertaSdpaSelfAttention, RobertaSelfAttention

class LoRALinear(nn.Linear):
    """
    This is nn.Linear with added options for LoRA.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        r=0, # LoRA rank
        alpha=1.0 # scaling factor
    ) -> None:
        self.factory_kwargs = {"device": device, "dtype": dtype}
        # add LoRA attributes
        self.r = r
        self.alpha = alpha

        super().__init__(in_features, out_features, bias, device, dtype)

    def reset_parameters(self) -> None:
        # Reset base nn.Linear parameters
        super().reset_parameters()

        # initiate LoRA parameters
        if self.r > 0:
            self.lora_B = Parameter(torch.randn((self.out_features, self.r), **self.factory_kwargs))
            self.lora_A = Parameter(torch.zeros((self.r, self.in_features), **self.factory_kwargs))
            self.scaling = self.alpha / self.r
            init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            init.zeros_(self.lora_B)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # custom LoRA forward pass with updated weight
        if self.r > 0:
            lora_update = (self.lora_B @ self.lora_A) * self.scaling
            effective_weight = self.weight + lora_update
        else:
            effective_weight = self.weight

        return F.linear(input, effective_weight, self.bias)


class LoRARobertaSelfAttention(RobertaSelfAttention):
    """
    This is a custom RobertaSelfAttention that is initialized with LoRALinear on `query` and `value` layers.
    """
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type)
        
        # modify query and value to use LoRALinear instead of nn.Linear
        self.query = LoRALinear(
            in_features=config.hidden_size, 
            out_features=self.all_head_size, 
            r=config.lora_rank, 
            alpha=config.lora_alpha
        )
        self.value = LoRALinear(
            in_features=config.hidden_size, 
            out_features=self.all_head_size, 
            r=config.lora_rank, 
            alpha=config.lora_alpha
        )

# Define the custom RobertaSdpaSelfAttention class that inherits from custom LoRA classes
LoRARobertaSdpaSelfAttention = type(
    'LoRARobertaSdpaSelfAttention',
    (LoRARobertaSelfAttention, RobertaSdpaSelfAttention),
    {}
)