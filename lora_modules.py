import torch
import math
from torch import Tensor, nn
from torch.nn import Module, init
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class LoRALinear(Module):
    """
    This is basically nn.Linear, with added options for LoRA.
    Can be used in place of nn.Linear.
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

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
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        # add LoRA matrices
        if r > 0:
            self.lora_A = Parameter(torch.randn((out_features, r)))
            self.lora_B = Parameter(torch.zeros(r, in_features))
            self.scaling = alpha / r

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

        # add init for LoRA matrices
        if self.r > 0:
            init.kaiming_uniform(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # custom LoRA forward pass with updated weight
        if self.r > 0:
            lora_update = (self.lora_B @ self.lora_A) * self.scaling
            effective_weight = self.weight + lora_update
        else:
            effective_weight = self.weight

        return F.linear(input, effective_weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class LoRARobertaSelfAttention(nn.Module):
    """
    This is a custom implementation of RobertaSelfAttention
    Compared to the original, the `self.query` and `self.value` layers are replaced by LoRALinear.
    """
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # modify query and value to use LoRALinear instead of nn.Linear
        self.query = LoRALinear(
            in_features=config.hidden_size, 
            out_features=self.all_head_size, 
            r=config.lora_rank, 
            alpha=config.lora_alpha
        )
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = LoRALinear(
            in_features=config.hidden_size, 
            out_features=self.all_head_size, 
            r=config.lora_rank, 
            alpha=config.lora_alpha
        )

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)