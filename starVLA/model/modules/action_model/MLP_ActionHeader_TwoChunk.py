"""TwoChunk-specific MLP action head.

This module intentionally keeps a separate state-dict layout from the
single-chunk ``MLP_ActionHeader``:

* single chunk: ``model.*``
* TwoChunk: ``token_mlp.*`` and ``output_proj.*``

Keeping the implementations separate preserves strict checkpoint loading for
both experiment families.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from starVLA.model.modules.action_model.MLP_ActionHeader import MLPResNet


class L1RegressionActionHead(nn.Module):
    """Regress a fixed action chunk from a variable-length token sequence."""

    def __init__(
        self,
        input_dim=2048,
        hidden_dim=4096,
        action_dim=7,
        NUM_ACTIONS_CHUNK=8,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.NUM_ACTIONS_CHUNK = NUM_ACTIONS_CHUNK
        self.token_mlp = MLPResNet(
            num_blocks=2,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
        )
        self.output_proj = nn.Linear(hidden_dim, action_dim)

    def _resize_seq_len(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[1] == self.NUM_ACTIONS_CHUNK:
            return hidden_states
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = F.adaptive_avg_pool1d(
            hidden_states,
            output_size=self.NUM_ACTIONS_CHUNK,
        )
        return hidden_states.transpose(1, 2)

    def predict_action(self, actions_hidden_states):
        batch_size, chunk_len, hidden_dim = actions_hidden_states.shape
        x = actions_hidden_states.reshape(batch_size * chunk_len, hidden_dim)
        x = self.token_mlp(x)
        x = x.view(batch_size, chunk_len, -1)
        x = self._resize_seq_len(x)
        return self.output_proj(x)

    def forward(self, actions_hidden_states):
        return self.predict_action(actions_hidden_states)
