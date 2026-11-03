# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from torch import nn

from vllm.model_executor.models.utils import WeightsMapper


class FinalizingLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(1))
        self.b = nn.Parameter(torch.zeros(1))
        self.finalized = []

    def load_weights(self, weights):
        loaded = set()
        for name, value in weights:
            getattr(self, name).data.copy_(value)
            loaded.add(name)
        self.process_weights_after_loading()
        return loaded

    def process_weights_after_loading(self):
        self.finalized.append((self.a.item(), self.b.item()))


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("vision_first", [False, True])
def test_streams_weights_with_one_finalization_and_owned_vision(vision_first):
    from vllm.models.deepseek_v4_1.nvidia.vl_model import DeepseekV41ForCausalLM

    model = object.__new__(DeepseekV41ForCausalLM)
    nn.Module.__init__(model)
    model.language_model = FinalizingLanguageModel()
    model.vision = nn.Linear(1, 1, bias=False)
    model.aligner = nn.Linear(1, 1, bias=False)
    model.hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={"text.": "language_model."}
    )

    def weights():
        buffer = torch.tensor([[2.0]])
        if vision_first:
            yield "vision.weight", buffer
        yield "text.b", torch.tensor([3.0])
        # Streaming must copy a language weight before advancing the iterator.
        assert model.language_model.b.item() == 3.0
        if not vision_first:
            yield "vision.weight", buffer
        # Deferred tensors must not alias a loader's reusable shard buffer.
        buffer.fill_(4.0)
        yield "aligner.weight", buffer
        buffer.fill_(-1.0)
        yield "text.a", torch.tensor([1.0])

    assert model.load_weights(weights()) == {
        "language_model.a",
        "language_model.b",
        "vision.weight",
        "aligner.weight",
    }
    model.process_weights_after_loading()
    assert model.language_model.finalized == [(1.0, 3.0)]
    assert model.vision.weight.item() == 2.0
    assert model.aligner.weight.item() == 4.0
