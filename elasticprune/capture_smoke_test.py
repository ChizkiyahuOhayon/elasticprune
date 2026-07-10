"""CPU smoke test for capture reuse and selector-risk signals."""

from types import SimpleNamespace

import torch

from elasticprune.pruning import prune_generate_from_capture
from elasticprune.signals import signal_budget_boundaries, signal_pair_stability


class DummyLanguageModel:
    def __init__(self):
        self.kwargs = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return torch.tensor([[7]])


class DummyModel:
    def __init__(self):
        self.config = SimpleNamespace(image_token_index=999)
        self.language_model = DummyLanguageModel()


def main():
    model = DummyModel()
    inputs = {"input_ids": torch.tensor([[1, 999, 999, 999, 999, 2]])}
    merged = torch.arange(18, dtype=torch.float32).reshape(1, 6, 3)
    attentions = [torch.zeros(1, 1, 6, 6) for _ in range(3)]
    attentions[2][0, 0, -1] = torch.tensor([0.0, 0.4, 0.3, 0.2, 0.1, 0.0])

    output = prune_generate_from_capture(
        model,
        inputs,
        merged,
        attentions,
        keep_ratio=0.5,
        prune_layer=2,
        max_new_tokens=1,
    )
    assert output.tolist() == [[7]]
    assert model.language_model.kwargs["inputs_embeds"].shape == (1, 4, 3)
    assert model.language_model.kwargs["position_ids"].shape == (1, 4)

    scores = torch.tensor([4.0, 3.0, 2.0, 1.0])
    boundaries = signal_budget_boundaries(scores, [0.25, 0.5])
    assert boundaries["keep25_mass"] > 0
    identical = signal_pair_stability(scores, scores, keep_ratio=0.5)
    reversed_scores = signal_pair_stability(scores, scores.flip(0), keep_ratio=0.5)
    assert abs(identical["cosine"] - 1.0) < 1e-6
    assert identical["topk_jaccard"] == 1.0
    assert reversed_scores["topk_jaccard"] < 1.0
    print("capture reuse smoke test passed")


if __name__ == "__main__":
    main()
