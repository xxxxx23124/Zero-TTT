from __future__ import annotations

import torch

from zero_ttt.config import load_config
from zero_ttt.inference import InferenceBatch, PublicationPositionEvaluator
from zero_ttt.model import PolicyValueTransformer
from zero_ttt.training.checkpoint import CheckpointManager, checkpoint_metadata


def test_publication_evaluator_loads_and_hides_fixed_padding(tmp_path) -> None:
    config = load_config("configs/test.toml")
    model = PolicyValueTransformer(config.model)
    manager = CheckpointManager(tmp_path, keep=1)
    publication = manager.save_publication(
        "test-run",
        3,
        8,
        model.state_dict(),
        checkpoint_metadata(config.canonical_json(), config.sha256),
    )
    evaluator = PublicationPositionEvaluator(
        publication,
        device="cpu",
        inference_batch_size=16,
    )
    single = InferenceBatch(
        board=torch.zeros(1, 25, 19, 19),
        global_features=torch.zeros(1, 5),
        legal=torch.cat(
            (torch.zeros(1, 1, dtype=torch.bool), torch.ones(1, 361, dtype=torch.bool)),
            dim=1,
        ),
    )
    full = InferenceBatch(
        board=single.board.expand(16, -1, -1, -1).clone(),
        global_features=single.global_features.expand(16, -1).clone(),
        legal=single.legal.expand(16, -1).clone(),
    )
    output = evaluator.evaluate(single)
    full_output = evaluator.evaluate(full)
    assert output.policy_logits.shape == (1, 362)
    assert output.value.shape == (1,)
    torch.testing.assert_close(output.policy_logits[0], full_output.policy_logits[0])
    torch.testing.assert_close(output.value[0], full_output.value[0])
    assert torch.softmax(output.policy_logits, dim=-1)[0, 0].item() == 0.0
    assert evaluator.publication_sha256 in evaluator.model_version
