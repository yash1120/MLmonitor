from __future__ import annotations

from mlmonitor.models.promotion import evaluate_and_promote
from mlmonitor.models.train import load_bundle, save_bundle


def test_first_model_becomes_champion(sandbox):
    result = evaluate_and_promote()
    assert result.promoted is True
    assert "no incumbent" in result.reason
    assert load_bundle().baseline_f1 > 0.5


def test_equal_challenger_is_promoted(sandbox):
    evaluate_and_promote()  # establish champion
    result = evaluate_and_promote()  # identical data → challenger F1 == champion
    assert result.promoted is True
    assert result.challenger_f1 >= result.champion_f1


def test_worse_challenger_is_rejected(sandbox):
    evaluate_and_promote()  # establish champion
    # Inflate the on-disk champion's recorded F1 so any challenger loses on held-out F1.
    champ = load_bundle()
    champ.baseline_f1 = 0.999
    save_bundle(champ)

    result = evaluate_and_promote()
    assert result.promoted is False
    assert result.eval_basis == "heldout"
    # champion retained (its inflated F1 is still on disk)
    assert load_bundle().baseline_f1 == 0.999
