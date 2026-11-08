"""The gates run this suite on every pull request, so it must never fetch the model."""

from huggingface_hub import constants


def test_the_hub_is_offline_for_the_whole_run() -> None:
    assert constants.HF_HUB_OFFLINE is True
