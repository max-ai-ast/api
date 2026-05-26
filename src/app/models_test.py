import pytest
from pydantic import ValidationError

from app.models import CandidateGenerateRequest, FeedConfig, GeneratorSpec


def _minimal_gen_request() -> CandidateGenerateRequest:
    return CandidateGenerateRequest.model_construct(
        generators=[GeneratorSpec(name="test", weight=1.0)],
        num_candidates=30,
        video_only=False,
        exclude_uris=[],
        infill=None,
    )


class TestFeedConfig:
    def test_public_defaults_to_false(self):
        cfg = FeedConfig(display_name="Test", gen_request_template=_minimal_gen_request())
        assert cfg.public is False

    def test_public_can_be_set_true(self):
        cfg = FeedConfig(display_name="Test", public=True, gen_request_template=_minimal_gen_request())
        assert cfg.public is True

    def test_rejects_display_name_over_19_chars(self):
        with pytest.raises(ValidationError):
            FeedConfig(display_name="A" * 20, gen_request_template=_minimal_gen_request())

    def test_accepts_display_name_of_exactly_19_chars(self):
        cfg = FeedConfig(display_name="A" * 19, gen_request_template=_minimal_gen_request())
        assert len(cfg.display_name) == 19
