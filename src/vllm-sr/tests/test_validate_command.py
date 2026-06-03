from types import SimpleNamespace

from cli.commands.validate import _projection_summary_lines, _signal_summary_lines


def test_signal_summary_lines_cover_v03_signal_surface():
    signals = SimpleNamespace(
        keywords=[object()],
        embeddings=[object()],
        domains=[object()],
        fact_check=[object()],
        user_feedbacks=[object()],
        reasks=[object()],
        preferences=[object()],
        language=[object()],
        context=[object()],
        structure=[object()],
        complexity=[object()],
        modality=[object()],
        role_bindings=[object()],
        jailbreak=[object()],
        pii=[object()],
        kb=[object()],
        conversation=[object()],
        events=[object()],
    )

    lines = _signal_summary_lines(signals)

    assert "  Keyword signals: 1" in lines
    assert "  Embedding signals: 1" in lines
    assert "  Domains: 1" in lines
    assert "  Fact check signals: 1" in lines
    assert "  User feedback signals: 1" in lines
    assert "  Reask signals: 1" in lines
    assert "  Preference signals: 1" in lines
    assert "  Language signals: 1" in lines
    assert "  Context signals: 1" in lines
    assert "  Structure signals: 1" in lines
    assert "  Complexity signals: 1" in lines
    assert "  Modality signals: 1" in lines
    assert "  Authz signals: 1" in lines
    assert "  Jailbreak signals: 1" in lines
    assert "  PII signals: 1" in lines
    assert "  Knowledge-base signals: 1" in lines
    assert "  Conversation signals: 1" in lines
    assert "  Event signals: 1" in lines


def test_projection_summary_lines_cover_v03_projection_surface():
    projections = SimpleNamespace(
        partitions=[object()],
        scores=[object()],
        mappings=[object()],
    )

    assert _projection_summary_lines(projections) == [
        "  Projection partitions: 1",
        "  Projection scores: 1",
        "  Projection mappings: 1",
    ]
