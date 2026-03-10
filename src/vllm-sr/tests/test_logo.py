from cli.logo import build_vllm_logo_lines, print_vllm_logo


def test_build_vllm_logo_lines_matches_current_wordmark():
    lines = build_vllm_logo_lines()
    rendered = "\n".join(lines)

    assert "Semantic Router" in rendered
    assert "local runtime" in rendered
    assert "########" not in rendered
    assert "█" in rendered


def test_print_vllm_logo_writes_banner(capsys):
    print_vllm_logo()

    output = capsys.readouterr().out

    assert "Semantic Router" in output
    assert "local runtime" in output
