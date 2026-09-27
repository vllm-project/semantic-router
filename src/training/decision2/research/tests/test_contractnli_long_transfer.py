"""The ContractNLI screen's admission and grouping are fail closed."""

from research.contractnli_long_transfer import item_id, normalized, select, source_items


def _corpus():
    labels = {
        "h-e": {"hypothesis": "The agreement requires notice."},
        "h-c": {"hypothesis": "The agreement forbids notice."},
        "h-n": {"hypothesis": "The agreement names a train station."},
    }
    annotations = {
        "h-e": {"choice": "Entailment"},
        "h-c": {"choice": "Contradiction"},
        "h-n": {"choice": "NotMentioned"},
    }
    return {
        "labels": labels,
        "documents": [
            {
                "id": value,
                "text": "Agreement text " * 200,
                "annotation_sets": [{"annotations": annotations}],
            }
            for value in (1, 2)
        ],
    }


def test_long_screen_uses_strict_native_bounds_and_document_groups():
    corpus = _corpus()
    keys = {key for key, _, _ in source_items(corpus)}
    gliner = dict.fromkeys(keys, (2050, 2050))
    qwen = dict.fromkeys(keys, (2051, 2053))
    qwen[item_id(corpus["documents"][0], "h-c")] = (1024, 1025)
    qwen[item_id(corpus["documents"][1], "h-e")] = (4050, 4097)

    selected, summary = select(corpus, gliner, qwen)

    assert len(selected) == 4
    assert summary["selected_documents"] == 2
    assert summary["selected_long_documents"] == 2
    assert summary["selected_class_counts"] == {
        "entailed": 1,
        "contradicted": 1,
        "not_mentioned": 2,
    }
    assert summary["population_pass"] is False
    assert selected == select(corpus, gliner, qwen)[0]


def test_normalization_collapses_unicode_and_spacing():
    assert normalized("ＡＧＲＥＥＭＥＮＴ\n  Text") == "agreement text"
