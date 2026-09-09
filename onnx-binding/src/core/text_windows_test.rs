use super::text_windows;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::processors::template::TemplateProcessing;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

fn tokenizer() -> Tokenizer {
    let model = WordLevel::builder()
        .vocab(
            [
                ("[UNK]".to_string(), 0),
                ("[CLS]".to_string(), 1),
                ("[SEP]".to_string(), 2),
            ]
            .into_iter()
            .collect(),
        )
        .unk_token("[UNK]".to_string())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(Whitespace));
    tokenizer.with_post_processor(Some(
        TemplateProcessing::builder()
            .try_single("[CLS] $A [SEP]")
            .unwrap()
            .special_tokens(vec![("[CLS]", 1), ("[SEP]", 2)])
            .build()
            .unwrap(),
    ));
    tokenizer
}

#[test]
fn text_windows_cover_tail_and_overlap() {
    let tokenizer = tokenizer();
    let text = "one two three four five six seven";
    let ranges = text_windows(&tokenizer, text, 6, None).unwrap();
    let chunks: Vec<_> = ranges
        .iter()
        .map(|&(start, end)| &text[start..end])
        .collect();
    assert_eq!(
        chunks,
        [
            "one two three four",
            "three four five six",
            "five six seven"
        ]
    );
    for chunk in chunks {
        assert!(tokenizer.encode(chunk, true).unwrap().len() <= 6);
    }
}

#[test]
fn text_windows_preserve_short_and_empty_inputs() {
    let tokenizer = tokenizer();
    for text in ["word", "  two words  "] {
        assert_eq!(
            text_windows(&tokenizer, text, 6, None).unwrap(),
            [(0, text.len())]
        );
    }
    for text in ["", "   "] {
        assert!(text_windows(&tokenizer, text, 6, None).unwrap().is_empty());
    }
}

#[test]
fn text_windows_use_utf8_byte_offsets() {
    let tokenizer = tokenizer();
    let text = "你好 世界 café 😀 再见";
    let ranges = text_windows(&tokenizer, text, 4, None).unwrap();
    assert_eq!(ranges.first().unwrap().0, 0);
    assert_eq!(ranges.last().unwrap().1, text.len());
    for &(start, end) in &ranges {
        assert!(text.is_char_boundary(start));
        assert!(text.is_char_boundary(end));
        assert!(tokenizer.encode(&text[start..end], true).unwrap().len() <= 4);
    }
    for pair in ranges.windows(2) {
        assert!(pair[1].0 < pair[0].1);
    }
}

#[test]
fn text_windows_respect_model_tokenizer_and_requested_limits() {
    let text = "one two three four five six seven";
    let mut tokenizer = tokenizer();
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: 6,
            ..Default::default()
        }))
        .unwrap();
    tokenizer.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::Fixed(12),
        ..Default::default()
    }));
    for (model_window, requested, expected_first) in [
        (32, None, "one two three four"),
        (5, None, "one two three"),
        (32, Some(4), "one two"),
        (32, Some(64), "one two three four"),
    ] {
        let ranges = text_windows(&tokenizer, text, model_window, requested).unwrap();
        assert_eq!(&text[ranges[0].0..ranges[0].1], expected_first);
        assert_eq!(ranges.last().unwrap().1, text.len());
    }
    assert_eq!(tokenizer.get_truncation().unwrap().max_length, 6);
    assert!(matches!(
        tokenizer.get_padding().unwrap().strategy,
        PaddingStrategy::Fixed(12)
    ));
}

#[test]
fn text_windows_reject_limits_without_content_budget() {
    let tokenizer = tokenizer();
    for window in [0, 1, 2] {
        assert!(text_windows(&tokenizer, "word", window, None).is_err());
    }
    assert_eq!(
        text_windows(&tokenizer, "one two", 3, None).unwrap(),
        [(0, 3), (4, 7)]
    );
}

#[test]
fn text_windows_use_model_limit_and_actual_special_token_count() {
    let mut tokenizer = tokenizer();
    let text = "word ".repeat(600);
    assert_eq!(
        text_windows(&tokenizer, &text, 1024, None).unwrap(),
        [(0, text.len())]
    );
    tokenizer.with_post_processor(Some(
        TemplateProcessing::builder()
            .try_single("[CLS] $A [SEP] [SEP]")
            .unwrap()
            .special_tokens(vec![("[CLS]", 1), ("[SEP]", 2)])
            .build()
            .unwrap(),
    ));
    let ranges = text_windows(&tokenizer, "one two three", 5, None).unwrap();
    assert_eq!(ranges, [(0, 7), (4, 13)]);
}

#[test]
fn text_windows_recheck_substring_tokenization() {
    use tokenizers::models::wordpiece::WordPiece;
    let mut tokenizer = tokenizer();
    let vocab = [
        ("[UNK]", 0),
        ("[CLS]", 1),
        ("[SEP]", 2),
        ("abc", 3),
        ("##de", 4),
        ("##f", 5),
        ("d", 6),
        ("##e", 7),
        ("f", 8),
    ]
    .map(|(token, id)| (token.to_string(), id));
    tokenizer.with_model(WordPiece::builder().vocab(vocab).build().unwrap());
    let text = "abcdef";
    let ranges = text_windows(&tokenizer, text, 4, None).unwrap();
    assert_eq!(ranges, [(0, 5), (3, 5), (5, 6)]);
    for (start, end) in ranges {
        assert!(tokenizer.encode(&text[start..end], true).unwrap().len() <= 4);
    }
}
