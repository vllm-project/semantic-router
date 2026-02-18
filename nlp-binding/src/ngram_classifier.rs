//! N-gram based keyword classification using the `ngrammatic` crate.
//!
//! Each keyword rule becomes a Corpus where each keyword is indexed.
//! Input text words are searched against the corpus; matches above the
//! similarity threshold are collected. This provides inherent typo tolerance
//! without explicit Levenshtein distance calculations.

use ngrammatic::{CorpusBuilder, IdentityKeyTransformer, Pad, SearchResult};

/// A single keyword rule backed by an n-gram corpus.
pub struct NgramRule {
    pub name: String,
    pub operator: String,
    pub keywords: Vec<String>,
    pub threshold: f32,
    pub case_sensitive: bool,
    pub arity: usize,
    corpus: ngrammatic::Corpus<IdentityKeyTransformer>,
}

impl NgramRule {
    pub fn new(
        name: String,
        operator: String,
        keywords: Vec<String>,
        threshold: f32,
        case_sensitive: bool,
        arity: usize,
    ) -> Self {
        let arity = if arity < 2 { 2 } else { arity };

        let mut corpus = CorpusBuilder::default()
            .arity(arity)
            .pad_full(Pad::Auto)
            .finish();

        for keyword in &keywords {
            let text = if case_sensitive {
                keyword.clone()
            } else {
                keyword.to_lowercase()
            };
            corpus.add_text(&text);
        }

        NgramRule {
            name,
            operator,
            keywords,
            threshold,
            case_sensitive,
            arity,
            corpus,
        }
    }
}

/// N-gram keyword classifier holding multiple rules.
pub struct NgramClassifier {
    rules: Vec<NgramRule>,
}

/// Result of an n-gram classification.
pub struct NgramClassifyResult {
    pub rule_name: String,
    pub matched_keywords: Vec<String>,
    pub similarities: Vec<f32>,
    pub match_count: usize,
    pub total_keywords: usize,
}

impl NgramClassifier {
    pub fn new() -> Self {
        NgramClassifier { rules: Vec::new() }
    }

    pub fn add_rule(
        &mut self,
        name: String,
        operator: String,
        keywords: Vec<String>,
        threshold: f32,
        case_sensitive: bool,
        arity: usize,
    ) {
        self.rules.push(NgramRule::new(
            name, operator, keywords, threshold, case_sensitive, arity,
        ));
    }

    /// Extract words from text for per-word matching.
    fn extract_words(text: &str) -> Vec<String> {
        text.split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
            .filter(|w| !w.is_empty())
            .map(|w| w.to_string())
            .collect()
    }

    /// Classify text against all rules (first-match semantics).
    pub fn classify(&self, text: &str) -> Option<NgramClassifyResult> {
        for rule in &self.rules {
            let input = if rule.case_sensitive {
                text.to_string()
            } else {
                text.to_lowercase()
            };

            let words = Self::extract_words(&input);

            // For each keyword, check if any word in the text matches it
            let mut matched_keywords = Vec::new();
            let mut matched_similarities = Vec::new();
            let mut matched_set = std::collections::HashSet::new();

            for word in &words {
                let results: Vec<SearchResult> =
                    rule.corpus.search(word, rule.threshold, 10);

                for result in results {
                    let matched_text = result.text.clone();
                    for (idx, kw) in rule.keywords.iter().enumerate() {
                        let kw_normalized = if rule.case_sensitive {
                            kw.clone()
                        } else {
                            kw.to_lowercase()
                        };
                        if kw_normalized == matched_text && !matched_set.contains(&idx) {
                            matched_set.insert(idx);
                            matched_keywords.push(rule.keywords[idx].clone());
                            matched_similarities.push(result.similarity);
                        }
                    }
                }
            }

            // Also try matching multi-word phrases against the full text
            if matched_keywords.len() < rule.keywords.len() {
                let results: Vec<SearchResult> =
                    rule.corpus.search(&input, rule.threshold, 10);
                for result in results {
                    let matched_text = result.text.clone();
                    for (idx, kw) in rule.keywords.iter().enumerate() {
                        let kw_normalized = if rule.case_sensitive {
                            kw.clone()
                        } else {
                            kw.to_lowercase()
                        };
                        if kw_normalized == matched_text && !matched_set.contains(&idx) {
                            matched_set.insert(idx);
                            matched_keywords.push(rule.keywords[idx].clone());
                            matched_similarities.push(result.similarity);
                        }
                    }
                }
            }

            let match_count = matched_keywords.len();

            let matches = match rule.operator.as_str() {
                "OR" => !matched_keywords.is_empty(),
                "AND" => matched_keywords.len() == rule.keywords.len(),
                "NOR" => matched_keywords.is_empty(),
                _ => false,
            };

            if matches {
                return Some(NgramClassifyResult {
                    rule_name: rule.name.clone(),
                    matched_keywords: if rule.operator == "NOR" {
                        Vec::new()
                    } else {
                        matched_keywords
                    },
                    similarities: if rule.operator == "NOR" {
                        Vec::new()
                    } else {
                        matched_similarities
                    },
                    match_count: if rule.operator == "NOR" { 0 } else { match_count },
                    total_keywords: rule.keywords.len(),
                });
            }
        }
        None
    }
}
