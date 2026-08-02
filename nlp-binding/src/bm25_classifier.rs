//! BM25-based keyword classification using the `bm25` crate.
//!
//! Each keyword rule becomes a BM25 search engine where each keyword is a "document".
//! Input text is used as a query; keywords scoring above the threshold are considered matched.

use bm25::{Document, Language, SearchEngineBuilder, SearchResult};

/// A single keyword rule backed by a BM25 search engine.
pub struct Bm25Rule {
    pub name: String,
    pub operator: String,
    pub keywords: Vec<String>,
    pub threshold: f32,
    pub case_sensitive: bool,
    engine: bm25::SearchEngine<usize>,
}

impl Bm25Rule {
    pub fn new(
        name: String,
        operator: String,
        keywords: Vec<String>,
        threshold: f32,
        case_sensitive: bool,
    ) -> Self {
        let documents: Vec<Document<usize>> = keywords
            .iter()
            .enumerate()
            .map(|(i, kw)| Document {
                id: i,
                contents: if case_sensitive {
                    kw.clone()
                } else {
                    kw.to_lowercase()
                },
            })
            .collect();

        let engine =
            SearchEngineBuilder::<usize>::with_documents(Language::English, documents).build();

        Bm25Rule {
            name,
            operator,
            keywords,
            threshold,
            case_sensitive,
            engine,
        }
    }
}

/// BM25 keyword classifier holding multiple rules.
pub struct Bm25Classifier {
    rules: Vec<Bm25Rule>,
}

/// Result of a BM25 classification.
pub struct Bm25ClassifyResult {
    pub rule_name: String,
    pub matched_keywords: Vec<String>,
    pub scores: Vec<f32>,
    pub match_count: usize,
    pub total_keywords: usize,
}

impl Bm25Classifier {
    pub fn new() -> Self {
        Bm25Classifier { rules: Vec::new() }
    }

    pub fn add_rule(
        &mut self,
        name: String,
        operator: String,
        keywords: Vec<String>,
        threshold: f32,
        case_sensitive: bool,
    ) {
        self.rules.push(Bm25Rule::new(
            name,
            operator,
            keywords,
            threshold,
            case_sensitive,
        ));
    }

    fn classify_rule(rule: &Bm25Rule, text: &str) -> Option<Bm25ClassifyResult> {
        let query = if rule.case_sensitive {
            text.to_string()
        } else {
            text.to_lowercase()
        };

        // Search for all keywords (limit = total keywords).
        let results: Vec<SearchResult<usize>> = rule.engine.search(&query, rule.keywords.len());

        let mut matched_keywords = Vec::new();
        let mut matched_scores = Vec::new();
        for result in &results {
            if result.score as f32 >= rule.threshold {
                matched_keywords.push(rule.keywords[result.document.id].clone());
                matched_scores.push(result.score as f32);
            }
        }

        let matches = match rule.operator.as_str() {
            "OR" => !matched_keywords.is_empty(),
            "AND" => matched_keywords.len() == rule.keywords.len(),
            "NOR" => matched_keywords.is_empty(),
            _ => false,
        };
        if !matches {
            return None;
        }

        let match_count = matched_keywords.len();
        let is_nor = rule.operator == "NOR";
        Some(Bm25ClassifyResult {
            rule_name: rule.name.clone(),
            matched_keywords: if is_nor { Vec::new() } else { matched_keywords },
            scores: if is_nor { Vec::new() } else { matched_scores },
            match_count: if is_nor { 0 } else { match_count },
            total_keywords: rule.keywords.len(),
        })
    }

    /// Classify text using the legacy first-match contract.
    pub fn classify(&self, text: &str) -> Option<Bm25ClassifyResult> {
        self.rules
            .iter()
            .find_map(|rule| Self::classify_rule(rule, text))
    }

    /// Return every matching rule in declaration order.
    pub fn classify_all(&self, text: &str) -> Vec<Bm25ClassifyResult> {
        self.rules
            .iter()
            .filter_map(|rule| Self::classify_rule(rule, text))
            .collect()
    }
}
