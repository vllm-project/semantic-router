//! KMeans nearest-centroid selector
//!
//! Inference-only implementation. Training is done in Python (src/training/model_selection/ml_model_selection/).
//! Models are loaded from JSON files trained by the Python scripts.
//!
//! A `format_version: 2` artifact carries a score per (cluster, candidate) and is
//! validated once at load. Unversioned artifacts load as one-hot scores.
//! Scoring is a linear scan over centroids, O(k·d) with no allocation per call.

use blake2::digest::{Update, VariableOutput};
use blake2::Blake2bVar;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

const FORMAT_VERSION: u32 = 2;
const TARGET_CONTRACT: &str = "selector.model-choice/v1";
const OBJECTIVE_VERSION: i64 = 1;
const DISTANCE: &str = "squared_l2";
const TIE_BREAK: &str = "lowest_index";
const FEATURE_NORMALIZATION: &str = "none";

/// KMeans selector over flat row-major centroid and score tables
#[derive(Debug)]
pub struct KMeansSelector {
    dim: usize,
    centroids: Vec<f64>,
    scores: Vec<f64>,
    model_names: Vec<String>,
    source: Source,
    trained: bool,
}

#[derive(Debug)]
enum Source {
    V1(KMeansModelData),
    V2(Box<KMeansV2Data>),
}

/// Unversioned model data for JSON serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KMeansModelData {
    pub algorithm: String,
    pub trained: bool,
    pub num_clusters: usize,
    pub centroids: Vec<Vec<f64>>,
    pub cluster_models: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub model_names: Vec<String>,
}

/// `format_version: 2` artifact; fields the loader does not check round-trip through `extra`
#[derive(Debug, Serialize, Deserialize)]
pub struct KMeansV2Data {
    pub algorithm: String,
    pub format_version: u32,
    pub target_contract: String,
    pub objective: ObjectiveData,
    pub candidate_set: CandidateSetData,
    pub feature: FeatureData,
    pub clustering: ClusteringData,
    pub distance: String,
    pub tie_break: String,
    pub num_clusters: usize,
    pub centroids: Vec<Vec<f64>>,
    pub cluster_sizes: Vec<u64>,
    pub scores: Vec<Vec<f64>>,
    pub support: Vec<Vec<u64>>,
    pub fallback: FallbackData,
    pub cluster_models: Vec<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ObjectiveData {
    pub id: String,
    pub version: i64,
    pub weights: ObjectiveWeights,
    pub latency_scale_ms: f64,
    pub cost_scale: f64,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ObjectiveWeights {
    pub quality: f64,
    pub latency: f64,
    pub cost: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CandidateSetData {
    pub id: String,
    pub models: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FeatureData {
    pub dim: usize,
    pub normalization: String,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ClusteringData {
    pub requested_k: usize,
    pub effective_k: usize,
    pub seed: i64,
    pub n_init: u64,
    pub max_iter: u64,
    pub tol: f64,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FallbackData {
    pub min_support: u64,
    pub global_scores: Vec<f64>,
    pub global_support: Vec<u64>,
}

/// Nearest cluster and the score of every candidate, in candidate-set order
#[derive(Debug, PartialEq)]
pub struct KMeansScore<'a> {
    pub cluster_id: usize,
    pub scores: &'a [f64],
}

/// Same digest as the trainer's `_digest`: 16-byte BLAKE2b, NUL after each part, first 16 hex chars.
fn digest<S: AsRef<str>>(parts: &[S]) -> String {
    let mut hasher = Blake2bVar::new(16).expect("16 is a valid BLAKE2b output size");
    for part in parts {
        hasher.update(part.as_ref().as_bytes());
        hasher.update(&[0]);
    }
    let mut out = [0u8; 16];
    hasher
        .finalize_variable(&mut out)
        .expect("buffer matches the output size");
    out.iter().map(|b| format!("{b:02x}")).collect::<String>()[..16].to_string()
}

/// Python's `repr(float)`: Rust's shortest Debug form with a signed, two-digit exponent.
fn py_float_repr(x: f64) -> String {
    let s = format!("{x:?}");
    match s.split_once('e') {
        Some((mantissa, exp)) => {
            let (sign, digits) = exp.strip_prefix('-').map_or(("+", exp), |d| ("-", d));
            format!("{mantissa}e{sign}{digits:0>2}")
        }
        None => s,
    }
}

impl ObjectiveData {
    fn validate(&self) -> Result<(), String> {
        let w = &self.weights;
        let weights = [w.quality, w.latency, w.cost];
        if self.version != OBJECTIVE_VERSION
            || weights.iter().any(|x| !x.is_finite() || *x < 0.0)
            || weights.iter().sum::<f64>() <= 0.0
            || !(self.latency_scale_ms.is_finite() && self.latency_scale_ms > 0.0)
            || !(self.cost_scale.is_finite() && self.cost_scale > 0.0)
        {
            return Err("Invalid KMeans objective".into());
        }
        let id = digest(&[
            format!("v{}", self.version),
            format!("q{}", py_float_repr(w.quality)),
            format!("l{}", py_float_repr(w.latency)),
            format!("c{}", py_float_repr(w.cost)),
            format!("ls{}", py_float_repr(self.latency_scale_ms)),
            format!("cs{}", py_float_repr(self.cost_scale)),
        ]);
        if self.id != id {
            return Err("KMeans objective id does not match its weights".into());
        }
        Ok(())
    }
}

/// Index of the first maximum, as np.argmax
fn argmax(row: &[f64]) -> usize {
    let mut best = 0;
    for (i, x) in row.iter().enumerate().skip(1) {
        if *x > row[best] {
            best = i;
        }
    }
    best
}

impl KMeansV2Data {
    fn validate(&self) -> Result<(), String> {
        if self.algorithm != "kmeans" || self.format_version != FORMAT_VERSION {
            return Err("Unsupported KMeans artifact format".into());
        }
        if self.target_contract != TARGET_CONTRACT
            || self.distance != DISTANCE
            || self.tie_break != TIE_BREAK
            || self.feature.normalization != FEATURE_NORMALIZATION
        {
            return Err(
                "Unsupported KMeans target contract, distance, tie-break or normalization".into(),
            );
        }
        self.objective.validate()?;
        let names = &self.candidate_set.models;
        if names.is_empty() || names.windows(2).any(|w| w[0] >= w[1]) {
            return Err("KMeans candidate set must be nonempty, sorted and unique".into());
        }
        if self.candidate_set.id != digest(names) {
            return Err("KMeans candidate set id does not match its models".into());
        }
        let (k, m, dim) = (self.centroids.len(), names.len(), self.feature.dim);
        let c = &self.clustering;
        if dim == 0
            || k == 0
            || self
                .centroids
                .iter()
                .any(|v| v.len() != dim || v.iter().any(|x| !x.is_finite()))
        {
            return Err("KMeans centroids must be finite and match the feature dimension".into());
        }
        if k != c.effective_k
            || k != self.num_clusters
            || k != self.cluster_sizes.len()
            || k > c.requested_k
            || self.cluster_sizes.contains(&0)
            || c.n_init == 0
            || c.max_iter == 0
            || !(c.tol.is_finite() && c.tol >= 0.0)
        {
            return Err("Invalid KMeans cluster counts".into());
        }
        let f = &self.fallback;
        if self.scores.len() != k
            || self.support.len() != k
            || self
                .scores
                .iter()
                .any(|r| r.len() != m || r.iter().any(|x| !x.is_finite()))
            || self.support.iter().any(|r| r.len() != m)
            || f.global_scores.len() != m
            || f.global_support.len() != m
            || f.global_scores.iter().any(|x| !x.is_finite())
            || f.global_support.contains(&0)
            || f.min_support == 0
        {
            return Err("Invalid KMeans score tables".into());
        }
        // Compared as bits, like np.array_equal on the fallback cells.
        let fallback_ok = self.scores.iter().zip(&self.support).all(|(row, support)| {
            row.iter()
                .zip(support)
                .zip(&f.global_scores)
                .all(|((s, n), g)| *n >= f.min_support || s == g)
        });
        if !fallback_ok {
            return Err("KMeans cells below min_support must hold the global score".into());
        }
        let argmax_ok = self.cluster_models.len() == k
            && self
                .scores
                .iter()
                .zip(&self.cluster_models)
                .all(|(row, name)| &names[argmax(row)] == name);
        if !argmax_ok {
            return Err("KMeans cluster_models is not the per-cluster argmax".into());
        }
        Ok(())
    }
}

impl KMeansSelector {
    /// Create an untrained KMeans selector
    pub fn new(num_clusters: usize) -> Self {
        Self {
            dim: 0,
            centroids: Vec::new(),
            scores: Vec::new(),
            model_names: Vec::new(),
            source: Source::V1(KMeansModelData {
                algorithm: "kmeans".into(),
                trained: false,
                num_clusters,
                centroids: Vec::new(),
                cluster_models: Vec::new(),
                model_names: Vec::new(),
            }),
            trained: false,
        }
    }

    /// Nearest centroid under squared L2 summed left to right, lowest index on ties
    fn nearest(&self, query: &[f64]) -> usize {
        let mut best = 0;
        let mut best_distance = f64::INFINITY;
        for (i, centroid) in self.centroids.chunks_exact(self.dim).enumerate() {
            let mut distance = 0.0;
            for (a, b) in query.iter().zip(centroid) {
                let d = a - b;
                distance += d * d;
            }
            if i == 0 || distance < best_distance {
                best = i;
                best_distance = distance;
            }
        }
        best
    }

    /// Score every candidate for a query: the nearest cluster and its score row
    pub fn score(&self, query: &[f64]) -> Result<KMeansScore<'_>, String> {
        if !self.trained {
            return Err("Model not trained".to_string());
        }
        if query.len() != self.dim || query.iter().any(|x| !x.is_finite()) {
            return Err(format!(
                "Expected {} finite KMeans features, got {}",
                self.dim,
                query.len()
            ));
        }
        let cluster_id = self.nearest(query);
        let m = self.model_names.len();
        Ok(KMeansScore {
            cluster_id,
            scores: &self.scores[cluster_id * m..(cluster_id + 1) * m],
        })
    }

    /// Select the best model over the whole candidate set
    pub fn select(&self, query: &[f64]) -> Result<String, String> {
        let scored = self.score(query)?;
        Ok(self.model_names[argmax(scored.scores)].clone())
    }

    /// Candidate names in score order
    pub fn model_names(&self) -> &[String] {
        &self.model_names
    }

    /// Check if model is trained
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// Save model to JSON in the format it was loaded from
    pub fn to_json(&self) -> Result<String, String> {
        let json = match &self.source {
            Source::V1(data) => serde_json::to_string_pretty(data),
            Source::V2(data) => serde_json::to_string_pretty(data),
        };
        json.map_err(|e| format!("JSON serialization failed: {}", e))
    }

    /// Load model from JSON, validating it once
    pub fn from_json(json: &str) -> Result<Self, String> {
        let value: Value =
            serde_json::from_str(json).map_err(|e| format!("JSON parse failed: {}", e))?;
        let version = match value.get("format_version") {
            None | Some(Value::Null) => Some(1),
            Some(v) => v.as_u64(),
        };
        match version {
            Some(1) => Self::from_v1(value),
            Some(2) => Self::from_v2(value),
            _ => Err("Unsupported KMeans artifact format".into()),
        }
    }

    fn from_v2(value: Value) -> Result<Self, String> {
        let data: KMeansV2Data = serde_json::from_value(value)
            .map_err(|e| format!("Malformed KMeans v2 artifact: {}", e))?;
        data.validate()?;
        Ok(Self {
            dim: data.feature.dim,
            centroids: data.centroids.concat(),
            scores: data.scores.concat(),
            model_names: data.candidate_set.models.clone(),
            source: Source::V2(Box::new(data)),
            trained: true,
        })
    }

    /// Unversioned artifacts score 1.0 for the cluster's model and 0.0 for the rest.
    fn from_v1(value: Value) -> Result<Self, String> {
        let data: KMeansModelData =
            serde_json::from_value(value).map_err(|e| format!("JSON parse failed: {}", e))?;
        if data.algorithm != "kmeans" {
            return Err("Unsupported KMeans artifact format".into());
        }
        let mut selector = Self::new(data.num_clusters);
        if data.centroids.is_empty() {
            selector.source = Source::V1(data);
            return Ok(selector);
        }
        let (k, dim) = (data.centroids.len(), data.centroids[0].len());
        if dim == 0
            || data
                .centroids
                .iter()
                .any(|v| v.len() != dim || v.iter().any(|x| !x.is_finite()))
            || data.cluster_models.len() < k
            || data.cluster_models.iter().any(String::is_empty)
        {
            return Err("Invalid KMeans centroids or cluster models".into());
        }
        let assigned = &data.cluster_models[..k];
        let mut names: Vec<String> = assigned.iter().chain(&data.model_names).cloned().collect();
        names.sort();
        names.dedup();
        let m = names.len();
        let mut scores = vec![0.0; k * m];
        for (i, name) in assigned.iter().enumerate() {
            let j = names
                .binary_search(name)
                .expect("assigned names are in the set");
            scores[i * m + j] = 1.0;
        }
        selector.dim = dim;
        selector.centroids = data.centroids.concat();
        selector.scores = scores;
        selector.model_names = names;
        selector.trained = data.trained;
        selector.source = Source::V1(data);
        Ok(selector)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_load_and_select() {
        // Pre-trained model with 2 clusters
        let json = r#"{
            "algorithm": "kmeans",
            "trained": true,
            "num_clusters": 2,
            "centroids": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0]
            ],
            "cluster_models": ["model-a", "model-b"]
        }"#;

        let selector = KMeansSelector::from_json(json).unwrap();
        assert!(selector.is_trained());

        // Query closer to first centroid
        let result = selector.select(&[0.9, 0.1, 0.0]).unwrap();
        assert_eq!(result, "model-a");

        // Query closer to second centroid
        let result = selector.select(&[0.1, 0.9, 0.0]).unwrap();
        assert_eq!(result, "model-b");

        let scored = selector.score(&[0.1, 0.9, 0.0]).unwrap();
        assert_eq!(scored.cluster_id, 1);
        assert_eq!(scored.scores, &[0.0, 1.0]);
        assert!(selector.score(&[0.1, 0.9]).is_err());
        assert!(selector.score(&[f64::NAN, 0.9, 0.0]).is_err());
    }

    #[test]
    fn test_kmeans_json_roundtrip() {
        let json = r#"{
            "algorithm": "kmeans",
            "trained": true,
            "num_clusters": 3,
            "centroids": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ],
            "cluster_models": ["model-a", "model-b", "model-c"]
        }"#;

        let selector = KMeansSelector::from_json(json).unwrap();
        let exported = selector.to_json().unwrap();
        let restored = KMeansSelector::from_json(&exported).unwrap();

        assert!(restored.is_trained());
        assert_eq!(restored.model_names(), ["model-a", "model-b", "model-c"]);
    }

    #[test]
    fn test_floats_parse_exactly() {
        // serde_json's default parser reads this literal one ULP low.
        let json = r#"{"algorithm": "kmeans", "trained": true, "num_clusters": 1,
            "centroids": [[0.9044128163048589]], "cluster_models": ["a"]}"#;
        let selector = KMeansSelector::from_json(json).unwrap();
        assert!(selector.to_json().unwrap().contains("0.9044128163048589"));
    }

    #[test]
    fn test_kmeans_v1_ties_and_extra_names() {
        let json = r#"{
            "algorithm": "kmeans", "trained": true, "num_clusters": 2,
            "centroids": [[-1.0], [1.0]],
            "cluster_models": ["b", "a", "unused"],
            "model_names": ["c"]
        }"#;
        let selector = KMeansSelector::from_json(json).unwrap();
        assert_eq!(selector.model_names(), ["a", "b", "c"]);
        // Equidistant from both centroids, so the lower index wins.
        let scored = selector.score(&[0.0]).unwrap();
        assert_eq!(
            (scored.cluster_id, scored.scores),
            (0, &[0.0, 1.0, 0.0][..])
        );
    }

    #[test]
    fn test_kmeans_v1_rejects() {
        for json in [
            r#"{"algorithm": "kmeans", "trained": true, "num_clusters": 2,
                "centroids": [[1.0], [0.0]], "cluster_models": ["a"]}"#,
            r#"{"algorithm": "kmeans", "trained": true, "num_clusters": 1,
                "centroids": [[1.0, 0.0], [0.0]], "cluster_models": ["a", "b"]}"#,
            r#"{"algorithm": "knn", "trained": true, "num_clusters": 1,
                "centroids": [[1.0]], "cluster_models": ["a"]}"#,
            r#"{"algorithm": "kmeans", "format_version": 3, "trained": true, "num_clusters": 1,
                "centroids": [[1.0]], "cluster_models": ["a"]}"#,
        ] {
            assert!(KMeansSelector::from_json(json).is_err(), "{json}");
        }
    }

    #[test]
    fn test_untrained_roundtrip() {
        let selector = KMeansSelector::new(4);
        let restored = KMeansSelector::from_json(&selector.to_json().unwrap()).unwrap();
        assert!(!restored.is_trained());
        assert!(restored.score(&[1.0]).is_err());
    }

    #[test]
    fn test_digest_and_float_repr_match_python() {
        // Expected strings are python3's repr(x) and the trainer's _digest.
        for (x, repr) in [
            (1e-5, "1e-05"),
            (1e16, "1e+16"),
            (1.2345678901234568e17, "1.2345678901234568e+17"),
            (0.0001, "0.0001"),
            (9999999999999998.0, "9999999999999998.0"),
            (0.1 + 0.2, "0.30000000000000004"),
            (5e-324, "5e-324"),
            (10000.0, "10000.0"),
        ] {
            assert_eq!(py_float_repr(x), repr);
        }
        assert_eq!(
            digest(&["model-0", "model-1", "model-2"]),
            "c06b436c80438781"
        );
        assert_eq!(digest(&["a", "b"]), "08b1e8b75c8635d3");
    }
}
