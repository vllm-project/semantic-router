//! Reconcile overlapping token predictions before BIO decoding. Keep equivalent
//! to the ONNX implementation: confidence must not choose a window.

use super::sequence_windows::TokenWindow;

pub struct TokenWindowMerger {
    predictions: Vec<Option<(usize, Vec<f32>)>>,
    labels: usize,
}

impl TokenWindowMerger {
    pub fn new(content_tokens: usize, labels: usize) -> Self {
        Self {
            predictions: vec![None; content_tokens],
            labels,
        }
    }

    /// Prefer the observation with most context on its shorter side. Ties keep
    /// the earlier window; O/B/I all follow exactly the same rule.
    pub fn add(
        &mut self,
        window: &TokenWindow,
        prefix: usize,
        logits: &[Vec<f32>],
    ) -> Result<(), String> {
        if self.labels == 0
            || logits.len() != window.ids.len()
            || window.start >= window.end
            || window.end > self.predictions.len()
            || prefix + window.end - window.start > logits.len()
            || logits
                .iter()
                .any(|row| row.len() != self.labels || row.iter().any(|v| !v.is_finite()))
        {
            return Err("invalid token-window logits or original-token range".into());
        }
        for token in window.start..window.end {
            let context = (token - window.start).min(window.end - token - 1);
            let selected = &mut self.predictions[token];
            if selected
                .as_ref()
                .is_none_or(|(previous, _)| context > *previous)
            {
                *selected = Some((context, logits[prefix + token - window.start].clone()));
            }
        }
        Ok(())
    }

    pub fn finish(self) -> Result<Vec<Vec<f32>>, String> {
        self.predictions
            .into_iter()
            .map(|row| {
                row.map(|(_, values)| values)
                    .ok_or_else(|| "token windows did not observe every original token".into())
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maximum_context_keeps_one_observation_including_outside_labels() {
        let first = TokenWindow {
            start: 0,
            end: 5,
            ids: vec![0; 7],
        };
        let second = TokenWindow {
            start: 2,
            end: 7,
            ids: vec![0; 7],
        };
        let mut merger = TokenWindowMerger::new(7, 2);
        merger.add(&first, 1, &vec![vec![1., 0.]; 7]).unwrap();
        merger.add(&second, 1, &vec![vec![0., 100.]; 7]).unwrap();
        let actual = merger.finish().unwrap();
        assert_eq!(
            actual,
            vec![
                vec![1., 0.],
                vec![1., 0.],
                vec![1., 0.],
                vec![1., 0.],
                vec![0., 100.],
                vec![0., 100.],
                vec![0., 100.]
            ]
        );
    }

    #[test]
    fn missing_tail_and_invalid_window_are_errors() {
        let window = TokenWindow {
            start: 0,
            end: 2,
            ids: vec![0; 4],
        };
        let mut merger = TokenWindowMerger::new(3, 2);
        assert!(merger
            .add(&window, 1, &vec![vec![f32::NAN, 0.]; 4])
            .is_err());
        assert!(merger.add(&window, 1, &vec![vec![0., 0.]; 3]).is_err());
        merger.add(&window, 1, &vec![vec![0., 0.]; 4]).unwrap();
        assert!(merger.finish().is_err());
    }
}
