/// Group token offsets into byte ranges of at most `budget` tokens each,
/// starting a new range every `stride` tokens.
///
/// A stride below the budget makes adjacent ranges overlap, preserving text
/// that would otherwise be split at a window boundary.
pub fn window_ranges(
    offsets: &[(usize, usize)],
    budget: usize,
    stride: usize,
) -> Vec<(usize, usize)> {
    if offsets.is_empty() || budget == 0 {
        return Vec::new();
    }

    let stride = stride.clamp(1, budget);
    let mut ranges = Vec::new();
    let mut start = 0;
    while start < offsets.len() {
        let end = (start + budget).min(offsets.len());
        ranges.push((offsets[start].0, offsets[end - 1].1));
        if end == offsets.len() {
            break;
        }
        start += stride;
    }
    ranges
}

#[cfg(test)]
mod tests {
    use super::window_ranges;

    #[test]
    fn covers_every_token() {
        let offsets = [(0, 2), (3, 5), (6, 8), (9, 11), (12, 14)];
        assert_eq!(
            window_ranges(&offsets, 2, 2),
            vec![(0, 5), (6, 11), (12, 14)]
        );
        assert_eq!(window_ranges(&offsets, 8, 4), vec![(0, 14)]);
    }

    #[test]
    fn overlaps_when_stride_is_shorter_than_budget() {
        let offsets = [(0, 2), (3, 5), (6, 8), (9, 11), (12, 14)];
        assert_eq!(window_ranges(&offsets, 4, 2), vec![(0, 11), (6, 14)]);
    }

    #[test]
    fn handles_empty_and_zero_budgets() {
        assert!(window_ranges(&[], 4, 2).is_empty());
        assert!(window_ranges(&[(0, 2)], 0, 1).is_empty());
        assert_eq!(window_ranges(&[(0, 2)], 1, 0), vec![(0, 2)]);
    }
}
