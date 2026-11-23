package presets

import "math"

// OutputLengthBand is one option of the output_length preset: a range of
// output tokens and the count it stands for in an expected length.
type OutputLengthBand struct {
	Key string
	// Min is the band's first token count and Max the first count past
	// it; the open top band has no Max.
	Min, Max int
	// Tokens is the count the band stands for: a bounded band's midpoint,
	// and twice the lower edge of the open top band.
	Tokens int
}

// outputLengthBands are the output_length options in order. Their edges are
// part of the preset, like its wording.
var outputLengthBands = []OutputLengthBand{
	{Key: "under_256", Min: 0, Max: 256, Tokens: 128},
	{Key: "256_to_1k", Min: 256, Max: 1024, Tokens: 640},
	{Key: "1k_to_4k", Min: 1024, Max: 4096, Tokens: 2560},
	{Key: "over_4k", Min: 4096, Tokens: 8192},
}

func outputLengthOptions() []Option {
	return []Option{
		{"under_256", "fewer than 256 tokens"},
		{"256_to_1k", "256 to 1,023 tokens"},
		{"1k_to_4k", "1,024 to 4,095 tokens"},
		{"over_4k", "4,096 tokens or more"},
	}
}

// OutputLengthBands are the output_length preset's bands in option order.
func OutputLengthBands() []OutputLengthBand {
	return append([]OutputLengthBand(nil), outputLengthBands...)
}

// ExpectedOutputTokens is the output length an output_length answer
// expects: each band's token count, bounded by capacity when capacity is
// positive, weighted by the band's probability. It is false unless the
// answer gives every band a probability in [0, 1] and some band a positive
// one.
func ExpectedOutputTokens(probabilities map[string]float64, capacity int) (float64, bool) {
	total, expected := 0.0, 0.0
	for _, band := range outputLengthBands {
		probability, ok := probabilities[band.Key]
		if !ok || math.IsNaN(probability) || probability < 0 || probability > 1 {
			return 0, false
		}
		tokens := band.Tokens
		if capacity > 0 {
			tokens = min(tokens, capacity)
		}
		total += probability
		expected += probability * float64(tokens)
	}
	if total <= 0 {
		return 0, false
	}
	return expected / total, true
}
