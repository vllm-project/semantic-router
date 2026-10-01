package embedding

import (
	"fmt"
	"math"
)

// AudioRequest preserves the original signal until the model's processor derives
// each component input. PCM is channel-major: all frames of channel 0, then 1.
type AudioRequest struct {
	PCM        []float32
	SampleRate int
	Channels   int
	Options    Options
}

func (r AudioRequest) Validate() error {
	if r.SampleRate < 1 || r.SampleRate > 384000 {
		return fmt.Errorf("audio sample rate must be between 1 and 384000 Hz")
	}
	if r.Channels < 1 || r.Channels > 8 {
		return fmt.Errorf("audio channels must be between 1 and 8")
	}
	if len(r.PCM) == 0 || len(r.PCM)%r.Channels != 0 {
		return fmt.Errorf("audio PCM must contain complete, nonempty channels")
	}
	if len(r.PCM)/r.Channels > 30*r.SampleRate {
		return fmt.Errorf("audio exceeds 30 seconds")
	}
	if r.Options.Layer != 0 || r.Options.Dimension < 0 {
		return fmt.Errorf("audio does not support layer exits or negative dimensions")
	}
	for _, sample := range r.PCM {
		if math.IsNaN(float64(sample)) || math.IsInf(float64(sample), 0) {
			return fmt.Errorf("audio PCM samples must be finite")
		}
	}
	return nil
}
