package config

import "fmt"

// DecisionMediaConfig opts a deployment into user media on Router decision calls.
// An omitted modality remains disabled; zero limits use the documented defaults.
type DecisionMediaConfig struct {
	Images *DecisionMediaLimit `yaml:"images,omitempty" json:"images,omitempty"`
	Videos *DecisionMediaLimit `yaml:"videos,omitempty" json:"videos,omitempty"`
}

type DecisionMediaLimit struct {
	MaxCount int `yaml:"max_count,omitempty" json:"max_count,omitempty"`
	MaxBytes int `yaml:"max_bytes,omitempty" json:"max_bytes,omitempty"`
}

func (d ModelDeployment) ValidateDecisionMedia() error {
	if d.Input.Media == nil {
		return nil
	}
	if d.Provider != ModelRuntimeProvider {
		return fmt.Errorf("input.media requires provider model_runtime")
	}
	for _, entry := range []struct {
		name  string
		limit *DecisionMediaLimit
		count int
	}{
		{"images", d.Input.Media.Images, 4}, {"videos", d.Input.Media.Videos, 1},
	} {
		if entry.limit == nil {
			continue
		}
		if entry.limit.MaxCount < 0 || entry.limit.MaxCount > entry.count || entry.limit.MaxBytes < 0 {
			return fmt.Errorf("input.media.%s: max_count must be between 0 (default) and %d; max_bytes must not be negative", entry.name, entry.count)
		}
	}
	return nil
}
