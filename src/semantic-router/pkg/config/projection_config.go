package config

// Projections contains derived routing constructs that coordinate or synthesize
// routing outputs from base signals without redefining the detector surface.
type Projections struct {
	Partitions []ProjectionPartition `yaml:"partitions,omitempty"`
	Scores     []ProjectionScore     `yaml:"scores,omitempty"`
	Mappings   []ProjectionMapping   `yaml:"mappings,omitempty"`
}

// ProjectionPartition declares a mutually exclusive partition over existing
// domain or embedding signals.
type ProjectionPartition struct {
	Name        string   `yaml:"name"`
	Semantics   string   `yaml:"semantics"`
	Temperature float64  `yaml:"temperature,omitempty"`
	Members     []string `yaml:"members"`
	Default     string   `yaml:"default,omitempty"`
}

// ProjectionScore computes a continuous derived score from existing signals.
type ProjectionScore struct {
	Name   string                 `yaml:"name"`
	Method string                 `yaml:"method"`
	Inputs []ProjectionScoreInput `yaml:"inputs"`
}

// ProjectionScoreInput defines one weighted signal contribution.
type ProjectionScoreInput struct {
	Type        string  `yaml:"type"`
	Name        string  `yaml:"name"`
	Weight      float64 `yaml:"weight"`
	ValueSource string  `yaml:"value_source,omitempty"`
	Match       float64 `yaml:"match,omitempty"`
	Miss        float64 `yaml:"miss,omitempty"`
}

// ProjectionMapping projects a score into named routing outputs.
type ProjectionMapping struct {
	Name        string                        `yaml:"name"`
	Source      string                        `yaml:"source"`
	Method      string                        `yaml:"method"`
	Calibration *ProjectionMappingCalibration `yaml:"calibration,omitempty"`
	Outputs     []ProjectionMappingOutput     `yaml:"outputs"`
}

// ProjectionMappingCalibration controls confidence generation for matched
// threshold bands.
type ProjectionMappingCalibration struct {
	Method string  `yaml:"method"`
	Slope  float64 `yaml:"slope,omitempty"`
}

// ProjectionMappingOutput is one named band produced by a mapping.
type ProjectionMappingOutput struct {
	Name string   `yaml:"name"`
	LT   *float64 `yaml:"lt,omitempty"`
	LTE  *float64 `yaml:"lte,omitempty"`
	GT   *float64 `yaml:"gt,omitempty"`
	GTE  *float64 `yaml:"gte,omitempty"`
}
