package config

import (
	"testing"

	"gopkg.in/yaml.v3"
)

func TestDecisionMediaConfig(t *testing.T) {
	for _, test := range []struct {
		body  string
		valid bool
	}{
		{"provider: model_runtime", true},
		{"provider: model_runtime\ninput: {media: {images: {max_count: 2}, videos: {max_bytes: 16000000}}}", true},
		{"provider: http\ninput: {media: {images: {}}}", false},
		{"provider: model_runtime\ninput: {media: {images: {max_count: 5}}}", false},
		{"provider: model_runtime\ninput: {media: {videos: {max_bytes: -1}}}", false},
	} {
		var d ModelDeployment
		if err := yaml.Unmarshal([]byte(test.body), &d); err != nil {
			t.Fatal(err)
		}
		if (d.ValidateDecisionMedia() == nil) != test.valid {
			t.Fatal(test.body)
		}
	}
}
