package config

import "testing"

// A model runtime binding that names no adapter reads with the one its
// model's card names, as a binding that wrote it did, so the form without an
// adapter serves the same task: Vela Halu reads a whole answer, and the
// Omni models report their adapter.
func TestModelRuntimeBindingTakesTheAdapterItsModelCardNames(t *testing.T) {
	for _, tc := range []struct{ consumer, artifact, want string }{
		{"hallucination_detector", "vllm-sr/Vela-1.0-Encoder-307M-Halu", "vela_halu"},
		{"embedding", "vllm-sr/Vela-1.0-Omni-Nano", "vela_omni"},
		{"embedding", "vllm-sr/Vela-1.0-Omni-Mini", "vela_omni"},
		{"pii_classifier", "vllm-sr/Vela-1.0-Encoder-307M-PII", ""},
		{"pii_classifier", "", ""},
	} {
		deployment := ModelDeployment{Provider: ModelRuntimeProvider, Artifact: tc.artifact}
		if got := ImpliedBinding(tc.consumer, ModelBinding{Deployment: "d"}, deployment).Adapter; got != tc.want {
			t.Errorf("%s on %q: adapter %q, want %q", tc.consumer, tc.artifact, got, tc.want)
		}
		if got := ImpliedBinding(tc.consumer, ModelBinding{Deployment: "d", Adapter: "written"}, deployment).Adapter; got != "written" {
			t.Errorf("%s on %q: the written adapter became %q", tc.consumer, tc.artifact, got)
		}
	}
}
