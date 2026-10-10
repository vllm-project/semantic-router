package llmprotocol

import "testing"

func TestZeroSamplingPenaltiesDoNotRequireBackendSupport(t *testing.T) {
	zero, nonzero := 0.0, 0.25
	for _, test := range []struct {
		name      string
		frequency *float64
		presence  *float64
		want      bool
	}{
		{name: "omitted"},
		{name: "explicit zeros", frequency: &zero, presence: &zero},
		{name: "frequency penalty", frequency: &nonzero, presence: &zero, want: true},
		{name: "presence penalty", frequency: &zero, presence: &nonzero, want: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			request := Request{Sampling: Sampling{FrequencyPenalty: test.frequency, PresencePenalty: test.presence}}
			if got := RequiredCapabilities(request).Supports(CapabilitySamplingPenalties); got != test.want {
				t.Fatalf("sampling_penalties required = %t, want %t", got, test.want)
			}
		})
	}
}
