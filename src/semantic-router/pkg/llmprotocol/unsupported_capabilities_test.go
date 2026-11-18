package llmprotocol

import (
	"reflect"
	"strings"
	"testing"
)

// A model can declare that its backends reject min_p and logit_bias, as a
// vLLM backend that runs speculative decoding does, or lack the developer
// role, as DeepSeek V4 does; nothing else can be declared unsupported.
func TestOnlyDeclarableCapabilitiesCanBeDeclaredUnsupported(t *testing.T) {
	want := []string{"sampling_min_p", "sampling_logit_bias", "developer_role"}
	if got := UnsupportedCapabilityNames(); !reflect.DeepEqual(got, want) {
		t.Fatalf("unsupported capability names = %v, want %v", got, want)
	}
	declared, err := ParseUnsupportedCapabilities([]string{"developer_role", "sampling_logit_bias", "sampling_min_p"})
	if err != nil {
		t.Fatal(err)
	}
	if !declared.Supports(CapabilitySamplingMinP) || !declared.Supports(CapabilitySamplingLogitBias) ||
		!declared.Supports(CapabilityDeveloperRole) {
		t.Fatalf("parsed %v", declared.Names())
	}
	for _, name := range []string{"image_input", "sampling_top_k", "Sampling_Min_P", "min_p", "developer", ""} {
		_, err := ParseUnsupportedCapabilities([]string{name})
		if err == nil || !strings.Contains(err.Error(), "use one of: sampling_min_p, sampling_logit_bias, developer_role") {
			t.Fatalf("%q was accepted or refused without the choices: %v", name, err)
		}
	}
}
