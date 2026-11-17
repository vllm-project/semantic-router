package llmprotocol

import (
	"reflect"
	"slices"
	"strconv"
	"strings"
	"testing"
)

// A model can declare that its backends reject min_p and logit_bias, as a
// vLLM backend that runs speculative decoding does; nothing else can be
// declared unsupported.
func TestOnlySamplingControlsCanBeDeclaredUnsupported(t *testing.T) {
	if got := UnsupportedCapabilityNames(); !reflect.DeepEqual(got, []string{"sampling_min_p", "sampling_logit_bias"}) {
		t.Fatalf("unsupported capability names = %v", got)
	}
	declared, err := ParseUnsupportedCapabilities([]string{"sampling_logit_bias", "sampling_min_p"})
	if err != nil {
		t.Fatal(err)
	}
	if !declared.Supports(CapabilitySamplingMinP) || !declared.Supports(CapabilitySamplingLogitBias) {
		t.Fatalf("parsed %v", declared.Names())
	}
	for _, name := range []string{"image_input", "sampling_top_k", "Sampling_Min_P", "min_p", ""} {
		_, err := ParseUnsupportedCapabilities([]string{name})
		if err == nil || !strings.Contains(err.Error(), "use one of: sampling_min_p, sampling_logit_bias") {
			t.Fatalf("%q was accepted or refused without the choices: %v", name, err)
		}
	}
}

// The projection leaves out only the controls the model declares it lacks,
// reports each as dropped, and retires the source bytes that still carry
// them; the request it started from keeps both.
func TestProjectUnsupportedSamplingDropsOnlyDeclaredControls(t *testing.T) {
	minP := 0.05
	request := Request{
		Generation: 3,
		Trusted:    TrustedMetadata{SourceFormat: OpenAIChatV1},
		Sampling:   Sampling{MinP: &minP, LogitBias: map[string]float64{"42": -100}},
	}
	for _, tc := range []struct {
		name     string
		declared []Capability
		dropped  []string
	}{
		{name: "both", declared: []Capability{CapabilitySamplingMinP, CapabilitySamplingLogitBias}, dropped: []string{"min_p", "logit_bias"}},
		{name: "min_p only", declared: []Capability{CapabilitySamplingMinP}, dropped: []string{"min_p"}},
		{name: "logit_bias only", declared: []Capability{CapabilitySamplingLogitBias}, dropped: []string{"logit_bias"}},
		{name: "none"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			projected, diagnostics := ProjectUnsupportedSampling(request, Capabilities(tc.declared...), OpenAIChatV1)
			var fields []string
			for _, diagnostic := range diagnostics {
				if diagnostic.Action != DiagnosticDropped || diagnostic.Source != OpenAIChatV1 || diagnostic.Target != OpenAIChatV1 ||
					diagnostic.Reason != "the selected model does not support "+diagnostic.Field {
					t.Fatalf("diagnostic = %+v", diagnostic)
				}
				fields = append(fields, diagnostic.Field)
			}
			if !reflect.DeepEqual(fields, tc.dropped) {
				t.Fatalf("dropped %v, want %v", fields, tc.dropped)
			}
			keepsMinP, keepsBias := !slices.Contains(tc.dropped, "min_p"), !slices.Contains(tc.dropped, "logit_bias")
			if (projected.Sampling.MinP != nil) != keepsMinP || (projected.Sampling.LogitBias != nil) != keepsBias {
				t.Fatalf("projected sampling = %+v", projected.Sampling)
			}
			wantGeneration := request.Generation
			if len(tc.dropped) > 0 {
				wantGeneration++
			}
			if projected.Generation != wantGeneration {
				t.Fatalf("generation = %d, want %d", projected.Generation, wantGeneration)
			}
			if request.Sampling.MinP == nil || len(request.Sampling.LogitBias) != 1 {
				t.Fatalf("the source request lost its controls: %+v", request.Sampling)
			}
		})
	}
	if _, diagnostics := ProjectUnsupportedSampling(Request{}, Capabilities(CapabilitySamplingMinP, CapabilitySamplingLogitBias), OpenAIChatV1); diagnostics != nil {
		t.Fatalf("a request without the controls reported %+v", diagnostics)
	}
}

func TestLogitBiasRequiresItsCapabilityAndBlocks(t *testing.T) {
	request := Request{Sampling: Sampling{LogitBias: map[string]float64{"42": 5}}}
	if !RequiredCapabilities(request).Supports(CapabilitySamplingLogitBias) {
		t.Fatal("logit_bias does not require sampling_logit_bias")
	}
	if RequiredCapabilities(Request{Sampling: Sampling{LogitBias: map[string]float64{}}}).Supports(CapabilitySamplingLogitBias) {
		t.Fatal("an empty logit_bias requires sampling_logit_bias")
	}
	if blocked, err := BlockRequestField(&request, "logit_bias"); err != nil || !blocked || request.Sampling.LogitBias != nil {
		t.Fatalf("logit_bias was not blocked: blocked=%t err=%v sampling=%+v", blocked, err, request.Sampling)
	}
	if blocked, _ := BlockRequestField(&request, "logit_bias"); blocked {
		t.Fatal("blocking an absent logit_bias reported a change")
	}
}

func TestLogitBiasValidation(t *testing.T) {
	limits := DefaultPolicy().Limits
	limits.LogitBiasEntries = 3
	for _, bias := range []map[string]float64{
		{"0": -100, "42": 100, "151643": 2.5},
		{},
	} {
		request := validSemanticRequest()
		request.Sampling.LogitBias = bias
		if err := ValidateRequest(request, limits); err != nil {
			t.Fatalf("valid logit_bias %v refused: %v", bias, err)
		}
	}
	for _, bias := range []map[string]float64{
		{"token": 1},
		{"-1": 1},
		{"+1": 1},
		{"1.5": 1},
		{"": 1},
		{strconv.FormatUint(1<<32, 10): 1},
		{"1": 100.5},
		{"1": -101},
		{"1": 1, "2": 1, "3": 1, "4": 1},
	} {
		request := validSemanticRequest()
		request.Sampling.LogitBias = bias
		if err := ValidateRequest(request, limits); err == nil {
			t.Fatalf("invalid logit_bias %v accepted", bias)
		}
	}
}
