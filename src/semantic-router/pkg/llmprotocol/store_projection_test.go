package llmprotocol

import "testing"

// store: false asks for no stored completion, so only store: true needs a
// backend that keeps completions.
func TestOnlyStoreTrueRequiresRequestStorage(t *testing.T) {
	keep, skip := true, false
	for _, tc := range []struct {
		store *bool
		want  bool
	}{{nil, false}, {&skip, false}, {&keep, true}} {
		if got := RequiredCapabilities(Request{Store: tc.store}).Supports(CapabilityRequestStorage); got != tc.want {
			t.Fatalf("store %v requires request_storage = %t, want %t", describeStore(tc.store), got, tc.want)
		}
	}
}

func TestProjectUnstoredRequestDropsStoreTrueWithItsDiagnostic(t *testing.T) {
	keep, skip := true, false
	request := Request{Store: &keep, Trusted: TrustedMetadata{SourceFormat: OpenAIChatV1}}
	projected, diagnostics := ProjectUnstoredRequest(request, AnthropicMessagesV1)
	if projected.Store != nil || projected.Generation != request.Generation+1 || request.Store == nil {
		t.Fatalf("projected store = %v generation %d, source store = %v", describeStore(projected.Store), projected.Generation, describeStore(request.Store))
	}
	if len(diagnostics) != 1 || diagnostics[0].Field != LossyFieldStore || diagnostics[0].Action != DiagnosticDropped ||
		diagnostics[0].Source != OpenAIChatV1 || diagnostics[0].Target != AnthropicMessagesV1 {
		t.Fatalf("diagnostics = %+v", diagnostics)
	}
	for _, store := range []*bool{nil, &skip} {
		unchanged, diagnostics := ProjectUnstoredRequest(Request{Store: store}, AnthropicMessagesV1)
		if unchanged.Store != store || unchanged.Generation != 0 || len(diagnostics) != 0 {
			t.Fatalf("store %v was projected: %+v %+v", describeStore(store), unchanged, diagnostics)
		}
	}
}

func describeStore(store *bool) string {
	if store == nil {
		return "absent"
	}
	if *store {
		return "true"
	}
	return "false"
}
