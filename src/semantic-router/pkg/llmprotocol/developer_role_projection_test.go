package llmprotocol

import "testing"

func developerRoleRequest() Request {
	text := func(value string) []Content { return []Content{{Kind: ContentText, Text: value}} }
	return Request{
		Generation: 4,
		Trusted:    TrustedMetadata{SourceFormat: OpenAIResponsesV1},
		Instructions: []InstructionBlock{
			{Role: RoleDeveloper, Content: text("Answer in French.")},
			{Role: RoleSystem, Content: text("Be brief.")},
		},
		Messages: []Message{
			{Role: RoleDeveloper, Content: text("The secret word is PAPAYA.")},
			{Role: RoleUser, Content: text("What is the secret word?")},
		},
	}
}

// A model that lacks the developer role receives every developer instruction
// and message as a system one, with one approximated diagnostic, and the
// source bytes retire; the request it started from keeps the developer role.
func TestProjectUnsupportedDeveloperRoleSendsDeveloperAsSystem(t *testing.T) {
	request := developerRoleRequest()
	projected, diagnostics := ProjectUnsupportedDeveloperRole(request, Capabilities(CapabilityDeveloperRole), OpenAIChatV1)
	if len(diagnostics) != 1 || diagnostics[0].Field != LossyFieldSystemMessageRole ||
		diagnostics[0].Action != DiagnosticApproximated || diagnostics[0].Source != OpenAIResponsesV1 ||
		diagnostics[0].Target != OpenAIChatV1 || diagnostics[0].Reason != DeveloperRoleSentAsSystem {
		t.Fatalf("diagnostics = %+v", diagnostics)
	}
	if projected.Instructions[0].Role != RoleSystem || projected.Instructions[1].Role != RoleSystem ||
		projected.Messages[0].Role != RoleSystem || projected.Messages[1].Role != RoleUser {
		t.Fatalf("projected roles: instructions %+v, messages %+v", projected.Instructions, projected.Messages)
	}
	if projected.Generation != request.Generation+1 {
		t.Fatalf("generation = %d, want %d", projected.Generation, request.Generation+1)
	}
	if request.Instructions[0].Role != RoleDeveloper || request.Messages[0].Role != RoleDeveloper {
		t.Fatal("the source request lost its developer role")
	}
}

// A model that keeps the developer role, or a request without developer
// messages, passes through unchanged and unreported.
func TestProjectUnsupportedDeveloperRoleLeavesOtherRequestsAlone(t *testing.T) {
	request := developerRoleRequest()
	for name, unsupported := range map[string]CapabilitySet{
		"not declared":       {},
		"sampling only":      Capabilities(CapabilitySamplingMinP, CapabilitySamplingLogitBias),
		"declared, no roles": Capabilities(CapabilityDeveloperRole),
	} {
		input := request
		if name == "declared, no roles" {
			input = Request{Generation: 4, Messages: []Message{{Role: RoleSystem}, {Role: RoleUser}}}
		}
		projected, diagnostics := ProjectUnsupportedDeveloperRole(input, unsupported, OpenAIChatV1)
		if diagnostics != nil || projected.Generation != input.Generation {
			t.Fatalf("%s: diagnostics %+v, generation %d", name, diagnostics, projected.Generation)
		}
		if name != "declared, no roles" && projected.Messages[0].Role != RoleDeveloper {
			t.Fatalf("%s: the developer role was changed", name)
		}
	}
}
