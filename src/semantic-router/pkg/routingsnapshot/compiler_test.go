package routingsnapshot

import (
	"encoding/json"
	"testing"
)

func validBundle() Bundle {
	input := "0.50"
	output := "1.50"
	return Bundle{
		NamespaceID: "ns-1", Revision: 7, Currency: "USD",
		Models: []Model{{
			ID: "mdl-frontier", Revision: 2,
			CatalogRevision: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
			Name:            "remote/frontier",
			Reasoning:       ReasoningFamily{Type: "effort", Efforts: []string{"high", "low"}},
			LoRAs:           []string{"code"},
			Execution:       ModelExecution{MaxRetries: 2, RequestTimeout: "5m", StreamTimeout: "15m"},
			Pricing:         ModelPricing{InputCostPerMillionTokens: &input, OutputCostPerMillionTokens: &output},
			Backends: []Backend{{
				ID: "be-1", ProviderID: "provider-openai-compatible", WireFormat: "openai.chat.v1",
				Origin: "https://models.example/v1", ProviderModelID: "frontier",
				Connection: BackendConnection{Path: "/chat/completions", Headers: map[string]string{"x-provider-version": "2026-08-22"}},
				Weight:     "1.0",
			}},
		}},
		Recipes: []Recipe{{
			ID: "rcp-balance", Revision: 3, Name: "balance",
			Decisions: []Decision{
				{ID: "dec-simple", Name: "Simple", DispatchCardinality: DispatchCardinalitySingle},
				{ID: "dec-agentic", Name: "Agentic", DispatchCardinality: DispatchCardinalitySingle},
			},
			Document: json.RawMessage(`{"signals":[],"decisions":[]}`),
		}},
		Entrypoints: []Entrypoint{{
			ID: "ep-blend", Revision: 4, Name: "blend", Aliases: []string{"vllm-sr/blend"},
			Rules: []EntrypointRule{
				{
					ID: "rule-premium", Name: "premium", RecipeID: "rcp-balance", RecipeRevision: 3,
					Matchers: []Matcher{{Claim: &ClaimMatcher{Name: "routing_tier", Value: ClaimValue{Kind: "string", String: "premium"}}}},
					Assignments: map[string]AssignmentSet{
						"dec-simple":  {Models: []Assignment{{ModelID: "mdl-frontier", ModelRevision: 2}}},
						"dec-agentic": {Models: []Assignment{{ModelID: "mdl-frontier", ModelRevision: 2, LoRAName: "code", Reasoning: &AssignmentReasoning{Enabled: true, Effort: "high"}}}},
					},
				},
				{
					ID: "rule-default", Name: "default", RecipeID: "rcp-balance", RecipeRevision: 3,
					Assignments: map[string]AssignmentSet{
						"dec-simple":  {Models: []Assignment{{ModelID: "mdl-frontier", ModelRevision: 2}}},
						"dec-agentic": {Models: []Assignment{{ModelID: "mdl-frontier", ModelRevision: 2}}},
					},
				},
			},
		}},
	}
}

func TestCompileIsDeterministicAndResolvesTrustedClaims(t *testing.T) {
	first, err := Compile(validBundle())
	if err != nil {
		t.Fatal(err)
	}
	second, err := Compile(validBundle())
	if err != nil {
		t.Fatal(err)
	}
	if first.Digest != second.Digest || first.SemanticDigest != second.SemanticDigest {
		t.Fatalf("digest mismatch: (%s, %s) != (%s, %s)",
			first.Digest, first.SemanticDigest, second.Digest, second.SemanticDigest)
	}

	resolution, err := first.Resolve(ResolveInput{
		Alias: "vllm-sr/blend", Path: "/v1/chat/completions",
		Claims: map[string]ClaimValue{"routing_tier": {Kind: "string", String: "premium"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resolution.Outcome != ResolveMatched || resolution.Rule == nil || resolution.Rule.ID != "rule-premium" {
		t.Fatalf("unexpected resolution: %#v", resolution)
	}
}

func TestSemanticDigestExcludesOnlyAggregatePublicationRevision(t *testing.T) {
	first, err := Compile(validBundle())
	if err != nil {
		t.Fatal(err)
	}
	changed := validBundle()
	changed.Revision++
	second, err := Compile(changed)
	if err != nil {
		t.Fatal(err)
	}
	if first.Digest == second.Digest {
		t.Fatal("aggregate publication revision did not change the exact snapshot digest")
	}
	if first.SemanticDigest != second.SemanticDigest {
		t.Fatal("aggregate publication revision changed executable routing semantics")
	}
}

func TestEntrypointRevisionParticipatesInDigest(t *testing.T) {
	first, err := Compile(validBundle())
	if err != nil {
		t.Fatal(err)
	}
	changed := validBundle()
	changed.Entrypoints[0].Revision++
	second, err := Compile(changed)
	if err != nil {
		t.Fatal(err)
	}
	if first.Digest == second.Digest {
		t.Fatal("entrypoint revision did not change the canonical digest")
	}
	if first.SemanticDigest == second.SemanticDigest {
		t.Fatal("entrypoint revision did not change executable routing semantics")
	}
}

func TestWireFormatParticipatesInDigest(t *testing.T) {
	first, err := Compile(validBundle())
	if err != nil {
		t.Fatal(err)
	}
	changed := validBundle()
	changed.Models[0].Backends[0].WireFormat = "anthropic.messages.v1"
	second, err := Compile(changed)
	if err != nil {
		t.Fatal(err)
	}
	if first.Digest == second.Digest {
		t.Fatal("wire format did not change the canonical digest")
	}
}

func TestBackendConnectionParticipatesInDigestAndIsCanonical(t *testing.T) {
	first, err := Compile(validBundle())
	if err != nil {
		t.Fatal(err)
	}
	connection := first.Models[0].Backends[0].Connection
	if connection.Headers["X-Provider-Version"] != "2026-08-22" {
		t.Fatalf("canonical connection = %+v", connection)
	}
	changed := validBundle()
	changed.Models[0].Backends[0].Connection.Path = "/v1/chat/completions"
	second, err := Compile(changed)
	if err != nil {
		t.Fatal(err)
	}
	if first.Digest == second.Digest {
		t.Fatal("backend connection did not change the canonical digest")
	}
}

func TestCompileRejectsUnsafeBackendConnection(t *testing.T) {
	for name, mutate := range map[string]func(*BackendConnection){
		"missing path": func(connection *BackendConnection) { connection.Path = "" },
		"query":        func(connection *BackendConnection) { connection.Path = "/chat/completions?secret=value" },
		"credential header": func(connection *BackendConnection) {
			connection.Headers = map[string]string{"Authorization": "Bearer secret"}
		},
		"control character": func(connection *BackendConnection) {
			connection.Headers = map[string]string{"X-Wire": "one\ntwo"}
		},
		"duplicate header": func(connection *BackendConnection) {
			connection.Headers = map[string]string{"X-Wire": "one", "x-wire": "two"}
		},
	} {
		t.Run(name, func(t *testing.T) {
			bundle := validBundle()
			mutate(&bundle.Models[0].Backends[0].Connection)
			if _, err := Compile(bundle); err == nil {
				t.Fatal("unsafe backend connection was accepted")
			}
		})
	}
}

func TestBackendConnectionRejectsSensitiveHeadersCaseInsensitively(t *testing.T) {
	for _, header := range []string{
		"Authorization",
		"proxy-authorization",
		"Cookie",
		"Set-Cookie",
		"X-API-Key",
		"x-goog-api-key",
		"X-Amz-Security-Token",
		"X-User-OpenAI-Key",
	} {
		t.Run(header, func(t *testing.T) {
			bundle := validBundle()
			bundle.Models[0].Backends[0].Connection.Headers = map[string]string{
				header: "must-not-enter-snapshot",
			}
			if _, err := Compile(bundle); err == nil {
				t.Fatalf("sensitive backend header %q was accepted", header)
			}
		})
	}
}

func TestCatalogRevisionParticipatesInDigest(t *testing.T) {
	first, err := Compile(validBundle())
	if err != nil {
		t.Fatal(err)
	}
	changed := validBundle()
	changed.Models[0].CatalogRevision = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
	second, err := Compile(changed)
	if err != nil {
		t.Fatal(err)
	}
	if first.Digest == second.Digest {
		t.Fatal("provider catalog revision did not change the canonical digest")
	}
}

func TestCompileRejectsMissingCatalogRevision(t *testing.T) {
	bundle := validBundle()
	bundle.Models[0].CatalogRevision = ""
	if _, err := Compile(bundle); err == nil {
		t.Fatal("model without provider catalog revision unexpectedly compiled")
	}
}

func TestCompileRejectsMissingWireFormat(t *testing.T) {
	bundle := validBundle()
	bundle.Models[0].Backends[0].WireFormat = ""
	if _, err := Compile(bundle); err == nil {
		t.Fatal("backend without wire format unexpectedly compiled")
	}
}

func TestCompileRejectsMissingEntrypointRevision(t *testing.T) {
	bundle := validBundle()
	bundle.Entrypoints[0].Revision = 0
	if _, err := Compile(bundle); err == nil {
		t.Fatal("missing entrypoint revision unexpectedly compiled")
	}
}

func TestCompileRejectsIncompleteAssignments(t *testing.T) {
	bundle := validBundle()
	delete(bundle.Entrypoints[0].Rules[0].Assignments, "dec-agentic")
	if _, err := Compile(bundle); err == nil {
		t.Fatal("incomplete assignments unexpectedly compiled")
	}
}

func TestCompileAcceptsCanonicalPriorityFallbackForSingleDispatch(t *testing.T) {
	bundle := validBundle()
	set := bundle.Entrypoints[0].Rules[0].Assignments["dec-simple"]
	set.Models = append(set.Models, Assignment{ModelID: "mdl-frontier", ModelRevision: 2, Priority: 1, LoRAName: "code"})
	set.Fallback = &FallbackPolicy{Strategy: "priority", On: []string{"timeout", "unavailable"}}
	bundle.Entrypoints[0].Rules[0].Assignments["dec-simple"] = set
	snapshot, err := Compile(bundle)
	if err != nil {
		t.Fatal(err)
	}
	var got AssignmentSet
	for _, rule := range snapshot.Entrypoints[0].Rules {
		if rule.ID == "rule-premium" {
			got = rule.Assignments["dec-simple"]
			break
		}
	}
	if len(got.Models) != 2 {
		t.Fatalf("canonical fallback = %#v", got)
	}
	if got.Models[1].Priority != 1 || got.Fallback == nil || len(got.Fallback.On) != 2 || got.Fallback.On[0] != "unavailable" {
		t.Fatalf("canonical fallback = %#v", got)
	}
}

func TestCompileRejectsInvalidPriorityFallback(t *testing.T) {
	for name, mutate := range map[string]func(*Bundle){
		"priority without fallback": func(bundle *Bundle) {
			set := bundle.Entrypoints[0].Rules[0].Assignments["dec-simple"]
			set.Models[0].Priority = 1
			bundle.Entrypoints[0].Rules[0].Assignments["dec-simple"] = set
		},
		"gap": func(bundle *Bundle) {
			set := bundle.Entrypoints[0].Rules[0].Assignments["dec-simple"]
			set.Models = append(set.Models, Assignment{ModelID: "mdl-frontier", ModelRevision: 2, Priority: 2, LoRAName: "code"})
			set.Fallback = &FallbackPolicy{Strategy: "priority", On: []string{"unavailable"}}
			bundle.Entrypoints[0].Rules[0].Assignments["dec-simple"] = set
		},
		"response-only overload trigger": func(bundle *Bundle) {
			set := bundle.Entrypoints[0].Rules[0].Assignments["dec-simple"]
			set.Models = append(set.Models, Assignment{ModelID: "mdl-frontier", ModelRevision: 2, Priority: 1, LoRAName: "code"})
			set.Fallback = &FallbackPolicy{Strategy: "priority", On: []string{"overloaded"}}
			bundle.Entrypoints[0].Rules[0].Assignments["dec-simple"] = set
		},
		"multi dispatch": func(bundle *Bundle) {
			bundle.Recipes[0].Decisions[0].DispatchCardinality = DispatchCardinalityMulti
			set := bundle.Entrypoints[0].Rules[0].Assignments["dec-simple"]
			set.Models = append(set.Models, Assignment{ModelID: "mdl-frontier", ModelRevision: 2, Priority: 1, LoRAName: "code"})
			set.Fallback = &FallbackPolicy{Strategy: "priority", On: []string{"unavailable"}}
			bundle.Entrypoints[0].Rules[0].Assignments["dec-simple"] = set
		},
	} {
		t.Run(name, func(t *testing.T) {
			bundle := validBundle()
			mutate(&bundle)
			if _, err := Compile(bundle); err == nil {
				t.Fatal("invalid priority fallback unexpectedly compiled")
			}
		})
	}
}

func TestCompileRejectsUnsupportedReasoningAndLoRA(t *testing.T) {
	for _, mutate := range []func(*Bundle){
		func(bundle *Bundle) {
			set := bundle.Entrypoints[0].Rules[0].Assignments["dec-agentic"]
			set.Models[0].Reasoning.Effort = "impossible"
			bundle.Entrypoints[0].Rules[0].Assignments["dec-agentic"] = set
		},
		func(bundle *Bundle) {
			set := bundle.Entrypoints[0].Rules[0].Assignments["dec-agentic"]
			set.Models[0].LoRAName = "missing"
			bundle.Entrypoints[0].Rules[0].Assignments["dec-agentic"] = set
		},
	} {
		bundle := validBundle()
		mutate(&bundle)
		if _, err := Compile(bundle); err == nil {
			t.Fatal("invalid assignment unexpectedly compiled")
		}
	}
}

func TestCompileCanonicalizesExplicitDisabledReasoning(t *testing.T) {
	bundle := validBundle()
	basic := bundle.Models[0]
	basic.ID = "mdl-basic"
	basic.Name = "remote/basic"
	basic.Reasoning = ReasoningFamily{}
	basic.LoRAs = nil
	basic.Backends = append([]Backend(nil), basic.Backends...)
	basic.Backends[0].ID = "be-basic"
	basic.Backends[0].ProviderModelID = "basic"
	bundle.Models = append(bundle.Models, basic)
	set := bundle.Entrypoints[0].Rules[0].Assignments["dec-simple"]
	set.Models[0].ModelID = basic.ID
	set.Models[0].ModelRevision = basic.Revision
	set.Models[0].Reasoning = &AssignmentReasoning{Enabled: false}
	bundle.Entrypoints[0].Rules[0].Assignments["dec-simple"] = set

	snapshot, err := Compile(bundle)
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	var assignment Assignment
	for _, rule := range snapshot.Entrypoints[0].Rules {
		if rule.ID == "rule-premium" {
			assignment = rule.Assignments["dec-simple"].Models[0]
			break
		}
	}
	if assignment.Reasoning != nil {
		t.Fatalf("explicit disabled reasoning was not canonicalized: %+v", assignment.Reasoning)
	}
}

func TestCompileRejectsDisabledReasoningWithActiveOptions(t *testing.T) {
	bundle := validBundle()
	set := bundle.Entrypoints[0].Rules[0].Assignments["dec-simple"]
	set.Models[0].Reasoning = &AssignmentReasoning{Enabled: false, Effort: "high"}
	bundle.Entrypoints[0].Rules[0].Assignments["dec-simple"] = set
	if _, err := Compile(bundle); err == nil {
		t.Fatal("disabled reasoning with an effort unexpectedly compiled")
	}
}

func TestResolveUsesSegmentAwarePrefix(t *testing.T) {
	bundle := validBundle()
	bundle.Entrypoints[0].Rules[0].Matchers = []Matcher{{PathPrefix: "/v1/chat"}}
	snapshot, err := Compile(bundle)
	if err != nil {
		t.Fatal(err)
	}
	matched, err := snapshot.Resolve(ResolveInput{Alias: "vllm-sr/blend", Path: "/v1/chat/completions"})
	if err != nil || matched.Rule == nil || matched.Rule.ID != "rule-premium" {
		t.Fatalf("expected prefix match: %#v, %v", matched, err)
	}
	notMatched, err := snapshot.Resolve(ResolveInput{Alias: "vllm-sr/blend", Path: "/v1/chatty"})
	if err != nil {
		t.Fatal(err)
	}
	if notMatched.Rule == nil || notMatched.Rule.ID != "rule-default" {
		t.Fatalf("segment prefix leaked: %#v", notMatched)
	}
}

func TestCompileRejectsDuplicateAlias(t *testing.T) {
	bundle := validBundle()
	other := bundle.Entrypoints[0]
	other.ID = "ep-other"
	other.Name = "other"
	bundle.Entrypoints = append(bundle.Entrypoints, other)
	if _, err := Compile(bundle); err == nil {
		t.Fatal("duplicate alias unexpectedly compiled")
	}
}

func TestCompileMaterializesExecutionAndCachePriceDefaults(t *testing.T) {
	bundle := validBundle()
	bundle.Models[0].Execution = ModelExecution{}
	blank := ""
	bundle.Models[0].Pricing.CacheReadCostPerMillionTokens = &blank
	bundle.Models[0].Pricing.CacheWriteCostPerMillionTokens = nil

	snapshot, err := Compile(bundle)
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	model := snapshot.Models[0]
	if model.Execution.RequestTimeout != "300s" || model.Execution.StreamTimeout != "300s" {
		t.Fatalf("execution defaults = %+v", model.Execution)
	}
	if model.Pricing.InputCostPerMillionTokens == nil ||
		model.Pricing.CacheReadCostPerMillionTokens == nil ||
		model.Pricing.CacheWriteCostPerMillionTokens == nil ||
		*model.Pricing.InputCostPerMillionTokens != "0.5" ||
		*model.Pricing.CacheReadCostPerMillionTokens != "0.5" ||
		*model.Pricing.CacheWriteCostPerMillionTokens != "0.5" {
		t.Fatalf("effective pricing = %+v", model.Pricing)
	}
}

func TestCompileRequiresCurrencyForPricedModels(t *testing.T) {
	bundle := validBundle()
	bundle.Currency = ""
	if _, err := Compile(bundle); err == nil {
		t.Fatal("priced bundle without currency unexpectedly compiled")
	}
}
