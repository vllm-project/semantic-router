package serving

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// fakeServices serves cards and answers classify requests like a runtime:
// sequence heads put 0.7 on the second label, scores heads score 0.2 / 0.9,
// token heads find "Tom Baker", and windowed inputs split into 4-token windows.
type fakeServices struct {
	mu       sync.Mutex
	cards    map[string]modelservice.ModelCard
	requests []modelservice.ClassifyRequest
	override func(modelservice.ClassifyRequest) (modelservice.ClassifyResponse, bool)
}

func (f *fakeServices) Card(_ context.Context, deployment string) (modelservice.ModelCard, error) {
	card, ok := f.cards[deployment]
	if !ok {
		return modelservice.ModelCard{}, modelservice.ErrUnknownDeployment
	}
	return card, nil
}

func (f *fakeServices) Classify(_ context.Context, deployment string, request modelservice.ClassifyRequest) (modelservice.ClassifyResponse, error) {
	f.mu.Lock()
	f.requests = append(f.requests, request)
	override := f.override
	f.mu.Unlock()
	if override != nil {
		if response, ok := override(request); ok {
			return response, nil
		}
	}
	head, _ := f.cards[deployment].Head(request.Head)
	result := modelservice.ClassifyResult{Input: &modelservice.InputUsage{Tokens: 6, ProcessedTokens: 6}}
	switch head.Kind {
	case kindSequence:
		result.Probabilities = []float64{0.2, 0.7, 0.1}
	case kindScores:
		result.Scores = []float64{0.2, 0.9}
	case kindToken:
		text := request.Inputs[0].Text + request.Inputs[0].Answer
		if index := strings.Index(text, "Tom Baker"); index >= 0 {
			start := len([]rune(text[:index]))
			result.Spans = []modelservice.Span{{Label: "PERSON", Start: start, End: start + 9, Text: "Tom Baker", Probability: 0.98}}
		}
	}
	if request.Overflow == "window" {
		result.Windows = []modelservice.ClassifyWindow{{Start: 0, End: 4}, {Start: 2, End: 6}}
		for i := range result.Windows {
			result.Windows[i].Probabilities = result.Probabilities
			result.Windows[i].Scores = []float64{0.2 + 0.5*float64(i), 0.9 - 0.5*float64(i)}
		}
	}
	return modelservice.ClassifyResponse{Model: deployment, Head: head.Name, Kind: head.Kind, Labels: head.Labels, Results: []modelservice.ClassifyResult{result}}, nil
}

func (f *fakeServices) Embed(context.Context, string, modelservice.EmbedRequest) (modelservice.EmbedResponse, error) {
	return modelservice.EmbedResponse{}, errors.New("not used")
}

func (f *fakeServices) Rerank(context.Context, string, modelservice.RerankRequest) (modelservice.RerankResponse, error) {
	return modelservice.RerankResponse{}, errors.New("not used")
}

func (f *fakeServices) lastRequest() modelservice.ClassifyRequest {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.requests[len(f.requests)-1]
}

func newFake() *fakeServices {
	card := func(head modelservice.HeadCard) modelservice.ModelCard {
		return modelservice.ModelCard{ID: "m", Family: "task_heads", ModelSHA256: strings.Repeat("a", 64), Surfaces: []string{"classify"}, Heads: []modelservice.HeadCard{head}, MaxInputTokens: 8192, Device: "cpu", Dtype: "float32", Ready: true}
	}
	hazard := card(modelservice.HeadCard{Name: "default", Kind: kindScores, Labels: []string{"violence", "self_harm"}, Thresholds: []float64{0.5, 0.6}, Window: &modelservice.Window{Tokens: 512, Overlap: 64}, Reduction: "max", OperatingPointSHA256: hazardPolicy})
	hazard.MaxInputTokens = 32768
	return &fakeServices{cards: map[string]modelservice.ModelCard{
		"domain": card(modelservice.HeadCard{Name: "default", Kind: kindSequence, Labels: []string{"math", "law", "other"}}),
		"hazard": hazard,
		"pii":    card(modelservice.HeadCard{Name: "default", Kind: kindToken, Labels: []string{"O", "B-PERSON", "I-PERSON"}}),
		"halu":   card(modelservice.HeadCard{Name: "default", Kind: kindToken, Labels: []string{"supported", "hallucinated"}, Inputs: []string{"grounded"}}),
		"embed":  {ID: "e", Surfaces: []string{"embeddings"}, Ready: true},
	}}
}

func spec(deployment, contract string, input config.ModelInputBudget) config.ResolvedModelBinding {
	if input.Overflow == "" {
		input.Overflow = "reject"
	}
	return config.ResolvedModelBinding{
		Recipe: "default", Name: deployment + "_consumer",
		Binding:    config.ModelBinding{Deployment: deployment, Contract: contract},
		Deployment: config.ModelDeployment{Provider: config.ModelRuntimeProvider, Artifact: "vllm-sr/x", Device: "cpu", Profile: "exact", Input: input},
	}
}

func TestSequenceBindsTheCardHeadAndForwardsTheBudget(t *testing.T) {
	fake := newFake()
	runtime := New(fake, nil)
	handle, err := runtime.Sequence(context.Background(), spec("domain", config.RemoteClassifierContractLabelDistribution, config.ModelInputBudget{MaxTokens: 512, Overflow: "truncate"}))
	if err != nil {
		t.Fatal(err)
	}
	defer handle.Close()
	capability := handle.Capability()
	if capability.Provider != Provider || capability.Device != "cpu" || capability.Precision != "float32" || strings.Join(capability.Labels, ",") != "math,law,other" {
		t.Fatalf("capability = %+v", capability)
	}
	result, err := handle.Call(context.Background(), "default", "what is 2+2?")
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Probabilities) != 3 || result.Probabilities[1] != float32(0.7) || result.Input.OriginalTokens != 6 {
		t.Fatalf("result = %+v", result)
	}
	if request := fake.lastRequest(); request.Head != "default" || request.Overflow != "truncate" || request.MaxTokens != 512 || request.Inputs[0].Text != "what is 2+2?" {
		t.Fatalf("request = %+v", request)
	}
	if len(fake.requests) != 2 {
		t.Fatalf("preparation warms up once, then the call: %d requests", len(fake.requests))
	}
	if prepared := runtime.PreparedBindings(); len(prepared) != 1 || prepared[0].Identity.Adapter != "auto" {
		t.Fatalf("prepared inventory = %+v", prepared)
	}
}

func TestLabelsAreTheServedHeadVocabulary(t *testing.T) {
	fake := newFake()
	runtime := New(fake, nil)
	domain := spec("domain", config.RemoteClassifierContractLabelDistribution, config.ModelInputBudget{})
	labels, err := runtime.Labels(context.Background(), domain)
	if err != nil || strings.Join(labels, ",") != "math,law,other" {
		t.Fatalf("labels = %v, %v", labels, err)
	}
	labels[0] = "changed"
	if again, _ := runtime.Labels(context.Background(), domain); again[0] != "math" {
		t.Fatalf("labels alias the card: %v", again)
	}
	if len(fake.requests) != 0 {
		t.Fatalf("reading labels ran the model: %d requests", len(fake.requests))
	}
	absent := domain
	absent.Binding.Head = "absent"
	for name, unservable := range map[string]config.ResolvedModelBinding{
		"unknown head": absent,
		"no heads":     spec("embed", config.RemoteClassifierContractLabelDistribution, config.ModelInputBudget{}),
	} {
		if _, err := runtime.Labels(context.Background(), unservable); !errors.Is(err, binding.ErrCapability) {
			t.Fatalf("%s: want a capability error, got %v", name, err)
		}
	}
	if _, err := New(nil, nil).Labels(context.Background(), domain); !errors.Is(err, ErrNotConfigured) {
		t.Fatalf("without services labels must fail with ErrNotConfigured, got %v", err)
	}
}

func TestPreparationRefusesBindingsTheCardCannotServe(t *testing.T) {
	runtime := New(newFake(), nil)
	cases := map[string]config.ResolvedModelBinding{
		"wrong head kind": spec("pii", config.RemoteClassifierContractLabelDistribution, config.ModelInputBudget{}),
		"no classify":     spec("embed", config.RemoteClassifierContractLabelDistribution, config.ModelInputBudget{}),
		"unknown":         spec("missing", config.RemoteClassifierContractLabelDistribution, config.ModelInputBudget{}),
		"window policy":   spec("domain", config.RemoteClassifierContractLabelDistribution, config.ModelInputBudget{MaxTokens: 4096, Overflow: "window"}),
	}
	named := spec("domain", config.RemoteClassifierContractLabelDistribution, config.ModelInputBudget{})
	named.Binding.Head = "absent"
	cases["unknown head"] = named
	legacy := spec("domain", config.RemoteClassifierContractLabelDistribution, config.ModelInputBudget{})
	legacy.Deployment.Provider = "candle"
	cases["other provider"] = legacy
	for name, binding := range cases {
		if _, err := runtime.Sequence(context.Background(), binding); err == nil {
			t.Fatalf("%s: expected a preparation error", name)
		}
	}
	if _, err := New(nil, nil).Sequence(context.Background(), spec("domain", config.RemoteClassifierContractLabelDistribution, config.ModelInputBudget{})); !errors.Is(err, ErrNotConfigured) {
		t.Fatalf("without services preparation must fail with ErrNotConfigured, got %v", err)
	}
}

func TestPreparationRefusesACardWithoutDeviceOrDtype(t *testing.T) {
	for _, field := range []string{"device", "dtype"} {
		fake := newFake()
		card := fake.cards["domain"]
		if field == "device" {
			card.Device = ""
		} else {
			card.Dtype = ""
		}
		fake.cards["domain"] = card
		_, err := New(fake, binding.NewPool()).Sequence(context.Background(), spec("domain", config.RemoteClassifierContractLabelDistribution, config.ModelInputBudget{}))
		if !errors.Is(err, binding.ErrCapability) || !strings.Contains(err.Error(), "reports no device or dtype") {
			t.Fatalf("a card without %s must be refused, got %v", field, err)
		}
	}
}

func TestCallsRejectResultsThatDisagreeWithThePreparedHead(t *testing.T) {
	fake := newFake()
	handle, err := New(fake, nil).Sequence(context.Background(), spec("domain", config.RemoteClassifierContractLabelDistribution, config.ModelInputBudget{}))
	if err != nil {
		t.Fatal(err)
	}
	fake.override = func(request modelservice.ClassifyRequest) (modelservice.ClassifyResponse, bool) {
		switch request.Inputs[0].Text {
		case "reordered":
			return modelservice.ClassifyResponse{Labels: []string{"law", "math", "other"}, Results: []modelservice.ClassifyResult{{Probabilities: []float64{0.2, 0.7, 0.1}}}}, true
		case "too long":
			return modelservice.ClassifyResponse{Labels: []string{"math", "law", "other"}, Results: []modelservice.ClassifyResult{{Error: "max_length_exceeded"}}}, true
		case "not a distribution":
			return modelservice.ClassifyResponse{Labels: []string{"math", "law", "other"}, Results: []modelservice.ClassifyResult{{Probabilities: []float64{0.9, 0.9, 0.9}}}}, true
		}
		return modelservice.ClassifyResponse{}, false
	}
	for text, want := range map[string]error{"reordered": binding.ErrInvalidResult, "too long": binding.ErrInputLimit, "not a distribution": binding.ErrInvalidResult} {
		if _, err := handle.Call(context.Background(), "default", text); !errors.Is(err, want) {
			t.Fatalf("%s: error = %v, want %v", text, err, want)
		}
	}
}

func TestTokensConvertCodePointsToBytesOnce(t *testing.T) {
	fake := newFake()
	handle, err := New(fake, nil).Tokens(context.Background(), spec("pii", config.RemoteClassifierContractTokenSpans, config.ModelInputBudget{}))
	if err != nil {
		t.Fatal(err)
	}
	text := "Grüße, ich bin Tom Baker."
	result, err := handle.Call(context.Background(), "default", text)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Entities) != 1 || text[result.Entities[0].Start:result.Entities[0].End] != "Tom Baker" || !result.HasScores() {
		t.Fatalf("entities = %+v", result.Entities)
	}
	fake.override = func(request modelservice.ClassifyRequest) (modelservice.ClassifyResponse, bool) {
		usage := &modelservice.InputUsage{Tokens: 900, ProcessedTokens: 512, Truncated: true}
		return modelservice.ClassifyResponse{Labels: []string{"O", "B-PERSON", "I-PERSON"}, Results: []modelservice.ClassifyResult{{Input: usage, Spans: []modelservice.Span{{Label: "PERSON", Start: 0, End: 3, Text: "Tom"}}}}}, true
	}
	partial, err := handle.Call(context.Background(), "default", "Tom and more")
	if !errors.Is(err, tasks.ErrTokenSpansTruncated) || len(partial.Entities) != 1 {
		t.Fatalf("a truncated scan keeps its spans and reports partial input: %+v %v", partial, err)
	}
	fake.override = func(modelservice.ClassifyRequest) (modelservice.ClassifyResponse, bool) {
		return modelservice.ClassifyResponse{Labels: []string{"O", "B-PERSON", "I-PERSON"}, Results: []modelservice.ClassifyResult{{Spans: []modelservice.Span{{Label: "PERSON", Start: 0, End: 3, Text: "Bob"}}}}}, true
	}
	if _, err := handle.Call(context.Background(), "default", "Tom and more"); !errors.Is(err, binding.ErrInvalidResult) {
		t.Fatalf("an offset disagreement must not be published, got %v", err)
	}
}

func TestGroundedSendsTheAnswerAgainstItsContext(t *testing.T) {
	fake := newFake()
	handle, err := New(fake, nil).Grounded(context.Background(), spec("halu", config.RemoteClassifierContractTokenSpans, config.ModelInputBudget{MaxTokens: 8192}), 0.4)
	if err != nil {
		t.Fatal(err)
	}
	result, err := handle.Call(context.Background(), "default", tasks.GroundedTextRequest{Context: "Ann met Bob.", Question: "Who?", Answer: "Ann met Tom Baker."})
	if err != nil {
		t.Fatal(err)
	}
	request := fake.lastRequest()
	if request.Inputs[0].Context != "Ann met Bob." || request.Inputs[0].Answer != "Ann met Tom Baker." || request.Threshold == nil || *request.Threshold != float64(float32(0.4)) {
		t.Fatalf("request = %+v", request)
	}
	if len(result.Entities) != 1 || result.Summary == nil || result.SummarySemantics.Unit != "max_hallucinated_token_score" {
		t.Fatalf("result = %+v", result)
	}
	if _, err := New(fake, nil).Grounded(context.Background(), spec("pii", config.RemoteClassifierContractTokenSpans, config.ModelInputBudget{}), 0); err == nil {
		t.Fatal("a head without grounded input cannot serve the grounded task")
	}
}

func TestWindowTasksSendGeometryAndReturnCoverage(t *testing.T) {
	fake := newFake()
	runtime := New(fake, nil)
	window := tasks.TextWindowsRequest{Size: 512, Overlap: 64}
	budget := config.ModelInputBudget{MaxTokens: 4096, Overflow: "window"}
	labels, err := runtime.SequenceWindows(context.Background(), spec("domain", config.RemoteClassifierContractLabelDistribution, budget), window)
	if err != nil {
		t.Fatal(err)
	}
	input := window
	input.Text = "a long document"
	out, err := labels.Call(context.Background(), "default", input)
	if err != nil {
		t.Fatal(err)
	}
	if request := fake.lastRequest(); request.Overflow != "window" || request.MaxTokens != 4096 || request.Window.Tokens != 512 || request.Window.Overlap != 64 {
		t.Fatalf("request = %+v", request)
	}
	if len(out.Windows) != 2 || out.ContentTokens != 6 || labels.Capability().Window.Size != 512 {
		t.Fatalf("windows = %+v", out)
	}
	if _, callErr := labels.Call(context.Background(), "default", tasks.TextWindowsRequest{Text: "x", Size: 256, Overlap: 64}); !errors.Is(callErr, binding.ErrInvalidInput) {
		t.Fatalf("a call cannot change the prepared geometry, got %v", callErr)
	}
	spans, err := runtime.TokenWindows(context.Background(), spec("pii", config.RemoteClassifierContractTokenSpans, budget), window)
	if err != nil {
		t.Fatal(err)
	}
	input.Text = "I am Tom Baker"
	tokens, err := spans.Call(context.Background(), "default", input)
	if err != nil || len(tokens.Result.Entities) != 1 || len(tokens.Windows) != 2 {
		t.Fatalf("token windows = %+v %v", tokens, err)
	}
	for name, budget := range map[string]config.ModelInputBudget{"no budget": {Overflow: "window"}, "over the card limit": {MaxTokens: 65536, Overflow: "window"}} {
		if _, err := runtime.SequenceWindows(context.Background(), spec("domain", config.RemoteClassifierContractLabelDistribution, budget), window); err == nil {
			t.Fatalf("%s: expected a preparation error", name)
		}
	}
	if _, err := runtime.TokenWindows(context.Background(), spec("pii", config.RemoteClassifierContractTokenSpans, config.ModelInputBudget{MaxTokens: 4096}), window); err == nil {
		t.Fatal("token windows require window overflow")
	}
}

func TestOperatingPointUsesTheHeadThresholdsAndWindow(t *testing.T) {
	fake := newFake()
	runtime := New(fake, nil)
	binding := spec("hazard", config.RemoteClassifierContractLabelScores, config.ModelInputBudget{MaxTokens: 32768})
	scorer, err := runtime.OperatingPoint(context.Background(), binding, []string{"violence", "self_harm"})
	if err != nil {
		t.Fatal(err)
	}
	result, err := scorer.Score(context.Background(), "default", "some text")
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprint(result.Scores) != "[0.7 0.9]" || len(result.Windows) != 2 || fmt.Sprint(scorer.Thresholds()) != "[0.5 0.6]" || len(scorer.PolicySHA256()) != 64 {
		t.Fatalf("result = %+v thresholds %v", result, scorer.Thresholds())
	}
	if request := fake.lastRequest(); request.Window.Tokens != 512 || request.Window.Overlap != 64 || request.MaxTokens != 32768 {
		t.Fatalf("the head's window applies: %+v", request)
	}
	diagnostic, err := runtime.DiagnoseScores(context.Background(), "default", binding.Name, "some text")
	if err != nil || diagnostic.Result.PolicySHA256 != scorer.PolicySHA256() {
		t.Fatalf("diagnostics use the operating point: %+v %v", diagnostic, err)
	}
	if _, err := runtime.OperatingPoint(context.Background(), binding, []string{"self_harm", "violence"}); err == nil {
		t.Fatal("thresholds are positional: the rule's labels must keep the head order")
	}
}

// hazardPolicy is the digest the fake runtime reports for Hazard's verified policy file.
const hazardPolicy = "e79a78f48bf45eb38e3f5402de3b3b18eeaa822e00b42b3640bf471276290de5"

func TestOperatingPointHonoursThePinnedPolicyDigest(t *testing.T) {
	labels := []string{"violence", "self_harm"}
	pinned := func(digest string) config.ResolvedModelBinding {
		bound := spec("hazard", config.RemoteClassifierContractLabelScores, config.ModelInputBudget{MaxTokens: 32768})
		bound.Binding.OperatingPoint = &config.OperatingPointReference{Path: "operating_point.json", SHA256: digest}
		return bound
	}
	if _, err := New(newFake(), nil).OperatingPoint(context.Background(), pinned(hazardPolicy), labels); err != nil {
		t.Fatalf("the served policy matches the pin: %v", err)
	}
	if _, err := New(newFake(), nil).OperatingPoint(context.Background(), pinned(strings.Repeat("0", 64)), labels); !errors.Is(err, binding.ErrCapability) {
		t.Fatalf("a different policy digest is refused, got %v", err)
	}
	unverified := newFake()
	unverified.cards["hazard"].Heads[0].OperatingPointSHA256 = ""
	if _, err := New(unverified, nil).OperatingPoint(context.Background(), pinned(hazardPolicy), labels); !errors.Is(err, binding.ErrCapability) {
		t.Fatalf("a pin needs a verified policy file, got %v", err)
	}
}

func TestDiagnosticsRunPreparedBindingsOfOneRecipe(t *testing.T) {
	runtime := New(newFake(), nil)
	domain := spec("domain", config.RemoteClassifierContractLabelDistribution, config.ModelInputBudget{})
	if _, err := runtime.Sequence(context.Background(), domain); err != nil {
		t.Fatal(err)
	}
	labels, err := runtime.DiagnoseLabels(context.Background(), "default", domain.Name, "text")
	if err != nil || labels.Result.Distribution == nil || labels.Binding.Identity.Deployment != "domain" {
		t.Fatalf("labels = %+v %v", labels, err)
	}
	if _, err := runtime.DiagnoseLabels(context.Background(), "other", domain.Name, "text"); !errors.Is(err, binding.ErrNotPrepared) {
		t.Fatalf("a binding is visible only in its own recipe, got %v", err)
	}
	pii := spec("pii", config.RemoteClassifierContractTokenSpans, config.ModelInputBudget{})
	if _, err := runtime.Tokens(context.Background(), pii); err != nil {
		t.Fatal(err)
	}
	if tokens, err := runtime.DiagnoseTokens(context.Background(), "default", pii.Name, "I am Tom Baker"); err != nil || len(tokens.Result.Spans.Entities) != 1 {
		t.Fatalf("tokens = %+v %v", tokens, err)
	}
}

func TestBindingsOfOneDeploymentShareItsAdmissionGate(t *testing.T) {
	runtime := New(newFake(), nil)
	first := spec("domain", config.RemoteClassifierContractLabelDistribution, config.ModelInputBudget{})
	first.Admission = config.AdmissionConfig{MaxConcurrency: 2}
	second := first
	second.Name = "classifier.topic"
	a, err := runtime.Sequence(context.Background(), first)
	if err != nil {
		t.Fatal(err)
	}
	b, err := runtime.Sequence(context.Background(), second)
	if err != nil {
		t.Fatal(err)
	}
	if a.Capability().Labels[0] != b.Capability().Labels[0] {
		t.Fatal("both bindings read the same head")
	}
	other := first
	other.Admission = config.AdmissionConfig{MaxConcurrency: 3}
	other.Name = "classifier.other"
	if _, err := runtime.Sequence(context.Background(), other); !errors.Is(err, binding.ErrCapability) {
		t.Fatalf("one deployment cannot have two admission budgets, got %v", err)
	}
}
