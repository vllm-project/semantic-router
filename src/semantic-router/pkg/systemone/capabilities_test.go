package systemone

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

const (
	boxRequest  = `{"model":"vllm-sr/auto","state":"Where is the logo?","images":["data:image/png;base64,iVBORw0KGgo="],"questions":{"where":{"type":"box","instructions":"Find the logo."}}}`
	boxResponse = `{"model":"vision-actual","answers":{"where":{"type":"box","box":{"top":0.1,"bottom":0.4,"left":0.2,"right":0.5},"confidence":0.8}},"usage":{"input_tokens":30,"output_tokens":0}}`
)

// A two-member pool with no quality policy: any complete answer is accepted.
func poolPlan() *config.AlgorithmConfig {
	return &config.AlgorithmConfig{
		Type:   config.DecisionAlgorithmEscalate,
		Budget: &config.AlgorithmBudget{Deadline: "3s", MaxCalls: 3},
		Escalate: &config.EscalateAlgorithmConfig{
			Stages: []config.EscalateStage{{Name: "text", Kind: "native", Model: "kai"}, {Name: "vision", Kind: "native", Model: "vega"}},
		},
	}
}

func textOnly(source string) Capabilities {
	return Capabilities{Source: source, QuestionTypes: []string{"choice", "noul", "score"}, Modalities: []string{"text"}, FullInput: true}
}

func vision(source string) Capabilities {
	return Capabilities{Source: source, QuestionTypes: []string{"choice", "noul", "score", "box"}, Modalities: []string{"text", "image"}, FullInput: true}
}

// members answers from a table; a model it lacks is described by nothing.
func members(table map[string]Capabilities) MemberCapabilities {
	return func(_ context.Context, model string) (Capabilities, bool) {
		capabilities, ok := table[model]
		return capabilities, ok
	}
}

func TestRequirementsNameTheTypesMediaAndFullInputATaskNeeds(t *testing.T) {
	request, err := ParseNativeRequest(json.RawMessage(boxRequest))
	if err != nil {
		t.Fatal(err)
	}
	if want := (Requirements{QuestionTypes: []string{"box"}, Modalities: []string{"image"}}); !reflect.DeepEqual(request.Requirements, want) {
		t.Fatalf("requirements %+v, want %+v", request.Requirements, want)
	}
	batch := `{"model":"vllm-sr/auto","assets":{"clip":{"type":"video","format":"mp4","data_base64":"AAAA"}},"states":{` +
		`"a":{"state":"clip","questions":{"q":{"type":"noul","instructions":"Is it raining?","require_full_input":true}}},` +
		`"b":{"state":"text","images":["data:image/png;base64,iVBORw0KGgo="],"questions":{"k":{"type":"choice","criteria":{"x":"X","y":"Y"}}}}}}`
	request, err = ParseNativeRequest(json.RawMessage(batch))
	if err != nil {
		t.Fatal(err)
	}
	want := Requirements{QuestionTypes: []string{"choice", "noul"}, Modalities: []string{"image", "video"}, FullInput: true}
	if !reflect.DeepEqual(request.Requirements, want) {
		t.Fatalf("batch requirements %+v, want %+v", request.Requirements, want)
	}
	if missing := textOnly("card").Missing(want); !reflect.DeepEqual(missing, []string{"image", "video"}) {
		t.Fatalf("missing %v", missing)
	}
	if missing := (Capabilities{QuestionTypes: []string{"choice"}, Modalities: []string{"text", "image", "video"}}).Missing(want); !reflect.DeepEqual(missing, []string{"noul", CapabilityFullInput}) {
		t.Fatalf("missing %v", missing)
	}
}

func TestBoxAnswersAreCheckedByTheirEdges(t *testing.T) {
	question := Question{Name: "where", Type: "box"}
	for _, tc := range []struct {
		answer string
		valid  bool
	}{
		{`{"type":"box","box":{"top":0.1,"bottom":0.4,"left":0.2,"right":0.5},"confidence":0.8}`, true},
		{`{"type":"box","box":{"top":0,"bottom":1,"left":0,"right":1},"confidence":null}`, true},
		{`{"type":"box","box":{"top":0.5,"bottom":0.4,"left":0.2,"right":0.5},"confidence":0.8}`, false},
		{`{"type":"box","box":{"top":0.1,"bottom":1.2,"left":0.2,"right":0.5},"confidence":0.8}`, false},
		{`{"type":"box","box":{"top":0.1,"bottom":0.4,"left":0.2},"confidence":0.8}`, false},
		{`{"type":"box","confidence":0.8}`, false},
		{`{"type":"choice","choice":"x"}`, false},
	} {
		if got := observeAnswer(question, json.RawMessage(tc.answer)); got.Valid != tc.valid {
			t.Errorf("%s: valid %v, want %v", tc.answer, got.Valid, tc.valid)
		}
	}
	if got := observeAnswer(question, json.RawMessage(`{"type":"box","box":{"top":0.1,"bottom":0.4,"left":0.2,"right":0.5},"confidence":0.8}`)); got.Confidence == nil || *got.Confidence != 0.8 {
		t.Fatalf("a box answer's confidence %v", got.Confidence)
	}
}

// A box question goes only to the member that answers box questions and reads
// images; the other is skipped without a call, and counted.
func TestABoxQuestionGoesOnlyToMembersThatAnswerIt(t *testing.T) {
	request, err := ParseNativeRequest(json.RawMessage(boxRequest))
	if err != nil {
		t.Fatal(err)
	}
	executor, err := NewExecutor(poolPlan(), nil)
	if err != nil {
		t.Fatal(err)
	}
	skippedBefore := testutil.ToFloat64(metrics.SystemOneStageTotal.WithLabelValues("escalate", "text", "kai", "incapable"))
	var calls []string
	invoke := func(_ context.Context, model string, _ json.RawMessage) (int, []byte, error) {
		calls = append(calls, model)
		return 200, []byte(boxResponse), nil
	}
	ctx := withMembers(context.Background(), members(map[string]Capabilities{"kai": textOnly("card"), "vega": vision("declared")}))
	result, err := executor.Execute(ctx, request, invoke)
	if err != nil || result.Model != "vega" || result.Stage != "vision" {
		t.Fatalf("result %+v, error %v", result, err)
	}
	if !reflect.DeepEqual(calls, []string{"vega"}) {
		t.Fatalf("calls %v", calls)
	}
	if got := testutil.ToFloat64(metrics.SystemOneStageTotal.WithLabelValues("escalate", "text", "kai", "incapable")) - skippedBefore; got != 1 {
		t.Fatalf("incapable stage count %v", got)
	}
}

// When no member answers what a request needs, it goes to none: the error
// names what is missing and each member with its source.
func TestNoCapableMemberRefusesTheRequestBeforeAnyCall(t *testing.T) {
	request, err := ParseNativeRequest(json.RawMessage(boxRequest))
	if err != nil {
		t.Fatal(err)
	}
	executor, err := NewExecutor(poolPlan(), nil)
	if err != nil {
		t.Fatal(err)
	}
	invoke := func(context.Context, string, json.RawMessage) (int, []byte, error) {
		t.Fatal("an incapable member was asked")
		return 0, nil, nil
	}
	lacking := vision("listing")
	lacking.Modalities = []string{"text"}
	ctx := withMembers(context.Background(), members(map[string]Capabilities{"kai": textOnly("card"), "vega": lacking}))
	_, err = executor.Execute(ctx, request, invoke)
	var refusal *IncapableError
	if !errors.As(err, &refusal) {
		t.Fatalf("error %v", err)
	}
	if !reflect.DeepEqual(refusal.Missing, []string{"box", "image"}) {
		t.Fatalf("missing %v", refusal.Missing)
	}
	refusal.Decision = "answer"
	for _, want := range []string{`decision "answer"`, "needs box, image", "kai lacks box, image (card)", "vega lacks image (listing)"} {
		if !strings.Contains(refusal.Error(), want) {
			t.Fatalf("%q lacks %q", refusal.Error(), want)
		}
	}
}

// A member nothing describes is asked as before, and so is every member for a
// request every System One model answers.
func TestUndescribedMembersAndStandardRequestsAreAskedAsBefore(t *testing.T) {
	executor, err := NewExecutor(poolPlan(), nil)
	if err != nil {
		t.Fatal(err)
	}
	box, _ := ParseNativeRequest(json.RawMessage(boxRequest))
	var calls []string
	invoke := func(_ context.Context, model string, _ json.RawMessage) (int, []byte, error) {
		calls = append(calls, model)
		return 200, []byte(boxResponse), nil
	}
	ctx := withMembers(context.Background(), members(map[string]Capabilities{"vega": textOnly("listing")}))
	if _, err := executor.Execute(ctx, box, invoke); err != nil || !reflect.DeepEqual(calls, []string{"kai"}) {
		t.Fatalf("calls %v, error %v", calls, err)
	}
	standard, _ := ParseNativeRequest(json.RawMessage(singleRequest))
	calls = nil
	invoke = func(_ context.Context, model string, _ json.RawMessage) (int, []byte, error) {
		calls = append(calls, model)
		return 200, []byte(certainResponse), nil
	}
	ctx = withMembers(context.Background(), members(map[string]Capabilities{"kai": {Source: "card"}, "vega": {Source: "card"}}))
	if _, err := executor.Execute(ctx, standard, invoke); err != nil || !reflect.DeepEqual(calls, []string{"kai"}) {
		t.Fatalf("a standard request: calls %v, error %v", calls, err)
	}
}

type fakeCards map[string]modelservice.ModelCard

func (c fakeCards) Card(deployment string) (modelservice.ModelCard, bool) {
	card, ok := c[deployment]
	return card, ok
}

func (fakeCards) Listings() *BackendListings { return nil }

// A member that names a deployment is described by the deployment's card when
// there is one, else by what the deployment declares; a backend alias without
// a listing is described by nothing.
func TestAMemberIsDescribedByItsCardElseItsDeclaration(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.ModelConfig = map[string]config.ModelParams{
		"listed":   {APIFormat: config.APIFormatSystemOne, Deployment: "remote-a"},
		"declared": {APIFormat: config.APIFormatSystemOne, Deployment: "remote-b"},
		"plain":    {APIFormat: config.APIFormatSystemOne},
	}
	declared := &config.DeploymentCapabilities{QuestionTypes: []string{"choice", "noul", "score", "box"}, Modalities: []string{"text", "image"}}
	cfg.ModelDeployments = map[string]config.ModelDeployment{
		"remote-a": {Provider: config.SystemOneProvider, Model: "a", Schema: "vllm-sr/5.0", Capabilities: declared},
		"remote-b": {Provider: config.SystemOneProvider, Model: "b", Schema: "systemone/0.2", Capabilities: declared},
	}
	cards := fakeCards{"remote-a": modelservice.RemoteCard("a", []string{"choice", "noul", "score"}, nil)}
	find := poolMembers(cfg, "native", cards, nil)
	listed, ok := find(context.Background(), "listed")
	if !ok || listed.Source != CapabilitySourceCard || !reflect.DeepEqual(listed.QuestionTypes, []string{"choice", "noul", "score"}) ||
		!reflect.DeepEqual(listed.Modalities, []string{"text", "image"}) || !listed.FullInput {
		t.Fatalf("listed %+v", listed)
	}
	fromConfig, ok := find(context.Background(), "declared")
	if !ok || fromConfig.Source != CapabilitySourceDeclared || !reflect.DeepEqual(fromConfig.QuestionTypes, declared.QuestionTypes) || fromConfig.FullInput {
		t.Fatalf("declared %+v", fromConfig)
	}
	if capabilities, ok := find(context.Background(), "plain"); ok {
		t.Fatalf("an undescribed backend %+v", capabilities)
	}
}

// A backend's listing is read once, the first request waiting for it; a
// listing that cannot be read describes nothing.
func TestABackendListingIsReadOnce(t *testing.T) {
	var listings BackendListings
	var reads atomic.Int32
	read := func(context.Context) (Capabilities, error) {
		reads.Add(1)
		return vision(CapabilitySourceListing), nil
	}
	for range 3 {
		if capabilities, ok := listings.capabilities(context.Background(), "vision", read); !ok || capabilities.Source != CapabilitySourceListing {
			t.Fatalf("listing %+v %v", capabilities, ok)
		}
	}
	if reads.Load() != 1 {
		t.Fatalf("reads %d", reads.Load())
	}
	failing := func(context.Context) (Capabilities, error) {
		reads.Add(1)
		return Capabilities{}, errors.New("no listing")
	}
	for range 2 {
		if _, ok := listings.capabilities(context.Background(), "plain", failing); ok {
			t.Fatal("a failed listing describes a member")
		}
	}
	if reads.Load() != 2 {
		t.Fatalf("reads %d", reads.Load())
	}
}

func TestTheListingSitsWhereTheSchemaLists(t *testing.T) {
	for _, tc := range []struct{ create, listing, want string }{
		{"/v1/systemone", "models", "/v1/models"},
		{"/gateway/v1/systemone", "models", "/gateway/v1/models"},
		{"/v1/systemone", "/models", "/v1/systemone/models"},
		{"/v1/systemone/", "/models", "/v1/systemone/models"},
	} {
		if got := listingPath(tc.create, tc.listing); got != tc.want {
			t.Errorf("listingPath(%q, %q) = %q, want %q", tc.create, tc.listing, got, tc.want)
		}
	}
}
