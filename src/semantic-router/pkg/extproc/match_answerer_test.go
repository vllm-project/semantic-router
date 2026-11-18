package extproc

import (
	"context"
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

type answererDecider struct {
	asked []string
	cards map[string]modelservice.ModelCard
}

func (d *answererDecider) Decide(_ context.Context, deployment string, _ modelservice.Request) (modelservice.Response, error) {
	d.asked = append(d.asked, deployment)
	return modelservice.Response{}, nil
}

func (d *answererDecider) Card(name string) (modelservice.ModelCard, bool) {
	card, ok := d.cards[name]
	return card, ok
}

func (d *answererDecider) PrepareCard(_ context.Context, name string) (modelservice.ModelCard, error) {
	card, ok := d.cards[name]
	if !ok {
		return modelservice.ModelCard{}, modelservice.ErrUnavailable
	}
	return card, nil
}

func TestMatchAnswererAsksEachDeploymentThroughItsProvider(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.ModelDeployments = map[string]config.ModelDeployment{
		"remote": {Provider: config.SystemOneProvider, Schema: "vllm-sr/4.0"},
		"local":  {Provider: config.ModelRuntimeProvider},
	}
	remote := &answererDecider{cards: map[string]modelservice.ModelCard{"remote": {ID: "remote-card"}}}
	local := &answererDecider{}
	answerer := routerMatchAnswerer{&OpenAIRouter{
		Config:           cfg,
		systemOneDecider: remote,
		decisionDecider:  local,
		decisionCards:    map[string]modelservice.ModelCard{"local": {ID: "local-card"}},
	}}

	for _, deployment := range []string{"remote", "local"} {
		if _, err := answerer.Decide(context.Background(), deployment, modelservice.Request{}); err != nil {
			t.Fatalf("%s: %v", deployment, err)
		}
	}
	if len(remote.asked) != 1 || remote.asked[0] != "remote" || len(local.asked) != 1 || local.asked[0] != "local" {
		t.Fatalf("systemone asked %v, model runtime asked %v", remote.asked, local.asked)
	}
	for deployment, want := range map[string]string{"remote": "remote-card", "local": "local-card"} {
		if card, ok := answerer.Card(deployment); !ok || card.ID != want {
			t.Fatalf("card of %s = %q, %v; want %q", deployment, card.ID, ok, want)
		}
	}
	if _, ok := answerer.Card("unprepared"); ok {
		t.Fatal("a deployment the generation did not prepare has no card")
	}

	answerer.r.systemOneDecider = nil
	if _, err := answerer.Decide(context.Background(), "remote", modelservice.Request{}); !errors.Is(err, modelservice.ErrUnavailable) {
		t.Fatalf("systemone without a client = %v, want unavailable", err)
	}
	if _, ok := answerer.Card("remote"); ok {
		t.Fatal("systemone without a client has no card")
	}
}

func TestMatchJudgesAreTheReachableDecisionsJudges(t *testing.T) {
	selecting := func(deployment string) config.Decision {
		payload, err := config.NewStructuredPayload(map[string]any{"enabled": true, "select": map[string]any{
			"mode":  config.ToolsSelectModeFilter,
			"match": map[string]any{"judge": map[string]any{"preset": "tool_rank", "deployment": deployment}},
		}})
		if err != nil {
			t.Fatal(err)
		}
		return config.Decision{Name: deployment, Plugins: []config.DecisionPlugin{{Type: config.DecisionPluginTools, Configuration: payload}}}
	}
	cfg := &config.RouterConfig{}
	cfg.Recipes = []config.RoutingRecipe{
		{Name: "served", Profile: config.RoutingProfile{Decisions: []config.Decision{selecting("lite"), {Name: "plain"}}}},
		{Name: "unserved", Profile: config.RoutingProfile{Decisions: []config.Decision{selecting("unused")}}},
	}
	cfg.Entrypoints = []config.EntrypointMapping{{Recipe: "served", ModelNames: []string{"public"}}}

	judges := matchJudges(cfg)
	if len(judges) != 1 || judges[0].Deployment != "lite" || judges[0].Preset != "tool_rank" {
		t.Fatalf("judges = %+v, want the served decision's tool_rank judge on lite", judges)
	}
}
