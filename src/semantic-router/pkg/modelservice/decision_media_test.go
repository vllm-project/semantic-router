package modelservice

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
)

func TestDecisionMediaWireAndIdentity(t *testing.T) {
	base := Request{State: "describe this", Questions: []Question{choiceQuestion("q")}}
	variants := []Request{base, base, base, base, base}
	variants[1].Images = []string{"a", "b"}
	variants[2].Images = []string{"b", "a"}
	variants[3].Videos = []string{"a", "b"}
	variants[4].Images = []string{"a", "c"}
	keys := map[cacheKey]bool{}
	states := map[string]bool{}
	for _, r := range variants {
		k, s := decideKey(r), stateText(r)
		if keys[k] || states[s] {
			t.Fatal("different media share an identity")
		}
		keys[k], states[s] = true, true
		wire, err := encodeDecisionRequest(context.Background(), r)
		if err != nil {
			t.Fatal(err)
		}
		encoded, _ := json.Marshal(wire)
		var got struct{ Images, Videos []string }
		if err := json.Unmarshal(encoded, &got); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got.Images, r.Images) || !reflect.DeepEqual(got.Videos, r.Videos) {
			t.Fatal("lost media")
		}
	}
	if stateText(base) != "sdescribe this" {
		t.Fatal("text-only identity changed")
	}
}

func TestDecisionMediaFusedStates(t *testing.T) {
	client := &Client{}
	client.bundleTasks.Store(64)
	var calls []*bundleCall
	for i, media := range []string{"a", "b", "a"} {
		id := []string{"q1", "q2", "q3"}[i]
		r := Request{State: "same", Images: []string{media}, Videos: []string{"clip"}, Questions: []Question{choiceQuestion(id)}}
		wire, err := encodeDecisionRequest(context.Background(), r)
		if err != nil {
			t.Fatal(err)
		}
		calls = append(calls, &bundleCall{ctx: context.Background(), task: api.BundleTask{Id: id, Decisions: &wire}, decision: &decisionCall{request: r}})
	}
	tasks := fuseDecisions(client, calls)
	if len(tasks) != 1 {
		t.Fatalf("tasks=%d", len(tasks))
	}
	b := tasks[0].task.Decisions
	if len(b.Questions) != 2 || b.Images == nil || (*b.Images)[0] != "a" {
		t.Fatal("same media did not share a state")
	}
	if b.States == nil || len(*b.States) != 1 {
		t.Fatal("different media merged")
	}
	for _, s := range *b.States {
		if s.Images == nil || (*s.Images)[0] != "b" || s.Videos == nil || (*s.Videos)[0] != "clip" {
			t.Fatal("additional state lost media")
		}
	}
}

func TestDecisionMediaContextIsolation(t *testing.T) {
	media := map[string]DecisionMedia{"vision": {Images: []string{"a"}}}
	ctx := WithDecisionMedia(context.Background(), media)
	media["vision"].Images[0] = "changed"
	if r := attachDecisionMedia(ctx, "vision", Request{}); r.Images[0] != "a" {
		t.Fatal("caller mutated scoped media")
	}
	if r := attachDecisionMedia(ctx, "text", Request{}); len(r.Images) != 0 {
		t.Fatal("media crossed deployments")
	}
}

func TestDecisionMediaCardValidation(t *testing.T) {
	d := config.ModelDeployment{Provider: config.ModelRuntimeProvider, Input: config.ModelInputBudget{Media: &config.DecisionMediaConfig{Images: &config.DecisionMediaLimit{}}}}
	c := ModelCard{Surfaces: []string{"decisions"}, Modalities: []string{"text", "image"}, ImageMaxBytes: 8000000}
	if err := ValidateDecisionMedia(d, c); err != nil {
		t.Fatal(err)
	}
	d.Input.Media.Images.MaxBytes = 8000001
	if ValidateDecisionMedia(d, c) == nil {
		t.Fatal("limit above card")
	}
	d.Input.Media.Images.MaxBytes = 0
	c.Modalities = []string{"text"}
	if ValidateDecisionMedia(d, c) == nil {
		t.Fatal("text card accepted")
	}
	d.Input.Media = nil
	if err := ValidateDecisionMedia(d, c); err != nil {
		t.Fatal(err)
	}
}

func TestDecisionMediaFusionRespectsSerializedLimit(t *testing.T) {
	client := &Client{}
	client.bundleTasks.Store(64)
	var calls []*bundleCall
	for _, id := range []string{"a", "b"} {
		r := Request{State: id, Images: []string{strings.Repeat(id, 500)}, Questions: []Question{choiceQuestion("q")}}
		wire, err := encodeDecisionRequest(context.Background(), r)
		if err != nil {
			t.Fatal(err)
		}
		calls = append(calls, &bundleCall{ctx: context.Background(), task: api.BundleTask{Id: id, Decisions: &wire}, decision: &decisionCall{request: r}})
	}
	one := fuseDecisions(client, calls[:1])
	size, _ := json.Marshal(api.BundleRequest{Tasks: []api.BundleTask{one[0].task}})
	client.requestBytes.Store(int64(len(size) + 10))
	tasks := fuseDecisions(client, calls)
	if len(tasks) != 2 {
		t.Fatal("oversized multi-state task was fused")
	}
	for _, task := range tasks {
		if err := client.checkMediaBody(api.BundleRequest{Tasks: []api.BundleTask{task.task}}, true); err != nil {
			t.Fatal(err)
		}
	}
}
