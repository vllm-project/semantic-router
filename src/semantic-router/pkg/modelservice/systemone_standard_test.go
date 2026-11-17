package modelservice

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// fakeStandardAPI is a System One API of schema systemone/0.2 or vllm-sr/5.0
// that answers every choice question "a" after a short wait. A request whose
// state is refuse is refused 422, one whose state is slow answers after its
// caller has stopped waiting, and one whose state is busy is refused 429 with
// Retry-After: 1 the first time. It keeps each call's path and body and the
// most calls it had in flight.
type fakeStandardAPI struct {
	refuse, slow, busy string

	mu             sync.Mutex
	paths, bodies  []string
	inFlight, peak int
	busied         bool
}

func (f *fakeStandardAPI) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	data, _ := io.ReadAll(r.Body)
	f.mu.Lock()
	f.paths, f.bodies = append(f.paths, r.URL.Path), append(f.bodies, string(data))
	f.inFlight++
	f.peak = max(f.peak, f.inFlight)
	f.mu.Unlock()
	defer func() {
		f.mu.Lock()
		f.inFlight--
		f.mu.Unlock()
	}()
	time.Sleep(50 * time.Millisecond)
	w.Header().Set("Content-Type", "application/json")
	type request struct {
		State     json.RawMessage            `json:"state"`
		Questions map[string]json.RawMessage `json:"questions"`
	}
	answers := func(questions map[string]json.RawMessage) map[string]any {
		answered := make(map[string]any, len(questions))
		for id := range questions {
			answered[id] = map[string]any{"type": "choice", "choice": "a", "probabilities": map[string]float64{"a": 0.7, "b": 0.3}, "confidence": 0.7}
		}
		return answered
	}
	if strings.HasSuffix(r.URL.Path, "/batch") {
		var body struct {
			Requests map[string]request `json:"requests"`
		}
		_ = json.Unmarshal(data, &body)
		responses := make(map[string]any, len(body.Requests))
		for name, entry := range body.Requests {
			responses[name] = map[string]any{"answers": answers(entry.Questions), "usage": map[string]int{"input_tokens": 3, "output_tokens": 0}}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"model": "vllm-sr/d3-ultra", "responses": responses, "usage": map[string]int{"input_tokens": 3 * len(responses), "output_tokens": 0}})
		return
	}
	var body request
	_ = json.Unmarshal(data, &body)
	var state string
	_ = json.Unmarshal(body.State, &state)
	switch state {
	case "":
	case f.refuse:
		w.WriteHeader(http.StatusUnprocessableEntity)
		fmt.Fprint(w, `{"detail":[{"loc":["body","state"],"msg":"the state is too long","type":"value_error"}]}`)
		return
	case f.slow:
		select {
		case <-r.Context().Done():
		case <-time.After(2 * time.Second):
		}
	case f.busy:
		f.mu.Lock()
		first := !f.busied
		f.busied = true
		f.mu.Unlock()
		if first {
			w.Header().Set("Retry-After", "1")
			w.WriteHeader(http.StatusTooManyRequests)
			fmt.Fprint(w, `{"detail":"too many requests"}`)
			return
		}
	}
	_ = json.NewEncoder(w).Encode(map[string]any{"model": "vllm-sr/d3-ultra", "answers": answers(body.Questions), "usage": map[string]int{"input_tokens": 3, "output_tokens": 0}})
}

func (f *fakeStandardAPI) calls() ([]string, []string, int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.Clone(f.paths), slices.Clone(f.bodies), f.peak
}

// A systemone/0.2 stage asks each of its states in a call of its own, at most
// max_concurrency of them in flight; a call refused, or one its deadline cut
// short, leaves only its own state's questions unknown.
func TestAStandardStageAsksEachStateInACallOfItsOwn(t *testing.T) {
	limit := 2
	api := &fakeStandardAPI{refuse: "refused", slow: "slow"}
	remote := fakeRemote(t, "standard", api, config.ModelDeployment{Schema: "systemone/0.2", MaxConcurrency: &limit})
	states := []string{"first", "refused", "slow"}
	responses := make([]Response, len(states))
	askers := make([]asker, len(states))
	for index, state := range states {
		askers[index] = asker{deployments: []string{"standard"}, ask: func(ctx context.Context) error {
			ctx, cancel := context.WithTimeout(ctx, 400*time.Millisecond)
			defer cancel()
			var err error
			responses[index], err = remote.Decide(ctx, "standard", Request{State: state, Questions: []Question{choiceQuestion("team")}})
			return err
		}}
	}
	_, errs := runStage(DefaultBundleWindow, askers...)
	if errs[0] != nil || responses[0].Answers["team"].Choice != "a" {
		t.Fatalf("the first state: %+v, %v", responses[0], errs[0])
	}
	if !errors.Is(errs[1], ErrRejected) {
		t.Fatalf("the refused state: %v, want rejected", errs[1])
	}
	if !errors.Is(errs[2], context.DeadlineExceeded) {
		t.Fatalf("the slow state: %v, want its deadline", errs[2])
	}
	paths, bodies, peak := api.calls()
	if len(bodies) != 3 || peak != 2 || slices.ContainsFunc(paths, func(path string) bool { return path != "/v1/systemone" }) {
		t.Fatalf("calls %v, peak %d in flight; want 3 calls to /v1/systemone, 2 in flight", paths, peak)
	}
	for _, body := range bodies {
		if strings.Contains(body, `"states"`) || strings.Contains(body, `"options"`) || !strings.Contains(body, `"criteria":{"a":null,"b":null}`) {
			t.Fatalf("a systemone/0.2 call carries one state and only standard fields: %s", body)
		}
	}
}

// A systemone/0.2 deployment that is overloaded is asked once more after its
// Retry-After, within the caller's deadline.
func TestAStandardDeploymentIsAskedAgainAfterRetryAfter(t *testing.T) {
	api := &fakeStandardAPI{busy: "busy"}
	remote := fakeRemote(t, "standard-busy", api, config.ModelDeployment{Schema: "systemone/0.2"})
	started := time.Now()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	response, err := remote.Decide(ctx, "standard-busy", Request{State: "busy", Questions: []Question{choiceQuestion("team")}})
	if err != nil || response.Answers["team"].Choice != "a" {
		t.Fatalf("%+v, %v", response, err)
	}
	if _, bodies, _ := api.calls(); len(bodies) != 2 || time.Since(started) < time.Second {
		t.Fatalf("%d calls in %s, want 2 a second apart", len(bodies), time.Since(started))
	}
}

// A state_text deployment gets an object state as its canonical text, in the
// state's order and without escaping what is not ASCII.
func TestAStateTextDeploymentGetsAnObjectStateAsItsText(t *testing.T) {
	api := &fakeStandardAPI{}
	remote := fakeRemote(t, "standard-text", api, config.ModelDeployment{Schema: "systemone/0.2", Deviations: []string{"state_text"}})
	parts := map[string]string{"request": "Book a flight to München", "context": "Fares rose 5 percent"}
	if _, err := remote.Decide(context.Background(), "standard-text", Request{Parts: parts, Questions: []Question{choiceQuestion("team")}}); err != nil {
		t.Fatal(err)
	}
	_, bodies, _ := api.calls()
	want := `"state":"{\"context\": \"Fares rose 5 percent\", \"request\": \"Book a flight to München\"}"`
	if len(bodies) != 1 || !strings.Contains(bodies[0], want) {
		t.Fatalf("sent %q, want the state %s", bodies, want)
	}
}

// A vllm-sr/5.0 stage asks all its states in one batch call, each object
// state with the roles its keys imply written out.
func TestAFiveStageAsksItsStatesInOneBatchWithTheirRoles(t *testing.T) {
	api := &fakeStandardAPI{}
	remote := fakeRemote(t, "five", api, config.ModelDeployment{Schema: "vllm-sr/5.0"})
	requests := []Request{
		{Parts: map[string]string{"request": "Book a flight", "context": "Fares rose"}, Questions: []Question{choiceQuestion("team")}},
		{State: "a tool result", Questions: []Question{choiceQuestion("team")}},
	}
	responses := make([]Response, len(requests))
	askers := make([]asker, len(requests))
	for index, request := range requests {
		askers[index] = asker{deployments: []string{"five"}, ask: func(ctx context.Context) error {
			var err error
			responses[index], err = remote.Decide(ctx, "five", request)
			return err
		}}
	}
	if _, errs := runStage(DefaultBundleWindow, askers...); errors.Join(errs...) != nil {
		t.Fatal(errors.Join(errs...))
	}
	for index, response := range responses {
		if response.Answers["team"].Choice != "a" {
			t.Fatalf("state %d: %+v", index, response)
		}
	}
	paths, bodies, _ := api.calls()
	if len(bodies) != 1 || paths[0] != "/v1/systemone/batch" {
		t.Fatalf("calls %v, want one to /v1/systemone/batch", paths)
	}
	var batch struct {
		Requests map[string]map[string]json.RawMessage `json:"requests"`
	}
	if err := json.Unmarshal([]byte(bodies[0]), &batch); err != nil || len(batch.Requests) != 2 {
		t.Fatalf("the batch %s: %v", bodies[0], err)
	}
	for name, entry := range batch.Requests {
		roles, named := entry["roles"]
		switch string(entry["state"]) {
		case `{"context":"Fares rose","request":"Book a flight"}`:
			if string(roles) != `{"context":"context","request":"user"}` {
				t.Fatalf("entry %s has roles %s, want the ones its keys imply", name, roles)
			}
		case `"a tool result"`:
			if named {
				t.Fatalf("a string state has roles %s", roles)
			}
		default:
			t.Fatalf("entry %s: %s", name, entry["state"])
		}
	}
}
