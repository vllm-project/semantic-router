package modelservice

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/runtimetest"
)

// windowText is the text of a guard scan's window of rank, words words long.
func windowText(rank, words int) string {
	return strings.Repeat(fmt.Sprintf("window%d ", rank), words)
}

// windowRank is the rank windowText wrote into a state.
func windowRank(state any) int {
	text, _ := state.(string)
	var rank int
	_, _ = fmt.Sscanf(text, "window%d", &rank)
	return rank
}

// A stage's guard windows each travel in a call of their own, with the other
// guards' questions about the same window, at most their parallel bound in
// flight and the lowest rank first; the routing question's call goes at once
// and does not wait for them, and each window answers as soon as the engine
// has read it.
func TestGuardWindowsTravelInCallsOfTheirOwn(t *testing.T) {
	runtime := runtimetest.New(runtimetest.Model{ID: "d3"})
	runtime.SetWordTime(time.Millisecond)
	lease := attachedLease(t, map[*runtimetest.Runtime][]string{runtime: {"d3"}})
	const windows, parallel = 6, 2
	var mu sync.Mutex
	answered := map[string]time.Duration{}
	started := time.Now()
	ask := func(key string, request Request) asker {
		return asker{deployments: []string{"d3"}, ask: func(ctx context.Context) error {
			response, err := lease.Decide(ctx, "d3", request)
			if err == nil && len(response.Answers) != 1 {
				err = fmt.Errorf("%s: answers %+v", key, response.Answers)
			}
			mu.Lock()
			answered[key] = time.Since(started)
			mu.Unlock()
			return err
		}}
	}
	askers := []asker{ask("routing", Request{State: "route this request", Questions: []Question{choiceQuestion("systemone:task")}})}
	for rank := windows; rank >= 1; rank-- {
		for _, guard := range []string{"jailbreak", "safety"} {
			request := Request{State: windowText(rank, 30), Window: rank, Parallel: parallel, Questions: []Question{choiceQuestion("preset:" + guard)}}
			askers = append(askers, ask(fmt.Sprintf("%s/%d", guard, rank), request))
		}
	}
	if _, errs := runStage(DefaultBundleWindow, askers...); errors.Join(errs...) != nil {
		t.Fatal(errors.Join(errs...))
	}
	if calls, tasks := runtime.Bundles(); calls != 1+windows || tasks != 1+windows {
		t.Fatalf("%d calls with %d tasks; want the routing call and one per window", calls, tasks)
	}
	if most := runtime.MaxConcurrentBundles(); most > parallel+1 {
		t.Fatalf("%d calls at once; at most %d windows and the routing call", most, parallel)
	}
	var ranks []int
	for _, decision := range runtime.Decisions() {
		if decision.States != nil {
			t.Fatal("a window travels alone, not in a batch")
		}
		if rank := windowRank(decision.State); rank > 0 {
			ranks = append(ranks, rank)
			if len(decision.Questions) != 2 {
				t.Fatalf("window %d asks %d questions; both guards share its call", rank, len(decision.Questions))
			}
		}
	}
	for position, rank := range ranks {
		if rank > position+parallel {
			t.Fatalf("windows arrived as %v: the lowest ranks go first", ranks)
		}
	}
	if answered["routing"] >= answered[fmt.Sprintf("jailbreak/%d", windows)] {
		t.Fatalf("the routing call waited for every window: %v", answered)
	}
	if answered["jailbreak/1"] >= answered[fmt.Sprintf("jailbreak/%d", windows)] {
		t.Fatalf("the first window answered with the last: %v", answered)
	}
}

// At the stage's deadline the windows read so far are answered, the one in
// flight fails at the deadline, and a window no caller still waits for is
// never sent.
func TestADeadlineLeavesTheWindowsReadAnswered(t *testing.T) {
	runtime := runtimetest.New(runtimetest.Model{ID: "d3"})
	runtime.SetWordTime(5 * time.Millisecond)
	lease := attachedLease(t, map[*runtimetest.Runtime][]string{runtime: {"d3"}})
	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel()
	var askers []asker
	for rank := 1; rank <= 4; rank++ {
		request := Request{State: windowText(rank, 40), Window: rank, Parallel: 1, Questions: []Question{choiceQuestion("preset:jailbreak")}}
		askers = append(askers, asker{deployments: []string{"d3"}, ask: func(ctx context.Context) error {
			_, err := lease.Decide(ctx, "d3", request)
			return err
		}})
	}
	_, errs := runStageIn(ctx, DefaultBundleWindow, askers...)
	if errs[0] != nil {
		t.Fatalf("the first window, read before the deadline: %v", errs[0])
	}
	for rank, err := range errs[1:] {
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("window %d past the deadline: %v", rank+2, err)
		}
	}
	time.Sleep(100 * time.Millisecond)
	sent := []int{}
	for _, decision := range runtime.Decisions() {
		sent = append(sent, windowRank(decision.State))
	}
	if !slices.Equal(sent, []int{1, 2}) {
		t.Fatalf("windows sent %v; want the first, and the second cut at the deadline", sent)
	}
}

// A caller that gives up on its windows, its context cancelled rather than
// past its deadline, as a stage whose window found an attack does, stops
// waiting for the one in flight at once, and the one queued behind it is
// never sent.
func TestACancelledCallerStopsWaitingForItsWindows(t *testing.T) {
	runtime := runtimetest.New(runtimetest.Model{ID: "d3"})
	runtime.SetWordTime(20 * time.Millisecond)
	lease := attachedLease(t, map[*runtimetest.Runtime][]string{runtime: {"d3"}})
	stop, cancel := context.WithCancel(context.Background())
	defer cancel()
	var askers []asker
	for rank := 1; rank <= 2; rank++ {
		request := Request{State: windowText(rank, 30), Window: rank, Parallel: 1, Questions: []Question{choiceQuestion("preset:jailbreak")}}
		askers = append(askers, asker{deployments: []string{"d3"}, ask: func(ctx context.Context) error {
			ctx, release := context.WithCancel(ctx)
			defer release()
			context.AfterFunc(stop, release)
			_, err := lease.Decide(ctx, "d3", request)
			return err
		}})
	}
	time.AfterFunc(100*time.Millisecond, cancel)
	started := time.Now()
	_, errs := runStage(DefaultBundleWindow, askers...)
	if elapsed := time.Since(started); elapsed > 400*time.Millisecond {
		t.Fatalf("the stage gave up after 100 ms but waited %v for the window in flight (600 ms)", elapsed)
	}
	for rank, err := range errs {
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("window %d: %v, want cancelled", rank+1, err)
		}
	}
	time.Sleep(700 * time.Millisecond)
	for _, decision := range runtime.Decisions() {
		if rank := windowRank(decision.State); rank != 1 {
			t.Fatalf("window %d was sent after its caller gave up", rank)
		}
	}
}
