package modelservice

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
)

// Questions to one model about one state travel in one call: a flush fuses
// the stage's decisions calls that name the same model, state and options
// into one task with every caller's questions, and hands each caller the
// answers to its own. Calls whose question IDs could share an answer key (an
// equal ID, or an ID under a Set question's "<id>." labels) stay apart.
//
// The result cache applies to the call the runtime answers: a fused call is
// looked up and stored under the key of all its questions, so an answer a
// model gave alongside other questions never serves a call that asked alone.

// fusedDecision is a decisions task's questions in ID order and its cache key.
type fusedDecision struct {
	request Request
	cache   *resultCache
	key     cacheKey
}

// decisionGroup is the fusion key of a decisions call: its model, state,
// options without the deadline and the cache it reads.
func decisionGroup(call *bundleCall) string {
	request := *call.task.Decisions
	shape := struct {
		Model   *string             `json:"model,omitempty"`
		State   interface{}         `json:"state"`
		Options *api.RequestOptions `json:"options,omitempty"`
	}{Model: request.Model, State: request.State}
	if request.Options != nil {
		options := *request.Options
		options.DeadlineMs = nil
		shape.Options = &options
	}
	key, err := json.Marshal(shape)
	if err != nil {
		return ""
	}
	return fmt.Sprintf("%s|%p", key, call.decision.cache)
}

// questionIDs are a decisions call's question IDs.
func questionIDs(call *bundleCall) []string {
	ids := make([]string, len(call.decision.request.Questions))
	for index, question := range call.decision.request.Questions {
		ids[index] = question.ID
	}
	return ids
}

// disjointAnswers reports whether no question ID of one set equals one of the
// other or names a label answer ("<id>.<label>") of it.
func disjointAnswers(ids []string, taken map[string]struct{}) bool {
	for _, id := range ids {
		if _, exists := taken[id]; exists {
			return false
		}
		for other := range taken {
			if strings.HasPrefix(id, other+".") || strings.HasPrefix(other, id+".") {
				return false
			}
		}
	}
	return true
}

// fuseDecision adds a decisions call to the first open task of its group it
// shares no answer key with, or opens a task for it; it returns the new task
// or nil.
func fuseDecision(open map[string][]*bundleTask, taken map[*bundleTask]map[string]struct{}, call *bundleCall) *bundleTask {
	ids := questionIDs(call)
	group := decisionGroup(call)
	if group != "" {
		for _, task := range open[group] {
			if disjointAnswers(ids, taken[task]) {
				task.calls = append(task.calls, call)
				for _, id := range ids {
					taken[task][id] = struct{}{}
				}
				return nil
			}
		}
	}
	task := &bundleTask{task: call.task, calls: []*bundleCall{call}, decision: &fusedDecision{}}
	taken[task] = make(map[string]struct{}, len(ids))
	for _, id := range ids {
		taken[task][id] = struct{}{}
	}
	if group != "" {
		open[group] = append(open[group], task)
	}
	return task
}

// prepareDecision builds a decisions task's request: every caller's
// questions in ID order and, for a fused task, the first call's body with all
// of them and the latest deadline (none if any call has none).
func (t *bundleTask) prepareDecision() {
	first := t.calls[0]
	request := first.decision.request
	request.Questions = nil
	for _, call := range t.calls {
		request.Questions = append(request.Questions, call.decision.request.Questions...)
	}
	sort.SliceStable(request.Questions, func(i, j int) bool { return request.Questions[i].ID < request.Questions[j].ID })
	t.decision.request = request
	t.decision.cache = first.decision.cache
	if t.decision.cache != nil {
		t.decision.key = decideKey(request)
	}
	if len(t.calls) == 1 {
		return
	}
	body := *first.task.Decisions
	body.Questions = make(map[string]api.Question, len(request.Questions))
	for _, call := range t.calls {
		for id, question := range call.task.Decisions.Questions {
			body.Questions[id] = question
		}
	}
	var latest *float64
	for index, call := range t.calls {
		options := call.task.Decisions.Options
		if options == nil || options.DeadlineMs == nil {
			latest = nil
			break
		}
		if index == 0 || *options.DeadlineMs > *latest {
			latest = options.DeadlineMs
		}
	}
	if body.Options != nil {
		options := *body.Options
		options.DeadlineMs = latest
		body.Options = &options
	}
	t.task = api.BundleTask{Id: first.task.Id, Decisions: &body}
}

// answerCached answers the decisions tasks the cache holds and returns the
// tasks still to send.
func answerCached(tasks []*bundleTask) []*bundleTask {
	unsent := tasks[:0]
	for _, task := range tasks {
		if task.decision != nil && task.decision.cache != nil {
			if cached, ok := task.decision.cache.get(task.decision.key); ok {
				for _, call := range task.calls {
					cacheTotal.WithLabelValues(call.decision.deployment, "hit").Inc()
				}
				task.deliver(cached.(Response), nil)
				continue
			}
		}
		unsent = append(unsent, task)
	}
	return unsent
}

// answerDecision decodes a decisions task's result once, stores a complete
// answer in the cache and hands every call its own answers.
func (t *bundleTask) answerDecision(result api.BundleResult, err error) {
	if err == nil {
		err = statusError(result.Status, result.Error)
	}
	if err == nil && result.Decisions == nil {
		err = fmt.Errorf("%w: missing decision response body", ErrFailed)
	}
	var response Response
	if err == nil {
		response = decodeResponse(*result.Decisions, t.decision.request.Questions)
		if cache := t.decision.cache; cache != nil {
			for _, call := range t.calls {
				cacheTotal.WithLabelValues(call.decision.deployment, "miss").Inc()
			}
			if complete(response) {
				cache.put(t.decision.key, response)
			}
		}
	}
	t.deliver(response, err)
}

// deliver hands every call of a decisions task the answers to its questions.
func (t *bundleTask) deliver(response Response, err error) {
	for _, call := range t.calls {
		call.err = err
		if err == nil {
			call.decision.response = response.part(call.decision.request.Questions)
		}
		close(call.done)
	}
}

// part is the response restricted to the given questions.
func (r Response) part(questions []Question) Response {
	answers := make(map[string]Answer, len(questions))
	for _, question := range questions {
		if answer, ok := r.Answers[question.ID]; ok {
			answers[question.ID] = answer
		}
	}
	return Response{Model: r.Model, Answers: answers, InputTokens: r.InputTokens}
}
