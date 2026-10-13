package modelservice

import (
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
)

// The questions a stage asks one served model travel in one call. A send
// fuses the parked decisions calls to one model with the same options (but
// the deadline) and cache into one task and hands each caller the answers to
// its own questions. Calls about one state share it, and the model reads
// their questions together, as one request with all of them. Calls about
// other states go in the same call as further states (DecisionRequest.states),
// each read exactly as a request of its own, when the runtime's contract has
// them (statesMinor); an older runtime gets one task per state. Calls about
// one state whose question IDs could share an answer key (an equal ID, or an
// ID under a Set question's "<id>." labels) stay apart.
//
// The result cache applies to the call the runtime answers: a fused call is
// looked up and stored under the key of all its states and questions, so an
// answer a model gave alongside other questions never serves a call that
// asked alone.

// statesMinor is the minor version of the runtime contract from which a
// decisions call carries further states.
const statesMinor = 2

// fusedDecision is a decisions task's states, the request's own state first,
// and its cache key.
type fusedDecision struct {
	states []*fusedState
	cache  *resultCache
	key    cacheKey
}

// fusedState is one state of a fused decisions task: the calls about it, its
// request (the state and every question about it, in ID order), its name in
// the request's states ("" for the request's own state) and its fusion key.
type fusedState struct {
	name    string
	text    string
	calls   []*bundleCall
	taken   map[string]struct{}
	request Request
}

// decisionGroup is the fusion key of a decisions call: its model, options
// without the deadline and the cache it reads.
func decisionGroup(call *bundleCall) string {
	request := *call.task.Decisions
	shape := struct {
		Model   *string             `json:"model,omitempty"`
		Options *api.RequestOptions `json:"options,omitempty"`
	}{Model: request.Model}
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

// stateText is the fusion key of a call's state: its text, or its named parts.
func stateText(request Request) string {
	if request.Parts == nil {
		return "s" + request.State
	}
	names := make([]string, 0, len(request.Parts))
	for name := range request.Parts {
		names = append(names, name)
	}
	sort.Strings(names)
	var key strings.Builder
	key.WriteByte('p')
	for _, name := range names {
		for _, field := range []string{name, request.Parts[name]} {
			key.WriteString(strconv.Itoa(len(field)))
			key.WriteByte(':')
			key.WriteString(field)
		}
	}
	return key.String()
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

// fuseDecisions turns one client's parked decisions calls into tasks. The
// calls are placed in a fixed order, so the same calls make the same tasks
// whatever order they parked in.
func fuseDecisions(client *Client, calls []*bundleCall) []*bundleTask {
	type placed struct {
		call  *bundleCall
		group string
		text  string
		first string
	}
	ordered := make([]placed, len(calls))
	for index, call := range calls {
		ids := questionIDs(call)
		sort.Strings(ids)
		first := ""
		if len(ids) > 0 {
			first = ids[0]
		}
		ordered[index] = placed{call: call, group: decisionGroup(call), text: stateText(call.decision.request), first: first}
	}
	sort.SliceStable(ordered, func(i, j int) bool {
		a, b := ordered[i], ordered[j]
		if a.group != b.group {
			return a.group < b.group
		}
		if a.text != b.text {
			return a.text < b.text
		}
		return a.first < b.first
	})
	multiple := client.takesStates()
	limit := max(1, int(client.bundleTasks.Load()))
	open := make(map[string][]*bundleTask)
	var tasks []*bundleTask
	for _, entry := range ordered {
		var target *bundleTask
		if entry.group != "" {
			for _, task := range open[entry.group] {
				if task.decision.fits(entry.call, entry.text, multiple, limit) {
					target = task
					break
				}
			}
		}
		if target == nil {
			target = &bundleTask{decision: &fusedDecision{cache: entry.call.decision.cache}}
			tasks = append(tasks, target)
			if entry.group != "" {
				open[entry.group] = append(open[entry.group], target)
			}
		}
		target.decision.add(entry.call, entry.text)
		target.calls = append(target.calls, entry.call)
	}
	for _, task := range tasks {
		task.prepareDecision()
	}
	return tasks
}

// fits reports whether a call can join the task: about one of its states
// with answer keys none of that state's questions has, or about another
// state while the runtime takes further states and the task has fewer than
// limit of them: as many states as the runtime takes tasks in one bundle,
// each of which it runs as a job of its own.
func (d *fusedDecision) fits(call *bundleCall, text string, multiple bool, limit int) bool {
	for _, state := range d.states {
		if state.text == text {
			return disjointAnswers(questionIDs(call), state.taken)
		}
	}
	return multiple && len(d.states) < limit
}

func (d *fusedDecision) add(call *bundleCall, text string) {
	var state *fusedState
	for _, existing := range d.states {
		if existing.text == text {
			state = existing
		}
	}
	if state == nil {
		state = &fusedState{text: text, taken: make(map[string]struct{}), request: call.decision.request}
		state.request.Questions = nil
		d.states = append(d.states, state)
	}
	state.calls = append(state.calls, call)
	for _, question := range call.decision.request.Questions {
		state.taken[question.ID] = struct{}{}
		state.request.Questions = append(state.request.Questions, question)
	}
}

// prepareDecision builds a decisions task's request. The state with the most
// questions (the first in order on a tie) is the request's own; the others
// are its states "1", "2", ... in order. The request carries the latest
// deadline of its calls (none if a call has none).
func (t *bundleTask) prepareDecision() {
	d := t.decision
	own := 0
	for index, state := range d.states {
		sort.SliceStable(state.request.Questions, func(i, j int) bool { return state.request.Questions[i].ID < state.request.Questions[j].ID })
		if len(state.request.Questions) > len(d.states[own].request.Questions) {
			own = index
		}
	}
	d.states = append(append([]*fusedState{d.states[own]}, d.states[:own]...), d.states[own+1:]...)
	first := d.states[0].calls[0]
	body := *first.task.Decisions
	body.Questions = d.states[0].questions()
	if len(d.states) > 1 {
		states := make(map[string]api.DecisionState, len(d.states)-1)
		for index, state := range d.states[1:] {
			state.name = strconv.Itoa(index + 1)
			states[state.name] = api.DecisionState{State: state.calls[0].task.Decisions.State, Questions: state.questions()}
		}
		body.States = &states
	}
	if body.Options != nil {
		options := *body.Options
		options.DeadlineMs = t.latestDeadline()
		body.Options = &options
	}
	t.task = api.BundleTask{Id: first.task.Id, Decisions: &body}
	if d.cache != nil {
		d.key = d.cacheKey()
	}
}

// questions is the encoded questions of every call about the state.
func (s *fusedState) questions() map[string]api.Question {
	questions := make(map[string]api.Question, len(s.taken))
	for _, call := range s.calls {
		for id, question := range call.task.Decisions.Questions {
			questions[id] = question
		}
	}
	return questions
}

// latestDeadline is the latest deadline the task's calls send, nil when one
// sends none.
func (t *bundleTask) latestDeadline() *float64 {
	var latest *float64
	for _, call := range t.calls {
		options := call.task.Decisions.Options
		if options == nil || options.DeadlineMs == nil {
			return nil
		}
		if latest == nil || *options.DeadlineMs > *latest {
			latest = options.DeadlineMs
		}
	}
	return latest
}

// cacheKey keys a task by its own state's request, and each further state's
// name and request after it.
func (d *fusedDecision) cacheKey() cacheKey {
	if len(d.states) == 1 {
		return decideKey(d.states[0].request)
	}
	requests := make([]Request, len(d.states))
	names := make([]string, len(d.states))
	for index, state := range d.states {
		requests[index], names[index] = state.request, state.name
	}
	return decideStatesKey(names, requests)
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
				task.deliver(cachedResponses(cached), nil)
				continue
			}
		}
		unsent = append(unsent, task)
	}
	return unsent
}

// cachedResponses reads a cached decisions result: one state's Response, or
// one per state.
func cachedResponses(value any) []Response {
	if responses, ok := value.([]Response); ok {
		return responses
	}
	return []Response{value.(Response)}
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
	var responses []Response
	if err == nil {
		responses, err = t.decision.decode(*result.Decisions)
	}
	if err == nil {
		if cache := t.decision.cache; cache != nil {
			for _, call := range t.calls {
				cacheTotal.WithLabelValues(call.decision.deployment, "miss").Inc()
			}
			if completeAll(responses) {
				if len(responses) == 1 {
					cache.put(t.decision.key, responses[0])
				} else {
					cache.put(t.decision.key, responses)
				}
			}
		}
	}
	t.deliver(responses, err)
}

// decode reads every state's answers: the request's own from the response,
// and each further state's from its entry.
func (d *fusedDecision) decode(body api.DecisionResponse) ([]Response, error) {
	responses := make([]Response, len(d.states))
	responses[0] = decodeResponse(body, d.states[0].request.Questions)
	for index, state := range d.states[1:] {
		var entry api.DecisionStateResponse
		found := false
		if body.States != nil {
			entry, found = (*body.States)[state.name]
		}
		if !found {
			return nil, fmt.Errorf("%w: the decision response has no answers about state %q", ErrFailed, state.name)
		}
		answered := api.DecisionResponse{
			Model: entry.Model, Answers: entry.Answers, Sets: entry.Sets, Spans: entry.Spans,
			Thresholds: entry.Thresholds, SpanHeads: entry.SpanHeads, Usage: entry.Usage,
		}
		responses[index+1] = decodeResponse(answered, state.request.Questions)
	}
	return responses, nil
}

func completeAll(responses []Response) bool {
	for _, response := range responses {
		if !complete(response) {
			return false
		}
	}
	return len(responses) > 0
}

// deliver hands every call of a decisions task the answers to its questions.
func (t *bundleTask) deliver(responses []Response, err error) {
	for index, state := range t.decision.states {
		for _, call := range state.calls {
			call.err = err
			if err == nil {
				call.decision.response = responses[index].part(call.decision.request.Questions)
			}
			close(call.done)
		}
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
