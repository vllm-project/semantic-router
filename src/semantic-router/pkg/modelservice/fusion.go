package modelservice

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
)

// A request stage fans a signal out into one classify call per text piece:
// PII and jailbreak chunks, and every message a PII rule reads. As separate
// bundle tasks they count against the runtime's bundle cap and its per-model
// admission one by one, so a flush fuses classify calls that differ only in
// their inputs and deadline into one task per model, head and options, up to
// the model's input cap, and splits the answer back to each caller.

// bundleTask is one task of a /v1/bundle request and the calls it answers; a
// fused classify task answers several, sizes[i] inputs each, and a decisions
// task answers each of its calls' questions (see fusion_decisions.go).
type bundleTask struct {
	task     api.BundleTask
	calls    []*bundleCall
	sizes    []int
	decision *fusedDecision
}

// fuse turns one client's parked classify, embeddings and rerank calls into
// bundle tasks in call order.
func fuse(client *Client, calls []*bundleCall) []*bundleTask {
	tasks := make([]*bundleTask, 0, len(calls))
	open := make(map[string]*bundleTask)
	items := make(map[*bundleTask]api.ClassifyItemList)
	for _, call := range calls {
		key, list, limit := fusible(client, call.task)
		task := open[key]
		if key == "" || task == nil || len(items[task])+len(list) > limit {
			task = &bundleTask{task: call.task}
			tasks = append(tasks, task)
			if key != "" {
				open[key] = task
			}
		}
		task.calls = append(task.calls, call)
		task.sizes = append(task.sizes, len(list))
		items[task] = append(items[task], list...)
	}
	for task, list := range items {
		if len(task.calls) > 1 {
			task.task = fusedClassify(task.task, task.calls, list)
		}
	}
	return tasks
}

// fusible returns the fusion key of a classify task (its request without the
// inputs and deadline), its inputs and the model's input cap, or "" when the
// task cannot be fused: another surface, an unknown cap, or inputs that are
// not an item list.
func fusible(client *Client, task api.BundleTask) (string, api.ClassifyItemList, int) {
	request := task.Classify
	if request == nil {
		return "", nil, 0
	}
	limit := client.inputCap(request.Model)
	list, err := request.Input.AsClassifyItemList()
	if limit < 2 || err != nil || len(list) == 0 || len(list) >= limit {
		return "", nil, 0
	}
	shape := *request
	shape.Input = api.ClassifyInput{}
	if request.Options != nil {
		options := *request.Options
		options.DeadlineMs = nil
		shape.Options = &options
	}
	key, err := json.Marshal(shape)
	if err != nil {
		return "", nil, 0
	}
	return string(key), list, limit
}

// fusedClassify is the first call's request with every call's inputs and the
// latest deadline (none if any call has none); each caller still stops
// waiting at its own deadline.
func fusedClassify(first api.BundleTask, calls []*bundleCall, list api.ClassifyItemList) api.BundleTask {
	request := *first.Classify
	if err := request.Input.FromClassifyItemList(list); err != nil {
		return first
	}
	var latest *float64
	for index, call := range calls {
		options := call.task.Classify.Options
		if options == nil || options.DeadlineMs == nil {
			latest = nil
			break
		}
		if index == 0 || *options.DeadlineMs > *latest {
			latest = options.DeadlineMs
		}
	}
	if request.Options != nil {
		options := *request.Options
		options.DeadlineMs = latest
		request.Options = &options
	}
	return api.BundleTask{Id: first.Id, Classify: &request}
}

// answer hands every call its part of the task's result.
func (t *bundleTask) answer(result api.BundleResult, err error) {
	if t.decision != nil {
		t.answerDecision(result, err)
		return
	}
	if len(t.calls) == 1 {
		call := t.calls[0]
		call.result, call.err = result, err
		close(call.done)
		return
	}
	var parts []api.ClassifyResponse
	if err == nil && result.Status == http.StatusOK && result.Classify != nil {
		parts = splitClassify(*result.Classify, t.sizes)
	}
	for index, call := range t.calls {
		switch {
		case err != nil:
			call.err = err
		case result.Status != http.StatusOK || result.Classify == nil:
			call.result = result
		case parts == nil:
			call.err = fmt.Errorf("%w: fused classify response does not match its inputs", ErrFailed)
		default:
			call.result = api.BundleResult{Id: call.task.Id, Status: result.Status, Classify: &parts[index]}
		}
		close(call.done)
	}
}

// splitClassify cuts a fused response into one per call, sizes[i] results
// each, reindexed from zero; nil when the results do not cover the inputs
// exactly once.
func splitClassify(body api.ClassifyResponse, sizes []int) []api.ClassifyResponse {
	starts := make([]int, len(sizes)+1)
	for index, size := range sizes {
		starts[index+1] = starts[index] + size
	}
	if len(body.Results) != starts[len(sizes)] {
		return nil
	}
	parts := make([]api.ClassifyResponse, len(sizes))
	for index := range parts {
		parts[index] = api.ClassifyResponse{Model: body.Model, Head: body.Head, Kind: body.Kind, Labels: body.Labels, Results: make([]api.ClassifyResult, 0, sizes[index])}
	}
	owner := 0
	for _, result := range body.Results {
		if result.Index < 0 || result.Index >= starts[len(sizes)] {
			return nil
		}
		for result.Index < starts[owner] || result.Index >= starts[owner+1] {
			owner = (owner + 1) % len(sizes)
		}
		result.Index -= starts[owner]
		parts[owner].Results = append(parts[owner].Results, result)
		if result.Input != nil {
			parts[owner].Usage.InputTokens += result.Input.Tokens
		}
	}
	for index := range parts {
		if len(parts[index].Results) != sizes[index] {
			return nil
		}
	}
	return parts
}
