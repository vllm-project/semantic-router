package extproc

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/match"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/presets"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

// toolNeedQuestionID is the Router's own ID for tool_need, so the question
// never turns on an adapter trained under another ID.
const toolNeedQuestionID = "tools:" + presets.ToolNeed

// What a selection did: selected the judge's tools, found that the request
// needs none, had no candidate, had more candidates than one question takes
// options, or failed and applied its on_error.
const (
	toolsSelected     = "selected"
	toolsNotNeeded    = "not_needed"
	toolsNoCandidates = "no_candidates"
	toolsOverflow     = "overflow"
	toolsSelectFailed = "failed"
)

// toolsSelection is what one selection did, for its replay record, metric
// and log.
type toolsSelection struct {
	mode       string
	deployment string
	candidates int
	kept       []string
	need       float64
	outcome    string
	// action is the on_error or on_overflow a failed or overflowing
	// selection applied, and reason why the call failed.
	action  string
	reason  string
	latency time.Duration
}

// selectTools asks the decision model which of the candidate tools the
// request needs, tool_need and the judge in one call, and keeps or adds
// the tools it chose: filter mode keeps them and the request's first
// preserve_count tools, in the request's order; add mode appends them after
// the request's own, best first. Below need_threshold it keeps or adds none
// of the judge's tools. It returns the failure to answer the client with
// when the selection refuses the request.
func (r *OpenAIRouter) selectTools(request *llmprotocol.Request, plugin *config.ToolsPluginConfig, ctx *RequestContext) *routingFailure {
	spec := plugin.Select
	snapshot := extractSemanticRequestSignalsWith(request, r.routingInputSelector())
	input := match.Input{Request: snapshot.UserContent}
	if spec.Input.History {
		input.History = strings.Join(snapshot.PriorUserMessages, "\n\n")
	}
	if strings.TrimSpace(input.Request) == "" {
		return nil
	}
	catalog, candidates := r.toolsCandidates(request, plugin)
	selection := toolsSelection{mode: spec.Mode, candidates: len(candidates)}
	defer r.recordToolsSelection(ctx, &selection)
	switch {
	case len(candidates) == 0:
		selection.outcome = toolsNoCandidates
		return nil
	case len(candidates) > presets.MaxChoices:
		selection.outcome, selection.action = toolsOverflow, spec.EffectiveOnOverflow()
		if selection.action == config.ToolsSelectOnOverflowReject {
			return &routingFailureTooManyTools
		}
		return nil
	}
	input.Candidates = candidates
	started := time.Now()
	kept, need, err := r.askToolsSelection(selectionRequestContext(ctx), spec, input, &selection)
	selection.latency = time.Since(started)
	if err != nil {
		selection.outcome, selection.action, selection.reason = toolsSelectFailed, spec.EffectiveOnError(), modelservice.ErrorReason(err)
		logging.ComponentWarnEvent("extproc", "tools_selection_failed", map[string]interface{}{
			"request_id": requestID(ctx), "deployment": selection.deployment, "on_error": selection.action, "error": err.Error(),
		})
		switch selection.action {
		case config.ToolsSelectOnErrorFail:
			return &routingFailureToolSelection
		case config.ToolsSelectOnErrorStrip:
			keepRequestTools(request, nil, 0)
		}
		return nil
	}
	selection.need, selection.outcome = need, toolsSelected
	if need < spec.EffectiveNeedThreshold() {
		selection.outcome, kept = toolsNotNeeded, nil
	}
	selection.kept = kept
	if spec.Mode == config.ToolsSelectModeAdd {
		addCatalogTools(request, catalog, kept)
	} else {
		keepRequestTools(request, kept, spec.PreserveCount)
	}
	return nil
}

// toolsCandidates are the tools a selection chooses among: in filter mode
// the request's own, which the policy already filtered; in add mode the
// catalog's tools the policy allows and the request does not carry, by name.
func (r *OpenAIRouter) toolsCandidates(request *llmprotocol.Request, plugin *config.ToolsPluginConfig) (map[string]llmprotocol.Tool, []match.Candidate) {
	var candidates []match.Candidate
	if plugin.Select.Mode == config.ToolsSelectModeFilter {
		for _, tool := range request.Tools {
			candidates = append(candidates, match.Candidate{ID: tool.Name, Text: tool.Description})
		}
		return nil, candidates
	}
	loaded := r.toolCatalogs[plugin.Select.ToolsDBPath]
	if loaded == nil {
		return nil, nil
	}
	carried := make(map[string]bool, len(request.Tools))
	for _, tool := range request.Tools {
		carried[tool.Name] = true
	}
	catalog := map[string]llmprotocol.Tool{}
	for _, entry := range plugin.Policy.AllowedTools(loaded.tools) {
		if carried[entry.Tool.Name] {
			continue
		}
		catalog[entry.Tool.Name] = entry.Tool
		candidates = append(candidates, match.Candidate{ID: entry.Tool.Name, Text: entry.Description})
	}
	return catalog, candidates
}

// askToolsSelection asks tool_need and the selection's judge about the same
// state, as two participants of one bundle, which sends them in one call to
// the judge's deployment. It returns the tools the judge kept, best first,
// and P(tool_need). The selection's timeout_ms bounds both, and a failure of
// either fails the selection, whose on_error decides what follows.
func (r *OpenAIRouter) askToolsSelection(ctx context.Context, spec *config.ToolsSelectConfig, input match.Input, selection *toolsSelection) ([]string, float64, error) {
	block := spec.EffectiveMatch()
	block.Judge.OnError, block.Judge.TimeoutMs = config.MatchOnErrorFail, spec.TimeoutMs
	selection.deployment = r.Config.MatchJudgeDeployment(*block.Judge)
	ctx, cancel := context.WithTimeout(ctx, r.Config.ToolsSelectTimeout(spec, selection.deployment))
	defer cancel()
	stage, bundle := modelservice.WithBundle(ctx, 0)
	leave := bundle.Join()
	var result match.Result
	var need float64
	errs := make([]error, 2)
	modelservice.Fan(stage, 2, func(i int) {
		if i == 0 {
			result, errs[0] = r.matcher().Match(stage, config.MatchToolsSelect, &block, input)
			return
		}
		need, errs[1] = r.askToolNeed(stage, selection.deployment, input.State())
	})
	leave()
	if err := errors.Join(errs...); err != nil {
		return nil, 0, err
	}
	kept := make([]string, len(result.Kept))
	for index, tool := range result.Kept {
		kept[index] = tool.ID
	}
	return kept, need, nil
}

// askToolNeed asks the tool_need preset, compiled for the deployment's
// model, about state and returns P(yes).
func (r *OpenAIRouter) askToolNeed(ctx context.Context, deployment string, state modelservice.Request) (float64, error) {
	answerer := routerMatchAnswerer{r}
	card, ok := answerer.Card(deployment)
	if !ok {
		return 0, fmt.Errorf("%w: deployment %q has no model card ready", modelservice.ErrUnavailable, deployment)
	}
	preset, _ := presets.Lookup(presets.ToolNeed, presets.StageRequest)
	question := modelservice.PresetQuestion(toolNeedQuestionID, preset)
	definition := modelservice.TaskDefinition{ID: question.ID, Stage: preset.Stage, Output: preset.Type, Preset: preset.ID}
	plan, err := modelservice.CompileTask(definition, question, card)
	if err != nil {
		return 0, err
	}
	response, _, err := modelservice.ExecuteTaskPlans(ctx, answerer, deployment, state, []modelservice.TaskPlan{plan})
	if err != nil {
		modelservice.RecordUnknown(deployment, modelservice.ErrorReason(err), 1)
		return 0, err
	}
	answer, ok := response.Answers[toolNeedQuestionID]
	if !ok || answer.Error != "" {
		modelservice.RecordUnknown(deployment, "answer_error", 1)
		return 0, fmt.Errorf("tool_need: the model could not answer: %q", answer.Error)
	}
	return answer.Noul, nil
}

// keepRequestTools keeps the request's tools the selection chose and its
// first preserve tools, in the request's order.
func keepRequestTools(request *llmprotocol.Request, chosen []string, preserve int) {
	kept := make([]llmprotocol.Tool, 0, len(request.Tools))
	for index, tool := range request.Tools {
		if index < preserve || slices.Contains(chosen, tool.Name) {
			kept = append(kept, tool)
		}
	}
	if len(kept) != len(request.Tools) {
		request.Tools = kept
		request.Generation++
	}
}

// addCatalogTools appends the catalog's tools the selection chose after the
// request's own, best first.
func addCatalogTools(request *llmprotocol.Request, catalog map[string]llmprotocol.Tool, chosen []string) {
	for _, name := range chosen {
		tool := catalog[name]
		tool.InputSchema = slices.Clone(tool.InputSchema)
		if tool.Strict != nil {
			strict := *tool.Strict
			tool.Strict = &strict
		}
		request.Tools = append(request.Tools, tool)
	}
	if len(chosen) > 0 {
		request.Generation++
	}
}

// recordToolsSelection counts the selection, logs it and adds it to the
// request's replay record: names and scores only, never the request.
func (r *OpenAIRouter) recordToolsSelection(ctx *RequestContext, selection *toolsSelection) {
	decision := requestDecisionStateKey(ctx)
	metrics.RecordPluginExecution(config.DecisionPluginTools, decision, selection.outcome, selection.latency.Seconds())
	metadata := map[string]string{
		"mode":       selection.mode,
		"candidates": strconv.Itoa(selection.candidates),
		"kept":       strings.Join(selection.kept, ","),
	}
	if selection.deployment != "" {
		metadata["deployment"] = selection.deployment
		metadata["latency_ms"] = formatMilliseconds(selection.latency)
	}
	if selection.outcome == toolsSelected || selection.outcome == toolsNotNeeded {
		metadata["tool_need"] = strconv.FormatFloat(selection.need, 'f', 4, 64)
	}
	if selection.action != "" {
		metadata["action"] = selection.action
	}
	logging.ComponentDebugEvent("extproc", "tools_selection", map[string]interface{}{
		"request_id": requestID(ctx), "decision": decision, "outcome": selection.outcome, "metadata": metadata,
	})
	r.recordToolsOutcome(ctx, routerreplay.Outcome{
		Timestamp: time.Now().UTC(),
		Source:    "tools-plugin",
		Target:    "select",
		Verdict:   selection.outcome,
		Reason:    selection.reason,
		Metadata:  metadata,
	})
}

func requestID(ctx *RequestContext) string {
	if ctx == nil {
		return ""
	}
	return ctx.RequestID
}
