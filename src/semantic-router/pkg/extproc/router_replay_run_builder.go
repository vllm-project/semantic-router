package extproc

import (
	"slices"
	"sort"
	"strconv"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

const replayRunMainAgent = "main"

// replayRunBuilder folds one session's records into its run, in any order.
type replayRunBuilder struct {
	sessionID string
	entries   []replayRunEntry
	cost      *replayCostAccumulator
	agentCost map[string]*replayCostAccumulator
	// firstMain is the main agent's first call, whose prompt is the task.
	firstMain struct {
		at       time.Time
		recordID string
		prompt   string
	}
	// named is the latest call that named the session, and titles counts
	// the session-title requests among the calls.
	named struct {
		at   time.Time
		id   string
		name string
	}
	titles int
}

func newReplayRunBuilder(sessionID string) *replayRunBuilder {
	return &replayRunBuilder{sessionID: sessionID, cost: newReplayCostAccumulator(), agentCost: make(map[string]*replayCostAccumulator)}
}

func (b *replayRunBuilder) add(record routerreplay.RoutingRecord) {
	entry := newReplayRunEntry(record)
	b.entries = append(b.entries, entry)
	b.cost.add(record)
	if b.agentCost[entry.agent.ID] == nil {
		b.agentCost[entry.agent.ID] = newReplayCostAccumulator()
	}
	b.agentCost[entry.agent.ID].add(record)
	// Scans come in no set order: between main calls of one instant, the
	// higher id is the first, as when scans came newest first.
	first := &b.firstMain
	if entry.agent.ID == "" && entry.parentRequestID == "" && record.RequestKind == "" &&
		(first.recordID == "" || record.Timestamp.Before(first.at) ||
			(record.Timestamp.Equal(first.at) && record.ID > first.recordID)) {
		first.at, first.recordID, first.prompt = record.Timestamp, record.ID, looper.WithoutSystemReminders(record.Prompt)
	}
	if record.RequestKind == store.RequestKindSessionTitle {
		b.titles++
	}
	named := &b.named
	if record.SessionName != "" && (named.at.Before(record.Timestamp) || (named.at.Equal(record.Timestamp) && named.id < record.ID)) {
		named.at, named.id, named.name = record.Timestamp, record.ID, record.SessionName
	}
}

// titlesOnly reports a session that only asked for a title: the thread
// Codex starts to title another, which no field ties to that one.
func (b *replayRunBuilder) titlesOnly() bool {
	return b.titles == len(b.entries)
}

// build returns the run's summary and its calls in time order, each Looper
// hop under the call that made it.
func (b *replayRunBuilder) build() (routerreplay.RunSummary, []routerreplay.RunCall) {
	sort.SliceStable(b.entries, func(i, j int) bool {
		if b.entries[i].call.Timestamp.Equal(b.entries[j].call.Timestamp) {
			return b.entries[i].call.RecordID < b.entries[j].call.RecordID
		}
		return b.entries[i].call.Timestamp.Before(b.entries[j].call.Timestamp)
	})
	entries := b.nestHops()
	lanes := newReplayRunLanes()
	calls := make([]routerreplay.RunCall, 0, len(entries))
	for _, entry := range entries {
		calls = append(calls, lanes.place(entry))
	}
	summary := routerreplay.RunSummary{
		SessionID: b.sessionID, Name: b.named.name, Prompt: b.firstMain.prompt, Calls: len(calls),
		Agents: lanes.agents(b.agentCost, calls), Cost: replayRunCost(b.cost, calls),
		Outcome: replayRunOutcome(entries, calls), Events: replayRunEvents(calls), Models: replayRunModels(calls),
	}
	for _, entry := range entries {
		if summary.Harness == "" {
			summary.Harness = entry.agent.Harness
		}
	}
	if len(calls) > 0 {
		summary.StartedAt, summary.EndedAt = replayRunSpan(calls)
		summary.DurationMS = summary.EndedAt.Sub(summary.StartedAt).Milliseconds()
	}
	return summary, calls
}

// replayRunSpan is when the first of the calls, or of the Looper hops they
// made, started and the last ended. A Looper call's record starts once its
// hops are done.
func replayRunSpan(calls []routerreplay.RunCall) (start, end time.Time) {
	for _, call := range calls {
		if start.IsZero() || call.Timestamp.Before(start) {
			start = call.Timestamp
		}
		if ended := call.Timestamp.Add(time.Duration(call.DurationMS) * time.Millisecond); ended.After(end) {
			end = ended
		}
		if len(call.Hops) > 0 {
			hopStart, hopEnd := replayRunSpan(call.Hops)
			start, end = earlierTime(start, hopStart), laterTime(end, hopEnd)
		}
	}
	return start, end
}

func earlierTime(a, b time.Time) time.Time {
	if b.Before(a) {
		return b
	}
	return a
}

func laterTime(a, b time.Time) time.Time {
	if b.After(a) {
		return b
	}
	return a
}

// nestHops moves each Looper hop under the call that made it, matched by
// request id. A hop whose call is missing stays a call of its own.
func (b *replayRunBuilder) nestHops() []replayRunEntry {
	byRequest := make(map[string]int)
	var calls, hops []replayRunEntry
	for _, entry := range b.entries {
		if entry.parentRequestID != "" {
			hops = append(hops, entry)
			continue
		}
		if entry.call.RequestID != "" {
			byRequest[entry.call.RequestID] = len(calls)
		}
		calls = append(calls, entry)
	}
	orphaned := false
	for _, hop := range hops {
		index, ok := byRequest[hop.parentRequestID]
		if !ok {
			calls, orphaned = append(calls, hop), true
			continue
		}
		calls[index].call.Hops = append(calls[index].call.Hops, hop.call)
	}
	if orphaned {
		sort.SliceStable(calls, func(i, j int) bool { return calls[i].call.Timestamp.Before(calls[j].call.Timestamp) })
	}
	return calls
}

// replayRunLanes names each agent of a run by its first call, main and then
// subagent 1, 2 and so on, and marks a call whose model differs from the
// one its agent's previous call to the same entrypoint was routed to.
type replayRunLanes struct {
	labels    map[string]string
	order     []*routerreplay.RunAgent
	byLabel   map[string]*routerreplay.RunAgent
	parents   map[string]string
	lastModel map[string]string
	lastCall  map[string]routerreplay.RunCall
}

func newReplayRunLanes() *replayRunLanes {
	lanes := &replayRunLanes{
		labels: map[string]string{"": replayRunMainAgent}, byLabel: make(map[string]*routerreplay.RunAgent),
		parents: make(map[string]string), lastModel: make(map[string]string), lastCall: make(map[string]routerreplay.RunCall),
	}
	main := &routerreplay.RunAgent{Label: replayRunMainAgent}
	lanes.order, lanes.byLabel[replayRunMainAgent] = []*routerreplay.RunAgent{main}, main
	return lanes
}

func (l *replayRunLanes) place(entry replayRunEntry) routerreplay.RunCall {
	call := entry.call
	label, known := l.labels[entry.agent.ID]
	if !known {
		label = "subagent " + strconv.Itoa(len(l.order))
		l.labels[entry.agent.ID] = label
		agent := &routerreplay.RunAgent{Label: label, ID: entry.agent.ID, Kind: entry.agent.Kind}
		agent.Parent = l.labels[entry.agent.ParentID]
		if agent.Parent == "" {
			agent.Parent = replayRunMainAgent
		}
		if spawn, ok := l.lastCall[agent.Parent]; ok {
			agent.SpawnedAfterRecordID = spawn.RecordID
		}
		l.order = append(l.order, agent)
		l.byLabel[label] = agent
	}
	call.Agent = label
	for index := range call.Hops {
		call.Hops[index].Agent = label
	}
	agent := l.byLabel[label]
	if agent.Calls == 0 {
		agent.FirstCallAt = call.Timestamp
	}
	agent.Calls++
	agent.LastCallAt = call.Timestamp
	if call.Model != "" && !slices.Contains(agent.Models, call.Model) {
		agent.Models = append(agent.Models, call.Model)
	}
	key := label + "\x00" + call.Entrypoint
	if previous := l.lastModel[key]; previous != "" && call.Model != "" && previous != call.Model && call.RequestKind == "" {
		call.PreviousModel = previous
		call.Markers = append([]string{routerreplay.RunMarkerSwitch}, call.Markers...)
	}
	if call.Model != "" && call.RequestKind == "" {
		l.lastModel[key] = call.Model
	}
	l.lastCall[label] = call
	return call
}

func (l *replayRunLanes) agents(costs map[string]*replayCostAccumulator, calls []routerreplay.RunCall) []routerreplay.RunAgent {
	byLabel := make(map[string][]routerreplay.RunCall)
	for _, call := range calls {
		byLabel[call.Agent] = append(byLabel[call.Agent], call)
	}
	agents := make([]routerreplay.RunAgent, 0, len(l.order))
	for id, label := range l.labels {
		if accumulator := costs[id]; accumulator != nil {
			l.byLabel[label].Cost = replayRunCost(accumulator, byLabel[label])
		}
	}
	for _, agent := range l.order {
		agents = append(agents, *agent)
	}
	return agents
}

// replayRunCost reads a run's or an agent's spend as the aggregate totals
// it, with the tokens of the calls given.
func replayRunCost(accumulator *replayCostAccumulator, calls []routerreplay.RunCall) routerreplay.RunCost {
	summary := accumulator.result()
	cost := routerreplay.RunCost{
		UnpricedRecords:     summary.Coverage.UnpricedRecords + summary.Coverage.PartlyPricedRecords,
		UnpricedRouterCalls: summary.Coverage.UnpricedRouterCalls,
	}
	for _, group := range summary.ByCurrency {
		cost.ByCurrency = append(cost.ByCurrency, routerreplay.RunCurrencyCost{
			Currency: group.Currency, Total: group.TotalSpend, RouterCallSpend: group.RouterCallSpend,
		})
	}
	for _, call := range calls {
		if call.PromptTokens != nil {
			cost.PromptTokens += *call.PromptTokens
		}
		if call.CompletionTokens != nil {
			cost.CompletionTokens += *call.CompletionTokens
		}
	}
	return cost
}

// replayRunOutcome reads how a run ended from its main agent's last call,
// or its last call when the main agent made none, counts each tool result
// once, and gathers the outcomes reported for its calls.
func replayRunOutcome(entries []replayRunEntry, calls []routerreplay.RunCall) routerreplay.RunOutcome {
	outcome := routerreplay.RunOutcome{Status: routerreplay.RunStatusUnknown}
	results, failures := make(map[string]bool), make(map[string]bool)
	lastMain, lastAny := -1, -1
	for index, entry := range entries {
		for _, id := range entry.toolResults {
			results[id] = true
		}
		for _, id := range entry.toolFailures {
			failures[id] = true
		}
		if calls[index].RequestKind == "" {
			lastAny = index
			if calls[index].Agent == replayRunMainAgent {
				lastMain = index
			}
		}
		for _, reported := range calls[index].Outcomes {
			outcome.Reported = append(outcome.Reported, routerreplay.RunOutcomeAt{RecordID: calls[index].RecordID, Outcome: reported})
		}
	}
	outcome.ToolResults, outcome.ToolFailures = len(results), len(failures)
	if lastMain < 0 {
		lastMain = lastAny
	}
	if lastMain < 0 {
		return outcome
	}
	final := calls[lastMain]
	switch {
	case final.LifecycleState == routerreplay.LifecycleInProgress:
		outcome.Status = routerreplay.RunStatusInProgress
	case final.LifecycleState == routerreplay.LifecycleFailed || final.ResponseStatus >= 400:
		outcome.Status = routerreplay.RunStatusFailed
	case final.LifecycleState == routerreplay.LifecycleAborted:
		outcome.Status = routerreplay.RunStatusAborted
	case final.FinalAnswer:
		outcome.Status, outcome.FinalAnswerRecordID = routerreplay.RunStatusAnswered, final.RecordID
	}
	return outcome
}

func replayRunEvents(calls []routerreplay.RunCall) map[string]int {
	events := make(map[string]int)
	var count func(routerreplay.RunCall)
	count = func(call routerreplay.RunCall) {
		for _, marker := range call.Markers {
			events[marker]++
		}
		for _, hop := range call.Hops {
			count(hop)
		}
	}
	for _, call := range calls {
		count(call)
	}
	if len(events) == 0 {
		return nil
	}
	return events
}

// replayRunModels counts the calls each model answered, Looper hops
// included, most used first.
func replayRunModels(calls []routerreplay.RunCall) []routerreplay.AggregateValue {
	counts := make(map[string]int)
	var count func(routerreplay.RunCall)
	count = func(call routerreplay.RunCall) {
		if call.Model != "" {
			counts[call.Model]++
		}
		for _, hop := range call.Hops {
			count(hop)
		}
	}
	for _, call := range calls {
		count(call)
	}
	return sortRouterReplayAggregateValues(counts, 0)
}
