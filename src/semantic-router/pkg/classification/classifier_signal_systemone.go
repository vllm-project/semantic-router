package classification

import (
	"context"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

const systemOneSignalErrorPrefix = "systemone_"

// unreadToolResult is why a rule that reads the tool results read only part
// of them: a tool result held content the Router cannot read as text.
const unreadToolResult = "unreadable_tool_result"

// unreadImage is why a rule that reads the images read only part of them: an
// image no decision model decodes, such as a remote URL or a GIF.
const unreadImage = "unreadable_image"

// SystemOneQuestionID is the ID a rule's question carries on the wire: the
// Router's own, with a ':', so it is never an ID a model family was trained
// on and never turns a trained adapter on.
func SystemOneQuestionID(rule string) string {
	return config.SignalTypeSystemOne + ":" + rule
}

// presetQuestionID is the Router's own ID of a preset's question, which every
// application rule of the preset asked in one call shares.
func presetQuestionID(preset string) string {
	return "preset:" + preset
}

// SetDecisionDecider replaces the model runtime used by systemone signals (tests and embedders).
func (c *Classifier) SetDecisionDecider(decider modelservice.Decider) {
	c.decisionDecider = decider
}

// SetSystemOneDecider replaces the remote System One clients used by
// systemone signals to systemone deployments (tests and embedders).
func (c *Classifier) SetSystemOneDecider(remote modelservice.Remote) {
	c.systemOneDecider = remote
}

func (c *Classifier) systemOne() modelservice.Remote {
	if c.systemOneDecider != nil {
		return c.systemOneDecider
	}
	if c.models != nil {
		return c.models.systemOne
	}
	return nil
}

func (c *Classifier) decider() modelservice.Decider {
	if c.decisionDecider != nil {
		return c.decisionDecider
	}
	if decider, ok := c.models.decider(); ok {
		return decider
	}
	return modelservice.Default()
}

// ConversationTurn is one user or assistant message before the current user
// turn, in conversation order.
type ConversationTurn struct {
	Role string
	Text string
}

// systemOneReading is what a stage gives its rules to read: the current user
// turn as it came, the conversation before it, the request's tool results,
// its images and the current turn's videos, and at the response stage the
// model's answer and its grounding context. toolResultsIncomplete is whether
// a tool result held content that is not text, which toolResults leaves out,
// and imagesIncomplete whether an image is one no decision model decodes,
// which images leaves out.
type systemOneReading struct {
	request               string
	history               []string
	turns                 []ConversationTurn
	toolResults           []string
	toolResultsIncomplete bool
	imagesIncomplete      bool
	answer                string
	context               string
	images                []string
	videos                []string
}

// previousAnswer is the last assistant turn before the current user turn,
// the answer that turn replies to.
func (r systemOneReading) previousAnswer() string {
	for index := len(r.turns) - 1; index >= 0; index-- {
		if r.turns[index].Role == "assistant" {
			return r.turns[index].Text
		}
	}
	return ""
}

// state is the request a rule's question is asked about. A rule that reads
// only the current turn asks about its text; any other reads named parts,
// those the runtime reads by role (request as the user part, answer, and the
// others as context). A part with nothing to read is left out.
func (r systemOneReading) state(rule config.SystemOneRule) modelservice.Request {
	input := rule.Input
	if !input.ReadsRequest() {
		return modelservice.Request{State: strings.Join(r.besidesRequest(input), "\n\n")}
	}
	if request, ok := r.spanScan(rule); ok {
		return request
	}
	parts := map[string]string{}
	add := func(name, text string) {
		if strings.TrimSpace(text) != "" {
			parts[name] = text
		}
	}
	if input.History {
		add("history", strings.Join(r.history, "\n\n"))
	}
	if input.PreviousTurns > 0 {
		turns := r.turns[max(0, len(r.turns)-input.PreviousTurns):]
		lines := make([]string, len(turns))
		for index, turn := range turns {
			lines[index] = turn.Role + ": " + turn.Text
		}
		add("previous_turns", strings.Join(lines, "\n"))
	}
	if input.PreviousAnswer {
		add("previous_answer", r.previousAnswer())
	}
	if input.ToolResults {
		add("tool_results", strings.Join(r.toolResults, "\n\n"))
	}
	if input.Context {
		add("context", r.context)
	}
	if input.Answer {
		add("answer", r.answer)
	}
	request := modelservice.Request{State: r.request}
	if len(parts) > 0 {
		add("request", r.request)
		request = modelservice.Request{Parts: parts}
	}
	if input.Image {
		request.Images = slices.Clone(r.images)
	}
	if input.Video {
		request.Videos = slices.Clone(r.videos)
	}
	return request
}

// spanScan is the state of a request-stage span rule that reads the earlier
// user turns or the tool results and nothing else: a span question locates
// its spans in the request alone, so that text is scanned with the request
// instead of being read as context.
func (r systemOneReading) spanScan(rule config.SystemOneRule) (modelservice.Request, bool) {
	input := rule.Input
	if rule.Question.Type != config.SystemOneQuestionSpan || input.Answer || input.Context || input.PreviousTurns > 0 {
		return modelservice.Request{}, false
	}
	scanned := r.besidesRequest(input)
	if len(scanned) == 0 {
		return modelservice.Request{}, false
	}
	request := modelservice.Request{State: strings.Join(append(scanned, r.request), "\n\n")}
	if input.Image {
		request.Images = slices.Clone(r.images)
	}
	if input.Video {
		request.Videos = slices.Clone(r.videos)
	}
	return request, true
}

// besidesRequest is the text a rule reads in the earlier user turns and the tool
// results, without the turns and results that hold none.
func (r systemOneReading) besidesRequest(input config.SystemOneInput) []string {
	var texts []string
	if input.History {
		texts = append(texts, r.history...)
	}
	if input.ToolResults {
		texts = append(texts, r.toolResults...)
	}
	return slices.DeleteFunc(texts, func(text string) bool { return strings.TrimSpace(text) == "" })
}

// hasInput reports whether a rule has text to read: a rule that leaves the
// current user turn out has none until the earlier turns or the tool results
// it reads hold some, and is not asked, so it does not match.
func (r systemOneReading) hasInput(rule config.SystemOneRule) bool {
	return rule.Input.ReadsRequest() || len(r.besidesRequest(rule.Input)) > 0
}

// unread is why a full-coverage rule misses part of its input, which it then
// resolves as on_unscanned says: a tool result it cannot read as text, or an
// image no decision model decodes. It is empty when the rule reads it all.
func (r systemOneReading) unread(rule config.SystemOneRule) string {
	switch {
	case !rule.FullCoverage():
		return ""
	case r.toolResultsIncomplete && rule.Input.ToolResults:
		return unreadToolResult
	case r.imagesIncomplete && rule.Input.Image:
		return unreadImage
	}
	return ""
}

// askedRule is one rule a stage asks: a systemone rule, or an application
// rule compiled into one, which publishes its matches and values under its
// application type and shares its preset's question with the stage's other
// rules of the preset. path is where the configuration asks it when that is
// not a rule (fusion grounding). unread is why it misses part of its input,
// empty when it reads it all. minCoverage is the share of a long input's
// windows best_effort must have read to decide on them.
type askedRule struct {
	signalType  string
	rule        config.SystemOneRule
	preset      string
	failClosed  bool
	path        string
	unread      string
	minCoverage float64
}

func systemOneAsked(rule config.SystemOneRule) askedRule {
	return askedRule{signalType: config.SignalTypeSystemOne, rule: rule}
}

func applicationAsked(rule config.ApplicationRule) askedRule {
	return askedRule{signalType: rule.Type, rule: rule.Rule, preset: rule.Preset, failClosed: rule.FailClosed, minCoverage: rule.MinCoverage}
}

// questionID is the ID of the question whose answer the rule reads.
func (a askedRule) questionID() string {
	if a.preset != "" {
		return presetQuestionID(a.preset)
	}
	return SystemOneQuestionID(a.rule.Name)
}

// systemOneCall is the rules a stage asks one deployment about one state.
// requestAlone is whether that state is the request text and nothing else.
type systemOneCall struct {
	deployment   string
	request      modelservice.Request
	rules        []askedRule
	requestAlone bool
	// scan is the guard scan a window call reads a window of, and window
	// that window's index.
	scan   *guardScan
	window int
}

// systemOneCalls groups rules by the deployment that answers them and the
// state they read, in rule order: every rule of a stage goes to its
// deployment in one call, a batch with an entry per state when it reads
// several. A rule that does not read its whole input reads the routing view
// of its state, and a guard rule whose input is longer than one window reads
// it in windows: the scans returned, whose windows windowReaders reads.
func (c *Classifier) systemOneCalls(rules []askedRule, reading systemOneReading) ([]*systemOneCall, []*guardScan) {
	var calls []*systemOneCall
	var scans []*guardScan
	for _, asked := range rules {
		if !reading.hasInput(asked.rule) {
			continue
		}
		asked.unread = reading.unread(asked.rule)
		deployment := c.Config.SystemOneRuleDeployment(asked.rule)
		state := reading.state(asked.rule)
		switch {
		case guardScanned(asked):
			if scan := c.guardScanOf(&scans, deployment, state, reading); scan != nil {
				scan.rules = append(scan.rules, asked)
				continue
			}
		case !asked.rule.FullCoverage():
			budget, headShare := c.routingViewBudget(deployment)
			state = boundedState(state, budget, headShare)
		}
		index := slices.IndexFunc(calls, func(call *systemOneCall) bool {
			return call.scan == nil && call.deployment == deployment && sameState(call.request, state)
		})
		if index < 0 {
			calls = append(calls, &systemOneCall{deployment: deployment, request: state, requestAlone: state.Parts == nil && state.State == reading.request})
			index = len(calls) - 1
		}
		calls[index].rules = append(calls[index].rules, asked)
	}
	return calls, scans
}

// guardScanOf is the scan of state on deployment among scans, a new one when
// there is none, or nil when the state fits one window.
func (c *Classifier) guardScanOf(scans *[]*guardScan, deployment string, state modelservice.Request, reading systemOneReading) *guardScan {
	for _, scan := range *scans {
		if scan.deployment == deployment && sameState(scan.whole, state) {
			return scan
		}
	}
	scan := newGuardScan(deployment, state, c.scanWindowBudget(deployment))
	if scan == nil {
		return nil
	}
	scan.requestAlone = state.Parts == nil && state.State == reading.request
	*scans = append(*scans, scan)
	return scan
}

func sameState(a, b modelservice.Request) bool {
	if a.State != b.State || len(a.Parts) != len(b.Parts) || !slices.Equal(a.Images, b.Images) || !slices.Equal(a.Videos, b.Videos) {
		return false
	}
	for name, text := range a.Parts {
		if other, ok := b.Parts[name]; !ok || other != text {
			return false
		}
	}
	return true
}

// evaluateSystemOneSignals asks every used request-stage rule, one call per
// deployment, with the request as it came as the current turn. Calls to
// different deployments run in parallel. A rule that names no deployment
// asks the decision model, like the application signals whose call it joins.
// A failed or late call leaves its signals unknown; it never fails the
// request.
func (c *Classifier) evaluateSystemOneSignals(
	ctx context.Context,
	results *SignalResults,
	mu *sync.Mutex,
	reading systemOneReading,
	usedSignals map[string]bool,
) {
	if ctx == nil {
		ctx = context.Background()
	}
	start := time.Now()
	var rules []askedRule
	for _, rule := range c.Config.SystemOneRules {
		if rule.EffectiveStage() == config.SystemOneStageRequest && signalRuleUsed(usedSignals, config.SignalTypeSystemOne, rule.Name) {
			rules = append(rules, systemOneAsked(rule))
		}
	}
	c.askSystemOneRules(ctx, results, mu, rules, reading)
	mu.Lock()
	results.Metrics.SystemOne.ExecutionTimeMs = float64(time.Since(start).Microseconds()) / 1000.0
	mu.Unlock()
}

// ResponseInput is what a response-stage rule reads: the request that was
// answered, the model's answer and its grounding context (retrieved
// documents and tool results), and the conversation of the request.
type ResponseInput struct {
	Request           string
	Answer            string
	Context           string
	PriorUserMessages []string
	Turns             []ConversationTurn
	ToolResultTexts   []string
	// ToolResultsIncomplete is whether a tool result held content that is
	// not text, which ToolResultTexts leaves out.
	ToolResultsIncomplete bool
	Images                []string
	Videos                []string
}

func (input ResponseInput) reading() systemOneReading {
	return systemOneReading{
		request: input.Request, history: input.PriorUserMessages, turns: input.Turns, toolResults: input.ToolResultTexts,
		toolResultsIncomplete: input.ToolResultsIncomplete, answer: input.Answer, context: input.Context, images: input.Images, videos: input.Videos,
	}
}

func newResponseSignalResults() *SignalResults {
	return &SignalResults{
		SignalConfidences: map[string]float64{}, SignalValues: map[string]float64{}, SignalErrors: map[string]string{},
		SignalSpans: map[string][]modelservice.Span{},
	}
}

// askSystemOneRules asks rules about reading. Rules read only the images a
// decision model decodes. A rule whose tool results hold no text at all, only
// content it cannot read, is not asked and resolves as its on_unscanned says.
func (c *Classifier) askSystemOneRules(ctx context.Context, results *SignalResults, mu *sync.Mutex, rules []askedRule, reading systemOneReading) {
	reading.images, reading.imagesIncomplete = decodedImages(reading.images)
	for _, asked := range rules {
		if reason := reading.unread(asked.rule); reason != "" && !reading.hasInput(asked.rule) {
			mu.Lock()
			c.recordGuardRead(results, asked, false)
			c.resolveUnread(results, asked, c.Config.SystemOneRuleDeployment(asked.rule), reason)
			mu.Unlock()
		}
	}
	calls, scans := c.systemOneCalls(rules, reading)
	work := make([]func(), 0, len(calls))
	for _, call := range calls {
		work = append(work, func() { c.evaluateSystemOneDeployment(ctx, results, mu, call) })
	}
	readers, release := c.windowReaders(ctx, scans)
	work = append(work, readers...)
	modelservice.Fan(ctx, len(work), func(i int) { work[i]() })
	release()
	for _, scan := range scans {
		c.applyScan(results, mu, scan)
	}
}

// decodedImageTypes are the image data URLs the decision models decode.
var decodedImageTypes = []string{"data:image/png;", "data:image/jpeg;", "data:image/jpg;", "data:image/webp;"}

// undecodedImage reports an image no decision model decodes: one that was
// not inline, an empty entry, or another format such as GIF.
func undecodedImage(image string) bool {
	return !slices.ContainsFunc(decodedImageTypes, func(prefix string) bool { return strings.HasPrefix(image, prefix) })
}

// decodedImages keeps the images a decision model decodes and reports
// whether it left one out.
func decodedImages(images []string) ([]string, bool) {
	kept := slices.DeleteFunc(slices.Clone(images), undecodedImage)
	return kept, len(kept) < len(images)
}

func (c *Classifier) evaluateSystemOneDeployment(
	ctx context.Context,
	results *SignalResults,
	mu *sync.Mutex,
	call *systemOneCall,
) {
	c.askCall(ctx, results, mu, call, nil)
}

// askCall sends one call, hands its response to observe (when set), and
// applies each rule's answer to results.
func (c *Classifier) askCall(
	ctx context.Context,
	results *SignalResults,
	mu *sync.Mutex,
	call *systemOneCall,
	observe func(modelservice.Response, error),
) {
	response, latency, err := c.sendCall(ctx, call)
	mu.Lock()
	defer mu.Unlock()
	if observe != nil {
		observe(response, err)
	}
	c.applyCall(results, call, response, err, latency)
}

// sendCall asks one call's questions and returns the answers, how long the
// call took, and why there are none.
func (c *Classifier) sendCall(ctx context.Context, call *systemOneCall) (modelservice.Response, float64, error) {
	deployment, rules := call.deployment, call.rules
	// The deployment's scan budget keeps a stage's questions to it in one
	// call: a full-coverage rule reads its input whole, up to the budget, and
	// any other only its first tokens.
	request := call.request
	request.MaxTokens = c.Config.ModelDeployments[deployment].ScanBudget()
	request.Questions = callQuestions(rules)
	remote := c.Config.IsSystemOneDeployment(deployment)
	callCtx, cancel := systemOneCallContext(ctx, rules, remote, c.Config.ModelDeployments[deployment].CallTimeout())
	defer cancel()
	started := time.Now()
	var response modelservice.Response
	var err error
	if remote {
		response, err = askSystemOneDeployment(callCtx, c.systemOne(), deployment, request)
	} else if card, ok := c.decisionTaskCard(deployment); ok {
		response, err = modelservice.ExecuteQuestions(callCtx, c.decider(), deployment, card, request)
	} else if c.models != nil && c.models.runtime != nil && c.models.runtime.Services() != nil {
		err = modelservice.ErrUnavailable
	} else {
		response, err = c.decider().Decide(callCtx, deployment, request)
	}
	if err != nil {
		modelservice.RecordUnknown(deployment, modelservice.ErrorReason(err), len(rules))
	}
	return response, time.Since(started).Seconds(), err
}

// applyCall applies each rule's answer in one call's response to results.
func (c *Classifier) applyCall(results *SignalResults, call *systemOneCall, response modelservice.Response, err error, latency float64) {
	deployment := call.deployment
	recordPrivacyEvidence(results, call, response, err)
	for _, asked := range call.rules {
		rule := asked.rule
		key := signalConfidenceKey(asked.signalType, rule.Name)
		c.recordSignalExtraction(asked.signalType, rule.Name, latency)
		if err != nil {
			c.recordGuardRead(results, asked, false)
			results.SignalErrors[key] = systemOneSignalErrorPrefix + modelservice.ErrorReason(err)
			c.applyFailClosed(results, asked, deployment, modelservice.ErrorReason(err))
			continue
		}
		answer, ok := response.Answers[asked.questionID()]
		if !ok || answer.Error != "" {
			c.recordGuardRead(results, asked, false)
			reason := answer.Error
			if !ok {
				reason = "missing_answer"
			}
			if ok && rule.FullCoverage() && unscanned(reason) {
				switch rule.EffectiveOnUnscanned() {
				case config.SystemOneUnscannedMatch, config.SystemOneUnscannedNoMatch:
					c.applyUnscanned(results, asked, deployment, reason)
					continue
				}
			}
			results.SignalErrors[key] = systemOneSignalErrorPrefix + reason
			modelservice.RecordUnknown(deployment, reason, 1)
			c.applyFailClosed(results, asked, deployment, reason)
			continue
		}
		c.recordGuardRead(results, asked, true)
		if asked.unread != "" && c.resolveUnread(results, asked, deployment, asked.unread) {
			continue
		}
		c.applyRuleAnswer(results, asked, answer)
	}
}

// recordGuardRead counts the coverage of a guard rule that read its input
// in one call: all of it, or none.
func (c *Classifier) recordGuardRead(results *SignalResults, asked askedRule, read bool) {
	if !guardScanned(asked) {
		return
	}
	if read {
		c.recordScanCoverage(results, asked, 1, 1)
	} else {
		c.recordScanCoverage(results, asked, 0, 1)
	}
}

// applyRuleAnswer publishes a rule's answer and records what it matched.
func (c *Classifier) applyRuleAnswer(results *SignalResults, asked askedRule, answer modelservice.Answer) {
	rule := asked.rule
	for _, matched := range applySystemOneAnswer(results, asked.signalType, rule, answer) {
		results.appendMatched(asked.signalType, rule, matched)
		c.recordSignalMatch(asked.signalType, matched)
	}
	if rule.Question.Type == config.SystemOneQuestionSpan && results.SignalSpans != nil {
		if spans := matchedSpans(rule, answer); len(spans) > 0 {
			results.SignalSpans[signalConfidenceKey(asked.signalType, rule.Name)] = spans
		}
	}
}

// callQuestions are the questions a call asks: each systemone rule's own and
// each preset's once, which reads the options every rule asking it reads (a
// family's trained form answers it when it covers them).
func callQuestions(rules []askedRule) []modelservice.Question {
	var questions []modelservice.Question
	for _, asked := range rules {
		id := asked.questionID()
		index := slices.IndexFunc(questions, func(question modelservice.Question) bool { return question.ID == id })
		if index < 0 {
			questions = append(questions, askedQuestion(asked))
			continue
		}
		questions[index].Reads = mergeReads(questions[index].Reads, ruleReads(asked.rule))
	}
	return questions
}

// askedQuestion is the question a rule asks: a full-coverage rule reads its
// whole input, which ExecuteQuestions requires proof of, and any other only
// its first tokens.
func askedQuestion(asked askedRule) modelservice.Question {
	rule := asked.rule
	question := SystemOneQuestion(asked.questionID(), rule.Question)
	question.TaskID, question.Stage = config.SignalTypeSystemOne, rule.EffectiveStage()
	if asked.preset != "" {
		question.TaskID = modelservice.PresetTask(asked.preset)
		question.Reads = ruleReads(rule)
	}
	question.Truncate = !rule.FullCoverage()
	return question
}

// ruleReads are the options a rule reads: its labels, or nil, every option,
// for a rule without labels, whose conditions may name any.
func ruleReads(rule config.SystemOneRule) []string {
	if len(rule.Labels) == 0 {
		return nil
	}
	return slices.Clone(rule.Labels)
}

// mergeReads joins the options two askers read; nil is every option.
func mergeReads(reads, more []string) []string {
	if reads == nil || more == nil {
		return nil
	}
	merged := slices.Clone(reads)
	for _, label := range more {
		if !slices.Contains(merged, label) {
			merged = append(merged, label)
		}
	}
	return merged
}

// unscanned reports whether an answer failed because the model could not
// read the whole input.
func unscanned(reason string) bool {
	switch reason {
	case "scan_budget_exceeded", "input_coverage_unknown", "max_length_exceeded", "partial_scan", unreadToolResult, unreadImage:
		return true
	}
	return false
}

// applyUnscanned resolves a full-coverage rule whose model could not read
// the whole input as its on_unscanned says, when that is not unknown: the
// rule and its counted labels match, or nothing does. An unscanned answer
// has no values, so a condition's own predicate resolves the match as the
// condition without one does. reason is the model's error code, which the
// match is logged under.
func (c *Classifier) applyUnscanned(results *SignalResults, asked askedRule, deployment, reason string) {
	if asked.rule.EffectiveOnUnscanned() != config.SystemOneUnscannedMatch {
		return
	}
	c.appendPolicyMatches(results, asked)
	c.recordSignalFailClosed(asked.signalType, asked.rule.Name, deployment, reason)
}

// resolveUnread resolves a full-coverage rule that read only part of its
// input as its on_unscanned says, and reports whether that settled it:
// best_effort leaves the rule to decide on what it read. reason is why part
// went unread, which a match is logged under and an unknown rule's error
// names.
func (c *Classifier) resolveUnread(results *SignalResults, asked askedRule, deployment, reason string) bool {
	policy := asked.rule.EffectiveOnUnscanned()
	// A systemone rule reads its input in one call, so best_effort has no
	// read part to decide on and reads as unknown, as the docs say.
	if policy == config.SystemOneUnscannedBestEffort && asked.signalType == config.SignalTypeSystemOne {
		policy = config.SystemOneUnscannedUnknown
	}
	switch policy {
	case config.SystemOneUnscannedMatch:
		c.applyUnscanned(results, asked, deployment, reason)
	case config.SystemOneUnscannedNoMatch:
	case config.SystemOneUnscannedUnknown:
		results.SignalErrors[signalConfidenceKey(asked.signalType, asked.rule.Name)] = systemOneSignalErrorPrefix + reason
		c.applyFailClosed(results, asked, deployment, reason)
	default:
		return false
	}
	return true
}

// applyFailClosed resolves a fail-closed rule whose deployment gave no answer
// as a match: a guard whose model is down does not pass the request. The
// signal error stays, and SignalErrorMatches marks the match as policy rather
// than evidence, so a decision's rules.on_unknown still decides it. reason is
// why there is no answer, which the match is logged under.
func (c *Classifier) applyFailClosed(results *SignalResults, asked askedRule, deployment, reason string) {
	if !asked.failClosed {
		return
	}
	if results.SignalErrorMatches == nil {
		results.SignalErrorMatches = make(map[string]bool)
	}
	results.SignalErrorMatches[signalConfidenceKey(asked.signalType, asked.rule.Name)] = true
	c.appendPolicyMatches(results, asked)
	c.recordSignalFailClosed(asked.signalType, asked.rule.Name, deployment, reason)
}

// appendPolicyMatches matches a rule and its counted labels without an
// answer to read them from.
func (c *Classifier) appendPolicyMatches(results *SignalResults, asked askedRule) {
	rule := asked.rule
	var matched []string
	if rule.MatchesItself() {
		matched = append(matched, rule.Name)
	}
	for _, label := range rule.CountedLabels() {
		matched = append(matched, rule.Name+":"+label)
	}
	for _, reference := range matched {
		results.appendMatched(asked.signalType, rule, reference)
		c.recordSignalMatch(asked.signalType, reference)
	}
}

// askSystemOneDeployment asks a systemone deployment's remote API. Without
// its client the questions stay unknown.
func askSystemOneDeployment(ctx context.Context, remote modelservice.Remote, deployment string, request modelservice.Request) (modelservice.Response, error) {
	if remote == nil {
		return modelservice.Response{}, modelservice.ErrUnavailable
	}
	card, ok := remote.Card(deployment)
	if !ok {
		return modelservice.Response{}, modelservice.ErrUnknownDeployment
	}
	return modelservice.ExecuteQuestions(ctx, remote, deployment, card, request)
}

// systemOneCallContext bounds a deployment's call by the shortest timeout its
// rules set. Questions to a model_runtime decision model that set none join
// the stage's other questions to it, so they take its deadline rather than
// the default timeout. Questions to a systemone deployment travel with the
// stage's other questions too, but a rule without a timeout takes the
// deployment's own, else the default: the remote API's latency is its own.
// Once the stage sends a call, its questions wait for it as long as its
// latest caller does: a timeout bounds the wait for the other askers, never
// an answer of a call the stage waits for anyway (modelservice.Bundle).
func systemOneCallContext(ctx context.Context, rules []askedRule, remote bool, deploymentTimeout time.Duration) (context.Context, context.CancelFunc) {
	var timeout time.Duration
	for _, asked := range rules {
		rule := asked.rule
		if rule.Deployment == "" && rule.TimeoutMs == 0 && !remote {
			continue
		}
		t := rule.EffectiveTimeout()
		if remote && rule.TimeoutMs == 0 && deploymentTimeout > 0 {
			t = deploymentTimeout
		}
		if timeout == 0 || t < timeout {
			timeout = t
		}
	}
	if timeout == 0 {
		return context.WithCancel(ctx)
	}
	return context.WithTimeout(ctx, timeout)
}

// SystemOneQuestion converts a configured question to the runtime request form.
func SystemOneQuestion(id string, question config.SystemOneQuestion) modelservice.Question {
	converted := modelservice.Question{ID: id, Type: question.Type, Instructions: question.Instructions, Threshold: question.Threshold, Head: question.Head}
	for _, choice := range question.Choices {
		converted.Choices = append(converted.Choices, modelservice.Choice{Key: choice.Key, Description: choice.Description})
	}
	for _, label := range question.Labels {
		converted.Labels = append(converted.Labels, modelservice.Choice{Key: label.Key, Description: label.Description})
	}
	converted.Levels = append(converted.Levels, question.Levels...)
	return converted
}

// applySystemOneAnswer publishes an answer's values and returns the matched
// references: the rule name when the rule matches itself, and "rule:label"
// for each matching option (choice) or label (set, span). A value is
// published under "<type>:<rule>" (P(yes), the expected level, the highest
// probability of a counted label, or for a choice rule without labels the
// chosen option's) and under "<type>:<rule>:<label>" for every option or
// label.
func applySystemOneAnswer(results *SignalResults, signalType string, rule config.SystemOneRule, answer modelservice.Answer) []string {
	key := signalConfidenceKey(signalType, rule.Name)
	predicate := rule.EffectivePredicate()
	publish := func(name string, value float64) {
		results.SignalValues[name] = value
		results.SignalConfidences[name] = value
	}
	switch rule.Question.Type {
	case config.SystemOneQuestionNoul:
		publish(key, answer.Noul)
		if predicateMatches(answer.Noul, predicate) {
			return []string{rule.Name}
		}
	case config.SystemOneQuestionScore:
		results.SignalValues[key] = answer.Score
		results.SignalConfidences[key] = maxProbability(answer.Probabilities)
		if predicateMatches(answer.Score, predicate) {
			return []string{rule.Name}
		}
	case config.SystemOneQuestionSet, config.SystemOneQuestionSpan:
		return applyLabelledAnswer(rule, answer, predicate, key, publish)
	default:
		return applyChoiceAnswer(rule, answer, predicate, key, publish)
	}
	return nil
}

// applyChoiceAnswer publishes each option's probability, and under the rule
// itself the highest of its labels' (the chosen option's for a rule without
// labels), and returns the matching options. With two options and a
// predicate, the predicate is an operating point: an option matches when its
// own probability satisfies it, whichever option the model chose. Otherwise
// the chosen option matches, when the predicate (a confidence floor) holds
// for its probability. The rule matches when a matching option is one of its
// labels.
func applyChoiceAnswer(rule config.SystemOneRule, answer modelservice.Answer, predicate *config.NumericPredicate, key string, publish func(string, float64)) []string {
	for option, probability := range answer.Probabilities {
		publish(key+":"+option, probability)
	}
	best := answer.Probabilities[answer.Choice]
	if len(rule.Labels) > 0 {
		best = 0
		for _, label := range rule.Labels {
			best = max(best, answer.Probabilities[label])
		}
	}
	publish(key, best)
	var matched []string
	if predicate != nil && len(rule.Question.Choices) == 2 {
		for _, option := range rule.Question.OptionKeys() {
			if predicateMatches(answer.Probabilities[option], predicate) {
				matched = append(matched, rule.Name+":"+option)
			}
		}
	} else {
		chosen := answer.Probabilities[answer.Choice]
		if answer.Choice == "" || (predicate != nil && !predicateMatches(chosen, predicate)) {
			return nil
		}
		matched = []string{rule.Name + ":" + answer.Choice}
	}
	for _, reference := range matched {
		if slices.Contains(rule.Labels, strings.TrimPrefix(reference, rule.Name+":")) {
			return append(matched, rule.Name)
		}
	}
	return matched
}

// applyLabelledAnswer publishes each Set label's probability, or each Span
// label's most probable span (0 when the model found none), and the highest
// of the counted labels under the rule itself. A label matches when its
// probability (a Set) or one of its spans' (a Span) meets the predicate, or
// without one when the model selected it or found a span of it; an ignored
// label never does. The rule matches when a counted label does.
func applyLabelledAnswer(rule config.SystemOneRule, answer modelservice.Answer, predicate *config.NumericPredicate, key string, publish func(string, float64)) []string {
	counted := rule.CountedLabels()
	var matched []string
	best, matchesItself := 0.0, false
	for _, label := range rule.Question.Labels {
		probability, found := labelEvidence(rule.Question.Type, answer, label.Key, predicate)
		publish(key+":"+label.Key, probability)
		if !slices.Contains(counted, label.Key) {
			continue
		}
		best = max(best, probability)
		if found {
			matched = append(matched, rule.Name+":"+label.Key)
			matchesItself = true
		}
	}
	publish(key, best)
	if matchesItself {
		matched = append([]string{rule.Name}, matched...)
	}
	return matched
}

// labelEvidence is one label's published probability and whether it
// matches.
func labelEvidence(questionType string, answer modelservice.Answer, label string, predicate *config.NumericPredicate) (float64, bool) {
	if questionType == config.SystemOneQuestionSet {
		probability := answer.Probabilities[label]
		if predicate == nil {
			return probability, slices.Contains(answer.Selected, label)
		}
		return probability, predicateMatches(probability, predicate)
	}
	best, found := 0.0, false
	for _, span := range answer.Spans {
		if span.Label != label {
			continue
		}
		best = max(best, span.Probability)
		found = found || predicate == nil || predicateMatches(span.Probability, predicate)
	}
	return best, found
}

// matchedSpans are the spans of an answer a span rule matches: those of its
// counted labels that satisfy its predicate, or all of them without one.
func matchedSpans(rule config.SystemOneRule, answer modelservice.Answer) []modelservice.Span {
	counted := rule.CountedLabels()
	predicate := rule.EffectivePredicate()
	var spans []modelservice.Span
	for _, span := range answer.Spans {
		if slices.Contains(counted, span.Label) && (predicate == nil || predicateMatches(span.Probability, predicate)) {
			spans = append(spans, span)
		}
	}
	return spans
}

func maxProbability(probabilities map[string]float64) float64 {
	best := 0.0
	for _, probability := range probabilities {
		if probability > best {
			best = probability
		}
	}
	return best
}
