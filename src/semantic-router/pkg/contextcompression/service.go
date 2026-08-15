package contextcompression

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
	"sort"
	"strings"
	"sync/atomic"
	"time"
)

const RetrieveToolName = "vsr_context_retrieve"

type Service struct {
	requests       atomic.Int64
	applied        atomic.Int64
	skipped        atomic.Int64
	failures       atomic.Int64
	recoveryWrites atomic.Int64
	recoveryReads  atomic.Int64
	recoveryErrors atomic.Int64
	tokensBefore   atomic.Int64
	tokensAfter    atomic.Int64
	costSavedBits  atomic.Uint64
}

func NewService() *Service {
	return &Service{}
}

func (s *Service) Capabilities(recoveryAvailable bool) Capabilities {
	return Capabilities{
		Strategies:             []string{"extractive", "recoverable"},
		ScoringMethods:         []string{"bm25", "embedding", "hybrid"},
		Targets:                []string{"tool_output", "history", "rag", "memory"},
		Recovery:               recoveryAvailable,
		JSONStructurePreserved: true,
		MultimodalPreserved:    true,
		ProtocolMetadataKept:   true,
	}
}

func (s *Service) Apply(ctx context.Context, request Request) ServiceResult {
	s.requests.Add(1)
	if request.Request == nil {
		s.failures.Add(1)
		return ServiceResult{Failure: fmt.Errorf("compression request IR is nil")}
	}
	counter := tokenCounterForRequest(request)
	plan, candidates, err := s.plan(ctx, request, counter)
	if err != nil {
		s.failures.Add(1)
		return s.failureResult(request, plan, err)
	}
	if plan.SkipReason != SkipNone {
		return s.skippedPlanResult(request, plan)
	}
	result, appliedCandidates, err := s.executePlan(
		ctx,
		request,
		counter,
		plan,
		candidates,
	)
	if err != nil {
		s.failures.Add(1)
		return s.failureResult(request, plan, err)
	}
	return s.finalizeResult(request, counter, result, appliedCandidates)
}

func tokenCounterForRequest(request Request) TokenCounter {
	if request.TokenCounter != nil {
		return request.TokenCounter
	}
	return HeuristicTokenCounter{}
}

func (s *Service) skippedPlanResult(
	request Request,
	plan Plan,
) ServiceResult {
	s.skipped.Add(1)
	s.tokensBefore.Add(int64(plan.OriginalTokens))
	s.tokensAfter.Add(int64(plan.OriginalTokens))
	return ServiceResult{
		Request:      request.Request,
		Plan:         plan,
		TokensBefore: plan.OriginalTokens,
		TokensAfter:  plan.OriginalTokens,
	}
}

func (s *Service) executePlan(
	ctx context.Context,
	request Request,
	counter TokenCounter,
	plan Plan,
	candidates []plannedCandidate,
) (ServiceResult, []plannedCandidate, error) {
	result := ServiceResult{
		Request:      request.Request,
		Plan:         plan,
		TokensBefore: plan.OriginalTokens,
	}
	recoveryBytes := 0
	appliedMessages := make(map[int]struct{})
	appliedCandidates := make([]plannedCandidate, 0, len(candidates))
	for _, candidate := range candidates {
		applied, blockResult, recoveryKey, applyErr := s.applyCandidate(
			ctx,
			request,
			counter,
			candidate,
			&recoveryBytes,
		)
		if applyErr != nil {
			return result, appliedCandidates, applyErr
		}
		if !applied {
			continue
		}
		result.Applied = true
		result.BlocksCompressed++
		appliedCandidates = append(appliedCandidates, candidate)
		appliedMessages[candidate.plan.MessageIndex] = struct{}{}
		result.OmittedChunks += blockResult.OmittedChunks
		if blockResult.Format == "json" {
			result.JSONBlocks++
		}
		if recoveryKey != "" {
			s.recoveryWrites.Add(1)
			result.RecoveryKeys = append(result.RecoveryKeys, recoveryKey)
		}
	}
	result.MessagesCompressed = len(appliedMessages)
	return result, appliedCandidates, nil
}

func (s *Service) finalizeResult(
	request Request,
	counter TokenCounter,
	result ServiceResult,
	appliedCandidates []plannedCandidate,
) ServiceResult {
	result.TokensAfter, _ = counter.CountRequest(request.Model, request.Request)
	if result.TokensAfter <= 0 {
		result.TokensAfter = result.Plan.OriginalTokens
	}
	if result.Applied && result.TokensAfter >= result.TokensBefore {
		for _, candidate := range appliedCandidates {
			candidate.block.SetText(candidate.originalText)
		}
		result.Applied = false
		result.BlocksCompressed = 0
		result.MessagesCompressed = 0
		result.OmittedChunks = 0
		result.JSONBlocks = 0
		result.RecoveryKeys = nil
		result.TokensAfter = result.TokensBefore
		result.Plan.SkipReason = SkipBudgetNotReduced
		result.Plan.Quality = "rejected_non_reducing"
	}
	if !result.Applied {
		s.skipped.Add(1)
		result.Plan.SkipReason = SkipBudgetNotReduced
	} else {
		s.applied.Add(1)
		if result.Plan.TargetRequestTokens > 0 &&
			result.TokensAfter > result.Plan.TargetRequestTokens {
			result.Plan.Quality = "budget_partial"
		} else {
			result.Plan.Quality = "budget_met"
		}
	}
	s.tokensBefore.Add(int64(result.TokensBefore))
	s.tokensAfter.Add(int64(result.TokensAfter))
	return result
}

func (s *Service) Stats() Stats {
	if s == nil {
		return Stats{}
	}
	before := s.tokensBefore.Load()
	after := s.tokensAfter.Load()
	return Stats{
		Requests:              s.requests.Load(),
		Applied:               s.applied.Load(),
		Skipped:               s.skipped.Load(),
		Failures:              s.failures.Load(),
		RecoveryWrites:        s.recoveryWrites.Load(),
		RecoveryReads:         s.recoveryReads.Load(),
		RecoveryFailures:      s.recoveryErrors.Load(),
		TokensBefore:          before,
		TokensAfter:           after,
		TokensSaved:           max(int64(0), before-after),
		EstimatedCostSavedUSD: math.Float64frombits(s.costSavedBits.Load()),
	}
}

func (s *Service) RecordRecoveryRead() {
	if s != nil {
		s.recoveryReads.Add(1)
	}
}

func (s *Service) RecordRecoveryFailure() {
	if s != nil {
		s.recoveryErrors.Add(1)
	}
}

func (s *Service) RecordEstimatedCostSavings(amount float64) {
	if s == nil || amount <= 0 {
		return
	}
	for {
		currentBits := s.costSavedBits.Load()
		current := math.Float64frombits(currentBits)
		if s.costSavedBits.CompareAndSwap(
			currentBits,
			math.Float64bits(current+amount),
		) {
			return
		}
	}
}

type plannedCandidate struct {
	plan         TargetPlan
	block        *TextBlockIR
	originalText string
}

func (s *Service) plan(
	ctx context.Context,
	request Request,
	counter TokenCounter,
) (Plan, []plannedCandidate, error) {
	originalTokens, source := counter.CountRequest(request.Model, request.Request)
	available, reserve := availableInputBudget(
		request.Capabilities,
		request.Policy.Budget,
		originalTokens,
	)
	targetRequest := resolveTargetRequestTokens(
		request.Policy.Budget,
		available,
		originalTokens,
	)
	plan := Plan{
		Model:                request.Model,
		OriginalTokens:       originalTokens,
		TargetRequestTokens:  targetRequest,
		AvailableInputTokens: available,
		ReservedOutputTokens: reserve,
		TokenCounterSource:   source,
	}
	candidates := collectCandidates(request, counter)
	if len(candidates) == 0 {
		plan.SkipReason = SkipNoTargets
		return plan, nil, nil
	}
	if shouldSkipForTrigger(request.Policy, originalTokens, available, candidates) {
		plan.SkipReason = SkipBelowTrigger
		return plan, nil, nil
	}
	if request.Policy.Mode == ModeAlways {
		plan.TriggerReason = "always"
	} else if available > 0 && originalTokens > available {
		plan.TriggerReason = "context_window"
	} else {
		plan.TriggerReason = "item_threshold"
	}
	if err := scoreCandidates(ctx, request, candidates); err != nil {
		if request.Policy.Scoring.Method == "embedding" {
			plan.SkipReason = SkipScoringUnavailable
			return plan, nil, err
		}
		plan.FallbackReason = "embedding_unavailable"
	}
	sort.SliceStable(candidates, func(left, right int) bool {
		if candidates[left].plan.Score == candidates[right].plan.Score {
			return candidates[left].plan.OriginalTokens > candidates[right].plan.OriginalTokens
		}
		return candidates[left].plan.Score < candidates[right].plan.Score
	})
	allocateGlobalBudget(candidates, originalTokens, targetRequest)
	plan.Targets = make([]TargetPlan, 0, len(candidates))
	for _, candidate := range candidates {
		plan.Targets = append(plan.Targets, candidate.plan)
	}
	return plan, candidates, nil
}

func collectCandidates(
	request Request,
	counter TokenCounter,
) []plannedCandidate {
	result := make([]plannedCandidate, 0)
	for _, message := range request.Request.Messages {
		for _, block := range message.Blocks {
			candidate, ok := candidateForBlock(request, counter, message, block)
			if ok {
				result = append(result, candidate)
			}
		}
	}
	return result
}

func candidateForBlock(
	request Request,
	counter TokenCounter,
	message *MessageIR,
	block *TextBlockIR,
) (plannedCandidate, bool) {
	kind := block.Source
	if kind == TargetHistory &&
		(message.Protected || message.Role == "tool" || message.Role == "function") {
		return plannedCandidate{}, false
	}
	policy := policyForTarget(request.Policy.Targets, kind)
	if policy.Mode == TargetPreserve {
		return plannedCandidate{}, false
	}
	tokens, _ := counter.CountText(request.Model, block.Text)
	minTokens := policy.MinTokens
	if minTokens <= 0 {
		minTokens = 2000
	}
	if tokens < minTokens && request.Policy.Mode != ModeAlways {
		return plannedCandidate{}, false
	}
	targetTokens := policy.TargetTokens
	if targetTokens <= 0 {
		targetTokens = max(1, minTokens/2)
	}
	query := request.Request.QueryFor(block)
	return plannedCandidate{
		block:        block,
		originalText: block.Text,
		plan: TargetPlan{
			MessageIndex:   block.MessageIndex,
			BlockIndex:     block.BlockIndex,
			Kind:           kind,
			Mode:           policy.Mode,
			OriginalTokens: tokens,
			TargetTokens:   targetTokens,
			Query:          query,
			Score:          lexicalScore(query, block.Text),
		},
	}, true
}

func scoreCandidates(
	ctx context.Context,
	request Request,
	candidates []plannedCandidate,
) error {
	if request.Policy.Scoring.Method == "" || request.Policy.Scoring.Method == "bm25" {
		return nil
	}
	if request.Scorer == nil {
		return fmt.Errorf("embedding scorer is unavailable")
	}
	texts := make([]string, len(candidates))
	queryParts := make([]string, 0, len(candidates))
	for index, candidate := range candidates {
		texts[index] = candidate.block.Text
		queryParts = append(queryParts, candidate.plan.Query)
	}
	scores, err := request.Scorer.Score(
		ctx,
		request.Policy.Scoring.EmbeddingModelRef,
		strings.Join(queryParts, "\n"),
		texts,
	)
	if err != nil {
		return err
	}
	if len(scores) != len(candidates) {
		return fmt.Errorf("embedding scorer returned %d scores for %d targets", len(scores), len(candidates))
	}
	for index, score := range scores {
		switch request.Policy.Scoring.Method {
		case "embedding":
			candidates[index].plan.Score = score
		case "hybrid":
			candidates[index].plan.Score = 0.4*candidates[index].plan.Score + 0.6*score
		}
	}
	return nil
}

func (s *Service) applyCandidate(
	ctx context.Context,
	request Request,
	counter TokenCounter,
	candidate plannedCandidate,
	recoveryBytes *int,
) (bool, Result, string, error) {
	block := candidate.block
	compressed := CompressToolOutput(
		block.Text,
		candidate.plan.Query,
		candidate.plan.OriginalTokens,
		candidate.plan.TargetTokens,
	)
	if candidate.plan.Mode == TargetExtractive {
		if !compressed.Applied {
			return false, compressed, "", nil
		}
		block.SetText(compressed.Content)
		return true, compressed, "", nil
	}
	if candidate.plan.Mode != TargetRecoverable {
		return false, compressed, "", nil
	}
	return applyRecoverableCandidate(
		ctx,
		request,
		counter,
		candidate,
		compressed,
		recoveryBytes,
	)
}

func applyRecoverableCandidate(
	ctx context.Context,
	request Request,
	counter TokenCounter,
	candidate plannedCandidate,
	compressed Result,
	recoveryBytes *int,
) (bool, Result, string, error) {
	block := candidate.block
	if block.JSON && !compressed.Applied {
		return false, compressed, "", nil
	}
	if request.Recovery == nil || !request.Policy.Recovery.Enabled {
		return false, compressed, "", fmt.Errorf("recovery store is unavailable")
	}
	originalBytes := len([]byte(block.Text))
	if request.Policy.Recovery.MaxBytesPerRequest > 0 &&
		*recoveryBytes+originalBytes > request.Policy.Recovery.MaxBytesPerRequest {
		return false, compressed, "", fmt.Errorf("recovery request byte limit exceeded")
	}
	key := recoveryKey(request.Scope, block.Text)
	ttl := request.Policy.Recovery.TTL
	if ttl <= 0 {
		ttl = 15 * time.Minute
	}
	now := time.Now().UTC()
	entry := RecoveryEntry{
		Key:       key,
		Scope:     request.Scope,
		Content:   block.Text,
		CreatedAt: now,
		ExpiresAt: now.Add(ttl),
	}
	if err := request.Recovery.Put(ctx, entry, ttl); err != nil {
		return false, compressed, "", fmt.Errorf("store recovery content: %w", err)
	}
	*recoveryBytes += originalBytes
	body := compressed.Content
	if !compressed.Applied {
		body = "[Context compressed; retrieve the original if needed.]"
		compressed = Result{
			Content:          body,
			OriginalTokens:   candidate.plan.OriginalTokens,
			CompressedTokens: EstimateTokens(body),
			Applied:          true,
			Format:           "text",
			OmittedChunks:    1,
		}
	}
	if !block.JSON {
		body += fmt.Sprintf(
			"\n[Recovery key=%s. Call %s with this key when full context is required.]",
			key,
			RetrieveToolName,
		)
	}
	block.SetText(body)
	compressed.Content = body
	compressed.CompressedTokens, _ = counter.CountText(request.Model, body)
	return true, compressed, key, nil
}

func (s *Service) failureResult(
	request Request,
	plan Plan,
	err error,
) ServiceResult {
	result := ServiceResult{
		Request:      request.Request,
		Plan:         plan,
		TokensBefore: plan.OriginalTokens,
		TokensAfter:  plan.OriginalTokens,
		Failure:      err,
	}
	return result
}

func policyForTarget(targets Targets, kind TargetKind) TargetPolicy {
	switch kind {
	case TargetRAG:
		return targets.RAG
	case TargetMemory:
		return targets.Memory
	case TargetHistory:
		return targets.History
	default:
		return targets.ToolOutputs
	}
}

func availableInputBudget(
	capabilities ModelContextCapabilities,
	budget Budget,
	originalTokens int,
) (int, int) {
	reserve := budget.ReserveOutputTokens
	if budget.ReserveAuto || reserve <= 0 {
		reserve = capabilities.RequestedOutput
		if reserve <= 0 {
			reserve = capabilities.MaxOutputTokens
		}
	}
	if capabilities.ContextWindow <= 0 {
		return originalTokens, max(0, reserve)
	}
	return max(0, capabilities.ContextWindow-reserve), max(0, reserve)
}

func resolveTargetRequestTokens(
	budget Budget,
	available int,
	original int,
) int {
	if !budget.TargetAuto && budget.TargetTokens > 0 {
		return min(budget.TargetTokens, max(1, available))
	}
	if available > 0 {
		return min(original, available)
	}
	return original
}

func shouldSkipForTrigger(
	policy Policy,
	originalTokens int,
	available int,
	candidates []plannedCandidate,
) bool {
	if policy.Mode == ModeAlways {
		return false
	}
	if !policy.Budget.TriggerAuto && policy.Budget.TriggerTokens > 0 {
		return originalTokens <= policy.Budget.TriggerTokens
	}
	if available > 0 && originalTokens > int(float64(available)*0.85) {
		return false
	}
	return len(candidates) == 0
}

func allocateGlobalBudget(
	candidates []plannedCandidate,
	originalTokens int,
	targetRequest int,
) {
	needed := max(0, originalTokens-targetRequest)
	for index := range candidates {
		if needed <= 0 {
			break
		}
		potential := max(0, candidates[index].plan.OriginalTokens-candidates[index].plan.TargetTokens)
		if potential > needed {
			candidates[index].plan.TargetTokens = max(
				candidates[index].plan.TargetTokens,
				candidates[index].plan.OriginalTokens-needed,
			)
			needed = 0
			continue
		}
		needed -= potential
	}
}

func lexicalScore(query string, text string) float64 {
	queryTerms := tokenizeTerms(query)
	textTerms := tokenizeTerms(text)
	if len(queryTerms) == 0 || len(textTerms) == 0 {
		return 0
	}
	frequency := make(map[string]int, len(textTerms))
	for _, term := range textTerms {
		frequency[term]++
	}
	matches := 0
	for _, term := range queryTerms {
		if frequency[term] > 0 {
			matches++
		}
	}
	return float64(matches) / float64(len(queryTerms))
}

func recoveryKey(scope string, content string) string {
	sum := sha256.Sum256([]byte(scope + "\x00" + content))
	return hex.EncodeToString(sum[:12])
}
