package topiccontinuity

import (
	"context"
	"strings"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func isToolResultMessage(message llmprotocol.Message) bool {
	if message.Role == llmprotocol.RoleTool {
		return true
	}
	if message.Role != llmprotocol.RoleUser || len(message.Content) == 0 {
		return false
	}
	return !llmprotocol.StartsConversationTurn(message)
}

func prepare(ctx context.Context, messages []llmprotocol.Message, available bool, policy HistoryPolicy) preparation {
	base := preparation{
		Policy: policy, Coverage: CoveragePartial,
		Scope: EvidenceScope{AssistantIncluded: policy.IncludeAssistant},
	}
	if !available {
		return terminal(base, ReasonHistoryUnavailable)
	}
	state := &prepareState{ctx: ctx, messages: messages, policy: policy, out: base}
	return state.run()
}

func terminal(base preparation, status Reason) preparation {
	base.Status = status
	base.Coverage = CoveragePartial
	return base
}

type prepareState struct {
	ctx      context.Context
	messages []llmprotocol.Message
	policy   HistoryPolicy
	out      preparation

	scanned       int
	blocks        int
	scanReached   bool
	blocksReached bool
}

// visit counts one message and its blocks. It returns false once a cap is
// reached; reaching a cap exactly counts as reaching it.
func (s *prepareState) visit(message llmprotocol.Message) bool {
	s.scanned++
	s.blocks += len(message.Content)
	if s.blocks >= maxContentBlocksPerPrepare {
		s.blocksReached = true
	}
	if s.scanned >= maxScanMessages {
		s.scanReached = true
	}
	return !s.scanReached && !s.blocksReached
}

func (s *prepareState) liveCapStatus() preparation {
	out := terminal(s.out, ReasonOverBudget)
	out.Scope.FeatureCapReached = s.blocksReached
	return out
}

func (s *prepareState) run() preparation {
	if len(s.messages) == 0 {
		return terminal(s.out, ReasonNoLiveUserTurn)
	}
	last := len(s.messages) - 1
	liveStart, kind, status := s.classifyLive(last)
	if status != "" {
		if status == ReasonOverBudget {
			return s.liveCapStatus()
		}
		out := terminal(s.out, status)
		return out
	}
	s.out.liveKind = kind
	return s.collect(liveStart, last)
}

func (s *prepareState) classifyLive(last int) (int, liveKind, Reason) {
	final := s.messages[last]
	if err := s.ctx.Err(); err != nil {
		return 0, liveNone, ReasonCancelled
	}
	if !s.visit(final) {
		return 0, liveNone, ReasonOverBudget
	}
	switch {
	case llmprotocol.StartsConversationTurn(final):
		return last, liveUser, ""
	case isToolResultMessage(final):
		return s.classifyToolContinuation(last)
	default:
		return 0, liveNone, ReasonNoLiveUserTurn
	}
}

// classifyToolContinuation walks back to the live turn's start and requires
// every message in between to be an assistant message or a tool result, and
// every tool result to answer a call made inside that span.
func (s *prepareState) classifyToolContinuation(last int) (int, liveKind, Reason) {
	calls := make(map[string]struct{})
	var results []string
	collect := func(message llmprotocol.Message) {
		for _, content := range message.Content {
			if content.Kind == llmprotocol.ContentToolCall && content.ToolCall != nil {
				calls[content.ToolCall.ID] = struct{}{}
			}
			if content.Kind == llmprotocol.ContentToolResult && content.ToolResult != nil {
				results = append(results, content.ToolResult.CallID)
			}
		}
	}
	collect(s.messages[last])
	for index := last - 1; index >= 0; index-- {
		if err := s.ctx.Err(); err != nil {
			return 0, liveNone, ReasonCancelled
		}
		message := s.messages[index]
		if !s.visit(message) {
			return 0, liveNone, ReasonOverBudget
		}
		if llmprotocol.StartsConversationTurn(message) {
			for _, id := range results {
				if _, ok := calls[id]; !ok || id == "" {
					return 0, liveNone, ReasonOrphanToolResult
				}
			}
			if len(results) == 0 {
				return 0, liveNone, ReasonOrphanToolResult
			}
			return index, liveToolContinuation, ""
		}
		if message.Role != llmprotocol.RoleAssistant && !isToolResultMessage(message) {
			return 0, liveNone, ReasonOrphanToolResult
		}
		collect(message)
	}
	return 0, liveNone, ReasonOrphanToolResult
}

func (s *prepareState) collect(liveStart, last int) preparation {
	limits := s.policy.Limits
	live := s.buildTurn(s.messages[liveStart:last+1], true)
	s.markExcluded(s.messages[liveStart : last+1])
	s.out.Live = live
	s.out.InputBytes = turnBytes(live)
	partial := live.Truncated
	if s.out.liveKind == liveUser {
		s.out.LiveOpaqueOnly = opaqueOnly(s.messages[liveStart], live)
	}

	var pending []llmprotocol.Message
	foundPrior, beyond := false, false
	for index := liveStart - 1; index >= 0; index-- {
		if err := s.ctx.Err(); err != nil {
			return s.cancelled()
		}
		message := s.messages[index]
		if !s.visit(message) {
			partial = true
			break
		}
		pending = append(pending, message)
		if !llmprotocol.StartsConversationTurn(message) {
			continue
		}
		foundPrior = true
		if len(s.out.Prior) >= limits.MaxPriorTurns {
			// A complete turn exists past the window: out of scope by design.
			beyond = true
			break
		}
		turnMessages := reverse(pending)
		pending = nil
		// Check the aggregate budget before reading any of the turn's text: a
		// turn that may not fit is dropped, with everything older, unread.
		if s.out.InputBytes+s.retainedUpperBound(turnMessages) > limits.MaxInputBytes {
			partial = true
			break
		}
		turn := s.buildTurn(turnMessages, false)
		s.markExcluded(turnMessages)
		s.out.Prior = append(s.out.Prior, turn)
		s.out.InputBytes += turnBytes(turn)
		if turn.Truncated {
			partial = true
		}
		if turn.toolCallCapReached {
			s.out.Scope.FeatureCapReached = true
		}
	}
	if s.blocksReached {
		s.out.Scope.FeatureCapReached = true
	}
	switch {
	case partial || s.out.Scope.FeatureCapReached:
		s.out.Coverage = CoveragePartial
	case beyond:
		s.out.Coverage = CoverageWindow
	default:
		s.out.Coverage = CoverageFull
	}
	if !foundPrior && !s.scanReached && !s.blocksReached {
		s.out.Status = ReasonNoPriorTurn
	}
	return s.out
}

func (s *prepareState) cancelled() preparation {
	out := s.out
	out.Status = ReasonCancelled
	out.Coverage = CoveragePartial
	out.Scope.FeatureCapReached = out.Scope.FeatureCapReached || s.blocksReached
	return out
}

// markExcluded records whether examined messages hold content outside the
// evidence policy: tool arguments or results, reasoning, refusals, or media.
func (s *prepareState) markExcluded(messages []llmprotocol.Message) {
	for _, message := range messages {
		for _, content := range message.Content {
			if excludedContent(content) {
				s.out.Scope.ExcludedContentPresent = true
				return
			}
		}
	}
}

func excludedContent(content llmprotocol.Content) bool {
	switch content.Kind {
	case llmprotocol.ContentText:
		return false
	case llmprotocol.ContentToolCall:
		return content.ToolCall != nil && content.ToolCall.Arguments != ""
	default:
		return true
	}
}

func opaqueKind(kind llmprotocol.ContentKind) bool {
	switch kind {
	case llmprotocol.ContentImage, llmprotocol.ContentAudio, llmprotocol.ContentVideo, llmprotocol.ContentFile:
		return true
	}
	return false
}

// opaqueOnly is decided from retained segments only, so it never scans text
// beyond the per-turn budget. A truncated live turn is never opaque-only.
func opaqueOnly(message llmprotocol.Message, live evidenceTurn) bool {
	if live.Truncated || !live.HasOpaque {
		return false
	}
	for _, content := range message.Content {
		if opaqueKind(content.Kind) {
			for _, segment := range live.User {
				if strings.TrimSpace(string(segment)) != "" {
					return false
				}
			}
			return true
		}
	}
	return false
}

// retainedUpperBound bounds the bytes buildTurn would retain for a prior turn
// without reading any text: it sums block lengths, which costs one step per
// content block (already bounded by maxContentBlocksPerPrepare), and caps the
// sum at MaxTurnBytes. The bound is conservative: a turn that would fit after
// rune-boundary trimming may be dropped a few bytes early, which only makes
// coverage partial.
func (s *prepareState) retainedUpperBound(messages []llmprotocol.Message) int {
	total, toolCalls := 0, 0
	for _, content := range messages[0].Content {
		if content.Kind == llmprotocol.ContentText {
			total += len(content.Text)
		}
	}
	for _, message := range messages {
		for _, content := range message.Content {
			switch {
			case content.Kind == llmprotocol.ContentToolCall && content.ToolCall != nil:
				if toolCalls < maxToolNamesPerTurn {
					total += min(len(content.ToolCall.Name), maxToolNameBytes)
				}
				toolCalls++
			case content.Kind == llmprotocol.ContentText && message.Role == llmprotocol.RoleAssistant &&
				s.policy.IncludeAssistant:
				total += len(content.Text)
			}
		}
	}
	return min(total, s.policy.Limits.MaxTurnBytes)
}

// buildTurn windows one turn's evidence under the per-turn byte budget. For
// the live turn only the starting user message's text is evidence.
func (s *prepareState) buildTurn(messages []llmprotocol.Message, live bool) evidenceTurn {
	var turn evidenceTurn
	budget := s.policy.Limits.MaxTurnBytes
	for _, message := range messages {
		for _, content := range message.Content {
			if opaqueKind(content.Kind) {
				turn.HasOpaque = true
			}
		}
	}
	for _, content := range messages[0].Content {
		if content.Kind == llmprotocol.ContentText {
			turn.User, budget = appendWindowed(turn.User, content.Text, budget, &turn.Truncated)
		}
	}
	if live {
		return turn
	}
	if s.policy.IncludeAssistant {
		for _, message := range messages[1:] {
			if message.Role != llmprotocol.RoleAssistant {
				continue
			}
			for _, content := range message.Content {
				if content.Kind == llmprotocol.ContentText {
					turn.Assistant, budget = appendWindowed(turn.Assistant, content.Text, budget, &turn.Truncated)
				}
			}
		}
	}
	toolCalls := 0
	for _, message := range messages {
		for _, content := range message.Content {
			if content.Kind != llmprotocol.ContentToolCall || content.ToolCall == nil {
				continue
			}
			toolCalls++
			if toolCalls >= maxToolNamesPerTurn {
				// Reaching the cap exactly counts as reaching it.
				turn.toolCallCapReached = true
			}
			if toolCalls > maxToolNamesPerTurn {
				continue
			}
			name := runePrefix(content.ToolCall.Name, maxToolNameBytes)
			if name == "" {
				continue
			}
			if len(name) > budget {
				turn.Truncated = true
				continue
			}
			turn.ToolNames = append(turn.ToolNames, name)
			budget -= len(name)
		}
	}
	return turn
}

// appendWindowed adds a text block whole when it fits, or as separate head
// and tail segments (each half of the remaining budget, cut on rune starts).
// Segments are never concatenated, so no token can form across the cut.
func appendWindowed(segments []textSegment, text string, budget int, truncated *bool) ([]textSegment, int) {
	if text == "" {
		return segments, budget
	}
	if len(text) <= budget {
		return append(segments, textSegment(text)), budget - len(text)
	}
	*truncated = true
	half := budget / 2
	if half == 0 {
		return segments, budget
	}
	head := runePrefix(text, half)
	tail := runeSuffix(text, half)
	if head != "" {
		segments = append(segments, textSegment(head))
	}
	if tail != "" {
		segments = append(segments, textSegment(tail))
	}
	return segments, budget - len(head) - len(tail)
}

// runePrefix returns the longest prefix of at most limit bytes that ends on a
// rune boundary.
func runePrefix(text string, limit int) string {
	if len(text) <= limit {
		return text
	}
	cut := limit
	for cut > 0 && !utf8.RuneStart(text[cut]) {
		cut--
	}
	return text[:cut]
}

// runeSuffix returns the longest suffix of at most limit bytes that starts on
// a rune boundary.
func runeSuffix(text string, limit int) string {
	if len(text) <= limit {
		return text
	}
	start := len(text) - limit
	for start < len(text) && !utf8.RuneStart(text[start]) {
		start++
	}
	return text[start:]
}

func turnBytes(turn evidenceTurn) int {
	total := 0
	for _, segment := range turn.User {
		total += len(segment)
	}
	for _, segment := range turn.Assistant {
		total += len(segment)
	}
	for _, name := range turn.ToolNames {
		total += len(name)
	}
	return total
}

func reverse(messages []llmprotocol.Message) []llmprotocol.Message {
	out := make([]llmprotocol.Message, len(messages))
	for i, message := range messages {
		out[len(messages)-1-i] = message
	}
	return out
}
