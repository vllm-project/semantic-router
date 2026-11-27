package turnselection

import (
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
)

// Turn is a protocol-neutral, read-only view of one complete conversation turn
// grouped from contextcompression.TransformationView.
type Turn struct {
	TurnID         int
	Ordinal        int
	FirstMessageID int
	MessageIDs     []int
	Messages       []contextcompression.MessageView
	ExchangeIDs    []string
	IsLive         bool
	Eligible       bool
	PinReason      string
	Tokens         int
	Text           string
}

// GroupTurns organizes detached MessageView entries into ordered complete turns
// using the shared TurnID, ContentSource, Protection, Eligibility, and
// ExchangeIDs metadata.
//
// Turn eligibility is evaluated at the whole-turn level: a turn is eligible for
// removal only when every message in the turn has SourceHistory, zero
// Protection bits, EligibleHistoryRemoval set, a non-negative TurnID, and no
// tool exchange shared with another turn. Cross-turn tool exchanges are pinned
// conservatively so tool call/result atomicity is never broken.
func GroupTurns(view contextcompression.TransformationView, maxTurnTextBytes int) []Turn {
	if len(view.Messages) == 0 {
		return nil
	}
	if maxTurnTextBytes <= 0 {
		maxTurnTextBytes = DefaultMaxTurnTextBytes
	}

	turns := make([]Turn, 0)
	indexByTurnID := make(map[int]int)
	exchangeTurns := make(map[string]map[int]struct{})

	for _, message := range view.Messages {
		turnPos, exists := indexByTurnID[message.TurnID]
		if !exists {
			turnPos = len(turns)
			indexByTurnID[message.TurnID] = turnPos
			turns = append(turns, Turn{
				TurnID:         message.TurnID,
				Ordinal:        turnPos,
				FirstMessageID: message.ID,
				Eligible:       message.TurnID >= 0,
			})
		}
		turn := &turns[turnPos]
		turn.MessageIDs = append(turn.MessageIDs, message.ID)
		turn.Messages = append(turn.Messages, message)
		if message.Protection&contextcompression.ProtectLiveTurn != 0 {
			turn.IsLive = true
		}
		for _, exchangeID := range message.ExchangeIDs {
			if exchangeID == "" {
				continue
			}
			members := exchangeTurns[exchangeID]
			if members == nil {
				members = make(map[int]struct{})
				exchangeTurns[exchangeID] = members
			}
			members[message.TurnID] = struct{}{}
		}
	}

	crossTurnIDs := make(map[int]bool)
	for _, members := range exchangeTurns {
		if len(members) > 1 {
			for turnID := range members {
				crossTurnIDs[turnID] = true
			}
		}
	}

	for idx := range turns {
		turn := &turns[idx]
		turn.ExchangeIDs = uniqueTurnExchanges(turn.Messages)
		turn.Eligible, turn.PinReason = classifyTurnEligibility(*turn, crossTurnIDs[turn.TurnID])
		turn.Text = buildBoundedTurnText(turn.Messages, maxTurnTextBytes)
	}
	return turns
}

func uniqueTurnExchanges(messages []contextcompression.MessageView) []string {
	seen := make(map[string]struct{})
	for _, msg := range messages {
		for _, id := range msg.ExchangeIDs {
			if id != "" {
				seen[id] = struct{}{}
			}
		}
	}
	if len(seen) == 0 {
		return nil
	}
	ids := make([]string, 0, len(seen))
	for id := range seen {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}

func classifyTurnEligibility(turn Turn, hasCrossTurnExchange bool) (bool, string) {
	if turn.TurnID < 0 {
		return false, PinUnknown
	}
	var combinedProtection contextcompression.Protection
	hasNonHistorySource := false
	hasIneligibleMessage := false
	for _, msg := range turn.Messages {
		combinedProtection |= msg.Protection
		if msg.Source != contextcompression.SourceHistory {
			hasNonHistorySource = true
		}
		if msg.Eligibility&contextcompression.EligibleHistoryRemoval == 0 {
			hasIneligibleMessage = true
		}
	}
	if reason := primaryProtectionPinReason(combinedProtection); reason != "" {
		return false, reason
	}
	if hasNonHistorySource {
		return false, PinNonHistorySource
	}
	if hasCrossTurnExchange {
		return false, PinCrossTurnExchange
	}
	if hasIneligibleMessage {
		return false, PinUnknown
	}
	return true, ""
}

func primaryProtectionPinReason(protection contextcompression.Protection) string {
	switch {
	case protection&contextcompression.ProtectLiveTurn != 0:
		return PinLiveTurn
	case protection&contextcompression.ProtectInstructions != 0:
		return PinInstructions
	case protection&contextcompression.ProtectAuthorization != 0:
		return PinAuthorization
	case protection&contextcompression.ProtectSafety != 0:
		return PinSafety
	case protection&contextcompression.ProtectMultimodal != 0:
		return PinMultimodal
	case protection&contextcompression.ProtectContinuation != 0:
		return PinContinuation
	case protection&contextcompression.ProtectUnknown != 0:
		return PinUnknown
	default:
		return ""
	}
}

func buildBoundedTurnText(messages []contextcompression.MessageView, maxBytes int) string {
	if maxBytes <= 0 {
		return ""
	}
	var b strings.Builder
	for _, msg := range messages {
		for _, block := range msg.Blocks {
			if block.Source == contextcompression.TargetRAG || block.Source == contextcompression.TargetMemory {
				continue
			}
			trimmed := strings.TrimSpace(block.Text)
			if trimmed == "" {
				continue
			}
			if b.Len() > 0 {
				if b.Len()+1 >= maxBytes {
					return b.String()
				}
				b.WriteByte('\n')
			}
			remaining := maxBytes - b.Len()
			if remaining <= 0 {
				return b.String()
			}
			b.WriteString(truncateUTF8(trimmed, remaining))
			if b.Len() >= maxBytes {
				return b.String()
			}
		}
	}
	return b.String()
}

// extractLiveQuery returns the bounded query text from the live turn without
// mixing injected RAG/Memory material or removed upstream history.
func extractLiveQuery(turns []Turn, maxQueryBytes int) string {
	if len(turns) == 0 || maxQueryBytes <= 0 {
		return ""
	}
	liveIndex := -1
	for idx := len(turns) - 1; idx >= 0; idx-- {
		if turns[idx].IsLive {
			liveIndex = idx
			break
		}
	}
	if liveIndex < 0 {
		liveIndex = len(turns) - 1
	}
	live := turns[liveIndex]

	// Prefer explicit user-authored history text from the live turn.
	var userMessages []contextcompression.MessageView
	for _, msg := range live.Messages {
		if msg.Role == "user" && msg.Source == contextcompression.SourceHistory {
			userMessages = append(userMessages, msg)
		}
	}
	if query := buildBoundedTurnText(userMessages, maxQueryBytes); query != "" {
		return query
	}

	// Tool-continuation live turns may not contain a new user text block;
	// fall back to history text present in the live turn itself.
	var historyMessages []contextcompression.MessageView
	for _, msg := range live.Messages {
		if msg.Source == contextcompression.SourceHistory {
			historyMessages = append(historyMessages, msg)
		}
	}
	return buildBoundedTurnText(historyMessages, maxQueryBytes)
}

func truncateUTF8(text string, maxBytes int) string {
	if maxBytes <= 0 || text == "" {
		return ""
	}
	if len(text) <= maxBytes {
		return text
	}
	cut := maxBytes
	for cut > 0 && !utf8.RuneStart(text[cut]) {
		cut--
	}
	return text[:cut]
}
