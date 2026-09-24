package contextdedup

import (
	"context"
	"sort"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
)

const cancellationCheckInterval = 64

// turn is one TurnID group in view order with its classification.
type turn struct {
	messages []contextcompression.MessageView
	retained string
	identity turnIdentity
}

func (t *turn) candidate() bool { return t.retained == "" }

func (t *turn) firstID() int { return t.messages[0].ID }

func (t *turn) textBytes() int {
	total := 0
	for _, message := range t.messages {
		for _, block := range message.Blocks {
			total += len(block.Text)
		}
	}
	return total
}

// plan selects the later copies of adjacent repeated turns. It returns the
// removable message IDs sorted ascending and the bounded diagnostics.
func plan(
	ctx context.Context,
	policy Policy,
	resolver Resolver,
	view contextcompression.TransformationView,
) (contextcompression.TransformationEdits, Diagnostics) {
	if ctx.Err() != nil {
		return contextcompression.TransformationEdits{}, skipped(policy, view, ReasonCancelled)
	}
	if !policy.withinLimits(view) {
		return contextcompression.TransformationEdits{}, skipped(policy, view, ReasonHistoryLimitExceeded)
	}
	diagnostics := newDiagnostics(policy, view)
	turns := groupTurns(view, diagnostics.Normalization)
	diagnostics.ExaminedTurns = len(turns)
	for _, item := range turns {
		if item.candidate() {
			diagnostics.CandidateTurns++
		}
	}
	if diagnostics.CandidateTurns == 0 {
		finishRetained(turns, &diagnostics)
		diagnostics.Reason = ReasonNoEligibleHistory
		return contextcompression.TransformationEdits{}, diagnostics
	}
	scan := &segmentScan{policy: policy, resolver: resolver, mode: diagnostics.Normalization}
	for _, run := range candidateRuns(turns) {
		if reason := scan.collapse(ctx, run); reason != "" {
			return contextcompression.TransformationEdits{}, skipped(policy, view, reason)
		}
	}
	countNonAdjacent(turns)
	finishRetained(turns, &diagnostics)
	if len(scan.ids) == 0 {
		diagnostics.Reason = ReasonNoDuplicates
		return contextcompression.TransformationEdits{}, diagnostics
	}
	sort.Ints(scan.ids)
	diagnostics.Outcome, diagnostics.Reason = OutcomeApplied, ReasonApplied
	diagnostics.RemovedMessages = len(scan.ids)
	diagnostics.RetainedMessages = diagnostics.ExaminedMessages - len(scan.ids)
	diagnostics.RemovedTurns = scan.removedTurns
	diagnostics.RemovedTextBytes = scan.removedBytes
	diagnostics.DuplicateSegments = len(scan.segments)
	diagnostics.Segments = scan.segments
	if len(diagnostics.Segments) > MaxReceiptSegments {
		diagnostics.Segments = diagnostics.Segments[:MaxReceiptSegments]
		diagnostics.SegmentsTruncated = true
	}
	return contextcompression.TransformationEdits{RemoveMessages: scan.ids}, diagnostics
}

// groupTurns splits the view into contiguous TurnID groups and classifies
// each against the eligibility rules, in rule order.
func groupTurns(view contextcompression.TransformationView, mode Normalization) []*turn {
	var turns []*turn
	for index, message := range view.Messages {
		if index == 0 || message.TurnID != view.Messages[index-1].TurnID {
			turns = append(turns, &turn{})
		}
		current := turns[len(turns)-1]
		current.messages = append(current.messages, message)
	}
	for _, item := range turns {
		item.retained = classify(item.messages)
		if item.candidate() {
			item.identity = identityOf(item.messages, mode)
		}
	}
	return turns
}

func classify(messages []contextcompression.MessageView) string {
	assistant := false
	for _, message := range messages {
		if message.Eligibility&contextcompression.EligibleHistoryRemoval == 0 ||
			message.Protection != 0 || message.Source != contextcompression.SourceHistory {
			return RetainedIneligible
		}
	}
	for _, message := range messages {
		if len(message.ExchangeIDs) > 0 {
			return RetainedToolExchange
		}
		if message.Role == string(llmRoleAssistant) {
			assistant = true
		}
	}
	if messages[0].Role != string(llmRoleUser) || !assistant {
		return RetainedIncompleteTurn
	}
	for _, message := range messages {
		if len(message.Blocks) == 0 {
			return RetainedOpaqueContent
		}
		for _, block := range message.Blocks {
			if block.Source != contextcompression.TargetHistory {
				return RetainedOpaqueContent
			}
		}
	}
	return ""
}

// candidateRuns returns the maximal runs of consecutive candidate turns.
func candidateRuns(turns []*turn) [][]*turn {
	var runs [][]*turn
	var current []*turn
	for _, item := range turns {
		if item.candidate() {
			current = append(current, item)
			continue
		}
		if len(current) > 1 {
			runs = append(runs, current)
		}
		current = nil
	}
	if len(current) > 1 {
		runs = append(runs, current)
	}
	return runs
}

type segmentScan struct {
	policy       Policy
	resolver     Resolver
	mode         Normalization
	ids          []int
	segments     []Segment
	removedTurns int
	removedBytes int
	checks       int
}

// collapse removes every adjacent repeated block inside one run. It returns
// a terminal reason when the scan cannot finish safely.
func (scan *segmentScan) collapse(ctx context.Context, run []*turn) string {
	for i := 0; i < len(run); {
		removed := false
		for k := min(scan.policy.segmentBound(), (len(run)-i)/2); k >= 1; k-- {
			scan.checks++
			if scan.checks%cancellationCheckInterval == 0 && ctx.Err() != nil {
				return ReasonCancelled
			}
			if !identicalBlocks(run[i:i+k], run[i+k:i+2*k]) {
				continue
			}
			proven, reason := scan.prove(ctx, run[i:i+k], run[i+k:i+2*k])
			if reason == ReasonEquivalenceUnverifiable || reason == ReasonCancelled {
				return reason
			}
			if !proven {
				continue
			}
			scan.remove(run[i], run[i+k:i+2*k])
			run = append(run[:i+k], run[i+2*k:]...)
			// A removal can create a new adjacency for any block whose later
			// copy overlaps the spliced position, so rescan from the earliest
			// start such a block can have.
			i = max(0, i+k-2*scan.policy.segmentBound()+1)
			removed = true
			break
		}
		if !removed {
			i++
		}
	}
	return ""
}

func identicalBlocks(earlier, later []*turn) bool {
	for index := range earlier {
		if !earlier[index].identity.equals(later[index].identity) {
			return false
		}
	}
	return true
}

// prove runs the stage-two check over every message pair of a matched block.
// A failed proof records the retention reason on the later turn whose message
// differed; an unresolvable message or a cancelled context aborts the whole
// step. The context is checked between message pairs because a turn may
// carry far more messages than the scan has positions.
func (scan *segmentScan) prove(ctx context.Context, earlier, later []*turn) (bool, string) {
	if scan.resolver == nil {
		return false, ReasonEquivalenceUnverifiable
	}
	for index := range earlier {
		for position := range earlier[index].messages {
			scan.checks++
			if scan.checks%cancellationCheckInterval == 0 && ctx.Err() != nil {
				return false, ReasonCancelled
			}
			first, ok := scan.resolver(earlier[index].messages[position].ID)
			second, found := scan.resolver(later[index].messages[position].ID)
			if !ok || !found {
				return false, ReasonEquivalenceUnverifiable
			}
			if proven, reason := EquivalentMessages(first, second, scan.mode); !proven {
				if later[index].retained == "" {
					later[index].retained = reason
				}
				return false, reason
			}
		}
	}
	return true, ""
}

func (scan *segmentScan) remove(retained *turn, later []*turn) {
	segment := Segment{RetainedFirstMessageID: retained.firstID(), RemovedFirstMessageID: later[0].firstID(), Turns: len(later)}
	for _, item := range later {
		for _, message := range item.messages {
			scan.ids = append(scan.ids, message.ID)
		}
		segment.Messages += len(item.messages)
		scan.removedBytes += item.textBytes()
		item.retained = removedMarker
	}
	scan.removedTurns += len(later)
	scan.segments = append(scan.segments, segment)
}

// removedMarker distinguishes removed turns from retained ones after the scan.
const removedMarker = "\x00removed"

// countNonAdjacent marks retained candidate turns that repeat another
// retained candidate turn somewhere else in the history.
func countNonAdjacent(turns []*turn) {
	seen := make(map[uint64][]*turn)
	for _, item := range turns {
		if !item.candidate() {
			continue
		}
		for _, other := range seen[item.identity.hash] {
			if other.identity.equals(item.identity) {
				item.retained = RetainedNonAdjacent
				break
			}
		}
		if item.candidate() {
			seen[item.identity.hash] = append(seen[item.identity.hash], item)
		}
	}
}

func finishRetained(turns []*turn, diagnostics *Diagnostics) {
	for _, item := range turns {
		if item.retained == "" || item.retained == removedMarker {
			continue
		}
		diagnostics.Retained[item.retained]++
	}
}
