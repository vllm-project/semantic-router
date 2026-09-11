package contextcompression

import "context"

// ContentSource records trusted origin independently of compression target policy.
type ContentSource string

const (
	SourceHistory ContentSource = "history"
	SourceRAG     ContentSource = "rag"
	SourceMemory  ContentSource = "memory"
	SourceMixed   ContentSource = "mixed"
)

type Eligibility uint8

const (
	EligibleHistoryRemoval Eligibility = 1 << iota
)

// Protection is a set of reasons a message cannot be removed. Multimodal and
// tool structure can still contain independently eligible text compression blocks.
type Protection uint16

const (
	ProtectInstructions Protection = 1 << iota
	ProtectLiveTurn
	ProtectMultimodal
	ProtectAuthorization
	ProtectSafety
	ProtectContinuation
	ProtectUnknown
)

type TransformationKind uint8

const (
	TransformReset TransformationKind = iota + 1
	TransformDeduplicate
	TransformSelectTurns
	TransformCompress
)

type TransformationInput string

type TransformationOutput string

const (
	InputEligibleHistory  TransformationInput  = "eligible_history"
	InputConfiguredBlocks TransformationInput  = "configured_blocks"
	OutputRemoveMessages  TransformationOutput = "remove_messages"
	OutputReplaceText     TransformationOutput = "replace_text"
)

// TransformationStep is an internal policy contract. Policies receive detached
// values and propose edits; they never receive mutable protocol messages.
// Disabled or empty proposals skip. FailureOpen rejects the entire step and
// continues; FailureClosed rejects the step and stops the remaining plan.
type TransformationStep struct {
	Kind        TransformationKind
	Enabled     bool
	FailureMode FailureMode
	Propose     func(context.Context, TransformationView) (TransformationEdits, error)
}

type TransformationView struct {
	Messages []MessageView
}

type MessageView struct {
	ID          int
	Role        string
	Source      ContentSource
	Eligibility Eligibility
	Protection  Protection
	TurnID      int
	ExchangeIDs []string
	Blocks      []BlockView
}

type BlockView struct {
	ID     int
	Source TargetKind
	Text   string
}

type TextReplacement struct {
	MessageID int
	BlockID   int
	Text      string
}

type TransformationEdits struct {
	RemoveMessages []int
	ReplaceText    []TextReplacement
}

type TransformationStatus string

const (
	TransformationApplied TransformationStatus = "applied"
	TransformationSkipped TransformationStatus = "skipped"
	TransformationFailed  TransformationStatus = "failed"
)

// TransformationReceipt intentionally excludes text, tool arguments, hashes,
// arbitrary error strings, and external identifiers.
type TransformationReceipt struct {
	Kind            TransformationKind
	Input           TransformationInput
	Output          TransformationOutput
	Status          TransformationStatus
	Reason          string
	MessagesRemoved int
	BlocksReplaced  int
}

// TransformationPlan belongs to one RequestIR and is request-local, not shared
// between goroutines. Repeating a completed step returns its original receipt.
type TransformationPlan struct {
	receipts       []TransformationReceipt
	historyEnabled bool
	last           TransformationKind
	terminal       error
}

func (plan *TransformationPlan) Receipts() []TransformationReceipt {
	return append([]TransformationReceipt(nil), plan.receipts...)
}

func (request *RequestIR) TransformationView() TransformationView {
	view := TransformationView{Messages: make([]MessageView, 0, len(request.Messages))}
	for _, message := range request.Messages {
		item := MessageView{
			ID: message.Index, Role: message.Role, Source: message.Source,
			Eligibility: message.Eligibility, Protection: message.Protection,
			TurnID: message.TurnID, ExchangeIDs: append([]string(nil), message.ExchangeIDs...),
		}
		for _, block := range message.Blocks {
			item.Blocks = append(item.Blocks, BlockView{ID: block.BlockIndex, Source: block.Source, Text: block.Text})
		}
		view.Messages = append(view.Messages, item)
	}
	return view
}
