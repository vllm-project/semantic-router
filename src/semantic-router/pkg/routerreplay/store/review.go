package store

import "context"

// RequestKindCodexApprovalReview marks a request Codex sent to judge one
// planned agent action for its "Approve for me" approvals.
const RequestKindCodexApprovalReview = "codex_approval_review"

// ReviewSchemaVersion is the version of the Review shape written now.
const ReviewSchemaVersion = 1

// How a request was recognised as an approval review.
const (
	// ReviewIdentifiedByEntrypoint: the request named an entrypoint that
	// another entrypoint uses as its review_model.
	ReviewIdentifiedByEntrypoint = "entrypoint"
	// ReviewIdentifiedByContent: the request carried Codex's review envelope,
	// its judge instructions, approval request block and verdict schema.
	ReviewIdentifiedByContent = "content"
)

// Review statuses. Only completed carries a verdict; a failed or cancelled
// call is never counted as a deny.
const (
	// ReviewStatusPending: the request is still in flight.
	ReviewStatusPending = "pending"
	// ReviewStatusCompleted: the reviewer answered the verdict schema.
	ReviewStatusCompleted = "completed"
	// ReviewStatusToolCall: the reviewer asked for a read-only tool check;
	// its verdict arrives in a later request of the same review.
	ReviewStatusToolCall = "tool_call"
	// ReviewStatusInvalid: the call completed but the answer was missing or
	// was not the verdict schema.
	ReviewStatusInvalid = "invalid"
	// ReviewStatusFailed: the upstream call or the network failed.
	ReviewStatusFailed = "failed"
	// ReviewStatusCancelled: the client went away before the answer ended.
	ReviewStatusCancelled = "cancelled"
)

const (
	ReviewOutcomeAllow = "allow"
	ReviewOutcomeDeny  = "deny"
)

// Review is one approval review observed on a request. It records what the
// reviewer was asked and what it answered; it never changes routing.
type Review struct {
	SchemaVersion        int    `json:"schema_version"`
	IdentificationSource string `json:"identification_source"`
	// ReviewedSessionID is the agent session whose action was judged.
	ReviewedSessionID string `json:"reviewed_session_id,omitempty"`
	// ThreadID and TurnID identify the reviewer's own thread and turn;
	// ParentTurnID is the reviewed session's turn that planned the action.
	ThreadID     string `json:"thread_id,omitempty"`
	TurnID       string `json:"turn_id,omitempty"`
	ParentTurnID string `json:"parent_turn_id,omitempty"`
	Status       string `json:"status"`
	// Assessment is set only when Status is completed.
	Assessment *ReviewAssessment `json:"assessment,omitempty"`
	Action     *ReviewAction     `json:"action,omitempty"`
	// LatencyMS is the review request's duration once it ended.
	LatencyMS int64 `json:"latency_ms,omitempty"`
}

// ReviewAssessment is the reviewer's verdict. Only Outcome is required by the
// review schema; the other fields stay empty when the reviewer omitted them.
type ReviewAssessment struct {
	Outcome           string `json:"outcome"`
	RiskLevel         string `json:"risk_level,omitempty"`
	UserAuthorization string `json:"user_authorization,omitempty"`
	// Rationale is kept only where replay keeps response text.
	Rationale          string `json:"rationale,omitempty"`
	RationaleTruncated bool   `json:"rationale_truncated,omitempty"`
}

// ReviewAction is the planned action under review. Summary, Cwd and
// Arguments are kept only where replay keeps request text; list responses
// drop Arguments.
type ReviewAction struct {
	Tool               string `json:"tool,omitempty"`
	Summary            string `json:"summary,omitempty"`
	Cwd                string `json:"cwd,omitempty"`
	Arguments          string `json:"arguments,omitempty"`
	ArgumentsTruncated bool   `json:"arguments_truncated,omitempty"`
}

// ReviewWriter settles the approval review observed on an existing record.
type ReviewWriter interface {
	UpdateReview(ctx context.Context, id string, review Review) error
}

// Outcome returns the verdict of a completed review, or "".
func (r *Review) Outcome() string {
	if r == nil || r.Status != ReviewStatusCompleted || r.Assessment == nil {
		return ""
	}
	return r.Assessment.Outcome
}

// RiskLevel returns the risk a completed review reported, or "".
func (r *Review) RiskLevel() string {
	if r == nil || r.Status != ReviewStatusCompleted || r.Assessment == nil {
		return ""
	}
	return r.Assessment.RiskLevel
}

// CloneReview returns a deep copy of review.
func CloneReview(review *Review) *Review {
	if review == nil {
		return nil
	}
	cloned := *review
	if review.Assessment != nil {
		assessment := *review.Assessment
		cloned.Assessment = &assessment
	}
	if review.Action != nil {
		action := *review.Action
		cloned.Action = &action
	}
	return &cloned
}
