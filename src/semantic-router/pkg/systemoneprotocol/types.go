// Package systemoneprotocol defines the schema-neutral System One contract
// the Router's schema codecs convert through (see systemonecodec): requests,
// batches, questions, answers, errors and listings, with JSON content kept as
// order-preserving Values, and the canonical rendering of a state.
//
// It is the only pivot between schemas. A listener's codec reads a client's
// request into it and a target's codec writes it in the target's schema; the
// Router's internal request form, which the model runtime's /v1/bundle and
// /v1/decisions read, is one more encoding of it (systemonecodec's
// InternalCodec), so a request read in that form writes back unchanged.
//
// The field registry (Fields) is generated from the vendored standard and the
// vllm-sr extension spec in config/schemas/systemone.
package systemoneprotocol

import (
	"fmt"
	"strings"
	"time"
)

// Role is what a state part holds for a role-aware family.
type Role string

const (
	RoleUser    Role = "user"
	RoleContext Role = "context"
	RoleAnswer  Role = "answer"
)

// The standard question types.
const (
	QuestionChoice = "choice"
	QuestionNoul   = "noul"
	QuestionScore  = "score"
	QuestionBox    = "box"
)

// RolesMode says how a request gives its state's parts their roles.
type RolesMode uint8

const (
	// RolesImplicit gives an object state's parts the roles their keys name,
	// as vllm-sr/4.0 does (ImplicitRole).
	RolesImplicit RolesMode = iota
	// RolesNone gives no part a role: the whole state is the user's content
	// on every family, as a vllm-sr/5.0 request without roles has it.
	RolesNone
	// RolesExplicit gives each part the role Roles.Map names.
	RolesExplicit
)

// Roles are the roles of a request's state parts.
type Roles struct {
	Mode RolesMode
	// Map is each part's role by state key, in order (RolesExplicit).
	Map []Member
}

// Request is one request: a state and the questions about it. Questions keep
// their order; the extension fields a question or the request carries stay in
// their Extensions, which a codec sends, strips or refuses. A form a schema
// spells another way (choices, levels and labels for criteria, images and
// videos for assets) stays as it was read, and a codec for a schema without
// it writes the standard form (Question.Standard).
type Request struct {
	Model     string
	State     Value
	Questions []NamedQuestion
	// Assets are images (and videos) by placeholder, in order.
	Assets []Member
	Roles  Roles
	// Extensions are the request's other members, by name.
	Extensions []Member
	// Order is the order of the request's members on the wire it was read
	// from (model and options among them for a request alone), which a codec
	// that keeps the wire form writes back; nil is the codec's own order.
	Order []string
}

// NamedQuestion is a question under the name its answer comes back under.
type NamedQuestion struct {
	Name     string
	Question Question
}

// Question is one question. Instructions and Criteria are null when the
// question has none.
type Question struct {
	Type         string
	Instructions Value
	Criteria     Value
	// Extensions are the question's other members (preset, over, head, and
	// the 4.0 spellings choices, levels and labels), in order.
	Extensions []Member
	// Order is the order of the question's members on the wire.
	Order []string
}

// Options are the request options a vllm-sr server reads. DeadlineMs and
// MaxTokens keep their number literals (null when unset). LongContext is
// options.long_context in the internal form and vllm-sr/4.0, and the
// standard's long-context request header elsewhere.
type Options struct {
	DeadlineMs  Value
	ReturnMeta  *bool
	MaxTokens   Value
	LongContext bool
	// Extra are options the internal form carried that the IR does not read
	// (an unknown name, or a value of another type), kept for the target to
	// judge. Order is the order of the options' members on the wire, and
	// Null says the wire form gave options as null.
	Extra []Member
	Order []string
	Null  bool
}

// IsZero reports whether the options set nothing.
func (o Options) IsZero() bool {
	return o.DeadlineMs.IsNull() && o.ReturnMeta == nil && o.MaxTokens.IsNull() && !o.LongContext && len(o.Extra) == 0
}

// Deadline is the deadline in milliseconds, when one is set.
func (o Options) Deadline() (float64, bool) {
	return o.DeadlineMs.Float64()
}

// BatchForm is how a batch travels on the wire.
type BatchForm uint8

const (
	// FormAlone is one request on its own, not a batch.
	FormAlone BatchForm = iota
	// FormStates is vllm-sr/4.0's and the internal form's batch: named
	// entries under states on the request endpoint.
	FormStates
	// FormRequests is vllm-sr/5.0's batch endpoint: named standard requests
	// under requests.
	FormRequests
)

// Batch is several requests answered together, each under its name; a
// request alone is a batch of one in FormAlone.
type Batch struct {
	Model   string
	Items   []NamedRequest
	Options Options
	Form    BatchForm
	// Shared are members a batch form gives every item at once, as the
	// internal form's assets beside states.
	Shared []Member
	// Order is the order of a batch form's members on the wire.
	Order []string
}

// NamedRequest is one request of a batch.
type NamedRequest struct {
	Name    string
	Request Request
}

// Response is the answer to one request: each question's answer under its
// name, in order, the usage, and the members beside them (spans, sets, meta).
type Response struct {
	Model      string
	Answers    []Member
	Usage      Usage
	Extensions []Member
}

// Usage is the token count of a response.
type Usage struct {
	InputTokens  int64
	OutputTokens int64
}

// Listing is a server's models.
type Listing struct {
	Models []ListedModel
}

// ListedModel is one model a server lists.
type ListedModel struct {
	Name        string
	Description string
	ReleaseDate string
	// Extensions are the entry's other members (version, features, card).
	Extensions []Member
}

// ProblemKind classifies why a server refused or failed a request.
type ProblemKind string

const (
	ProblemRejected    ProblemKind = "rejected"
	ProblemCredential  ProblemKind = "credential"
	ProblemOverloaded  ProblemKind = "overloaded"
	ProblemUnavailable ProblemKind = "unavailable"
	ProblemDeadline    ProblemKind = "deadline"
	ProblemFailed      ProblemKind = "failed"
)

// Problem is an error answer: its status, its kind, what the server said, and
// the question (and batch request) it names when one failed question answers
// the whole request.
type Problem struct {
	Status   int
	Kind     ProblemKind
	Detail   string
	Code     string
	Reason   string
	Question string
	// QuestionError is the failed question's code (invalid_question, ...).
	QuestionError string
	// Item is the batch request the failed question belongs to, "" when the
	// server names none.
	Item       string
	Limit      *int64
	RetryAfter time.Duration
}

func (p *Problem) Error() string {
	var text strings.Builder
	fmt.Fprintf(&text, "systemone %s (status %d)", p.Kind, p.Status)
	if p.Question != "" {
		fmt.Fprintf(&text, ": question %q", p.Question)
	}
	if p.Detail != "" {
		fmt.Fprintf(&text, ": %s", p.Detail)
	}
	return text.String()
}

// Members is a list of members as an object Value.
func Members(members []Member) Value { return Object(members...) }

// QuestionValue is the question as a JSON object: its members in their wire
// order when it was read from one, else type, instructions and criteria
// (when it has them), then its extensions.
func (q Question) QuestionValue() Value {
	members := map[string]Value{"type": String(q.Type)}
	if q.Type == "" {
		delete(members, "type")
	}
	if !q.Instructions.IsNull() || q.has("instructions") {
		members["instructions"] = q.Instructions
	}
	if !q.Criteria.IsNull() || q.has("criteria") {
		members["criteria"] = q.Criteria
	}
	for _, member := range q.Extensions {
		members[member.Key] = member.Value
	}
	order := q.Order
	if order == nil {
		order = []string{"type", "instructions", "criteria"}
		for _, member := range q.Extensions {
			order = append(order, member.Key)
		}
	}
	return orderedObject(order, members, func(key string) int { return extensionIndex(q.Extensions, key) })
}

// has reports whether the question's wire form had a member, null or not.
func (q Question) has(key string) bool {
	for _, name := range q.Order {
		if name == key {
			return true
		}
	}
	return false
}

func extensionIndex(members []Member, key string) int {
	for index, member := range members {
		if member.Key == key {
			return index
		}
	}
	return -1
}

// orderedObject writes members in order, then any member order does not
// name in the order of rank.
func orderedObject(order []string, members map[string]Value, rank func(string) int) Value {
	object := Object()
	written := make(map[string]bool, len(members))
	for _, key := range order {
		if value, ok := members[key]; ok && !written[key] {
			object, written[key] = object.With(key, value), true
		}
	}
	rest := make([]string, 0, len(members))
	for key := range members {
		if !written[key] {
			rest = append(rest, key)
		}
	}
	sortByRank(rest, rank)
	for _, key := range rest {
		object = object.With(key, members[key])
	}
	return object
}

func sortByRank(keys []string, rank func(string) int) {
	for i := 1; i < len(keys); i++ {
		for j := i; j > 0 && lessRank(keys[j], keys[j-1], rank); j-- {
			keys[j], keys[j-1] = keys[j-1], keys[j]
		}
	}
}

func lessRank(a, b string, rank func(string) int) bool {
	ra, rb := rank(a), rank(b)
	if ra != rb {
		return ra < rb
	}
	return a < b
}

// ParseQuestion reads a question object: its type, instructions and
// criteria, and every other member as an extension, as they were written.
func ParseQuestion(object Value) (Question, error) {
	if object.Kind() != KindObject {
		return Question{}, fmt.Errorf("a question must be an object")
	}
	question := Question{Order: object.Keys()}
	for _, member := range object.Members() {
		switch member.Key {
		case "type":
			text, ok := member.Value.Text()
			if !ok || text == "" {
				return Question{}, fmt.Errorf("a question's type must be a non-empty string")
			}
			question.Type = text
		case "instructions":
			question.Instructions = member.Value
		case "criteria":
			question.Criteria = member.Value
		default:
			question.Extensions = append(question.Extensions, member)
		}
	}
	return question, nil
}

// Extension is a question's extension member.
func (q Question) Extension(name string) (Value, bool) {
	for _, member := range q.Extensions {
		if member.Key == name {
			return member.Value, true
		}
	}
	return Value{}, false
}

// WithoutExtension is the question without an extension member.
func (q Question) WithoutExtension(name string) Question {
	kept := make([]Member, 0, len(q.Extensions))
	for _, member := range q.Extensions {
		if member.Key != name {
			kept = append(kept, member)
		}
	}
	q.Extensions = kept
	return q
}

// Standard is the question in the standard's form: the 4.0 spellings choices
// and labels become a criteria object of option keys to descriptions (a
// missing description null), and levels a criteria array; the question then
// has no wire order.
func (q Question) Standard() (Question, error) {
	var rest []Member
	for _, member := range q.Extensions {
		switch member.Key {
		case "choices", "labels":
			if !q.Criteria.IsNull() {
				return q, fmt.Errorf("use criteria or %s, not both", member.Key)
			}
			criteria := Object()
			for _, option := range member.Value.Items() {
				keyValue, _ := option.Get("key")
				key, ok := keyValue.Text()
				if !ok || strings.TrimSpace(key) == "" {
					return q, fmt.Errorf("each entry of %s needs a key", member.Key)
				}
				description, _ := option.Get("description")
				criteria = criteria.With(key, description)
			}
			q.Criteria = criteria
		case "levels":
			if !q.Criteria.IsNull() {
				return q, fmt.Errorf("use criteria or levels, not both")
			}
			q.Criteria = Array(member.Value.Items()...)
		default:
			rest = append(rest, member)
		}
	}
	q.Extensions, q.Order = rest, nil
	return q, nil
}

// CanonicalState is the state's canonical text: a string as it is, and an
// object or array as Canonical writes it.
func (r Request) CanonicalState() string {
	if text, ok := r.State.Text(); ok {
		return text
	}
	return Canonical(r.State)
}

// ImplicitRole is the role vllm-sr/4.0 gives a state key, whatever its case,
// as the model runtime's STATE_KEY_ROLES reads it: request, user and prompt
// hold the user's content, answer and response the answer, and every other
// key context.
func ImplicitRole(key string) Role {
	switch strings.ToLower(key) {
	case "request", "user", "prompt":
		return RoleUser
	case "answer", "response":
		return RoleAnswer
	}
	return RoleContext
}

// ImplicitRoles are the roles vllm-sr/4.0 gives an object state's keys, in
// key order; nil for a state that is not an object.
func ImplicitRoles(state Value) []Member {
	if state.Kind() != KindObject {
		return nil
	}
	roles := make([]Member, 0, len(state.Members()))
	for _, member := range state.Members() {
		roles = append(roles, Member{Key: member.Key, Value: String(string(ImplicitRole(member.Key)))})
	}
	return roles
}

// SameAsImplicit reports whether explicit roles give an object state's keys
// exactly the roles vllm-sr/4.0's implicit roles give them.
func (r Roles) SameAsImplicit(state Value) bool {
	switch r.Mode {
	case RolesImplicit:
		return true
	case RolesNone:
		return state.Kind() != KindObject
	}
	if state.Kind() != KindObject {
		return len(r.Map) == 0
	}
	if len(r.Map) != len(state.Members()) {
		return false
	}
	for _, member := range r.Map {
		if !state.Has(member.Key) {
			return false
		}
		role, _ := member.Value.Text()
		if Role(role) != ImplicitRole(member.Key) {
			return false
		}
	}
	return true
}
