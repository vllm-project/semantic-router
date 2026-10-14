package parity

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

// Record is the trace of one request through a gateway.
type Record struct {
	Case   string        `json:"case"`
	Phases []PhaseRecord `json:"phases,omitempty"`
	// Route and Upstream describe the planned upstream call, if any.
	Route    string           `json:"route,omitempty"`
	Upstream *Message         `json:"upstream,omitempty"`
	Response *Message         `json:"response,omitempty"`
	Evidence routing.Evidence `json:"evidence"`
	Error    string           `json:"error,omitempty"`
}

// PhaseRecord is one phase: what the routing core saw and what it answered.
type PhaseRecord struct {
	Phase       routing.Phase  `json:"phase"`
	EndOfStream bool           `json:"end_of_stream,omitempty"`
	Header      routing.Header `json:"header,omitempty"`
	Body        Text           `json:"body,omitempty"`
	Effect      *Effect        `json:"effect,omitempty"`
	Error       string         `json:"error,omitempty"`
}

// Message is a request or response as it crosses the gateway.
type Message struct {
	Status int            `json:"status,omitempty"`
	Header routing.Header `json:"header"`
	Body   Text           `json:"body,omitempty"`
	// Chunks holds a streamed body as the client received it.
	Chunks []Text `json:"chunks,omitempty"`
}

// Effect mirrors routing.Effect with readable bodies.
type Effect struct {
	Header           *routing.HeaderMutation `json:"header,omitempty"`
	Body             *BodyMutation           `json:"body,omitempty"`
	ClearRouteCache  bool                    `json:"clear_route_cache,omitempty"`
	ResponseBodyMode routing.BodyMode        `json:"response_body_mode,omitempty"`
	Immediate        *Immediate              `json:"immediate,omitempty"`
}

// BodyMutation mirrors routing.BodyMutation.
type BodyMutation struct {
	Clear bool `json:"clear,omitempty"`
	Body  Text `json:"body,omitempty"`
}

// Immediate mirrors routing.ImmediateResponse.
type Immediate struct {
	Status  int                     `json:"status"`
	Header  *routing.HeaderMutation `json:"header,omitempty"`
	Body    Text                    `json:"body,omitempty"`
	Details string                  `json:"details,omitempty"`
}

// EffectRecord converts a routing effect for a record.
func EffectRecord(effect *routing.Effect) *Effect {
	if effect == nil {
		return nil
	}
	out := &Effect{
		Header:           effect.Header,
		ClearRouteCache:  effect.ClearRouteCache,
		ResponseBodyMode: effect.ResponseBodyMode,
	}
	if effect.Body != nil {
		out.Body = &BodyMutation{Clear: effect.Body.Clear, Body: effect.Body.Body}
	}
	if effect.Immediate != nil {
		out.Immediate = &Immediate{
			Status:  effect.Immediate.Status,
			Header:  effect.Immediate.Header,
			Body:    effect.Immediate.Body,
			Details: effect.Immediate.Details,
		}
	}
	return out
}

// Text is a byte string that encodes as a JSON string when it is valid UTF-8
// and as {"base64": "..."} otherwise.
type Text []byte

func (t Text) MarshalJSON() ([]byte, error) {
	if utf8.Valid(t) {
		var out bytes.Buffer
		encoder := json.NewEncoder(&out)
		encoder.SetEscapeHTML(false)
		if err := encoder.Encode(string(t)); err != nil {
			return nil, err
		}
		return bytes.TrimSuffix(out.Bytes(), []byte("\n")), nil
	}
	return json.Marshal(map[string]string{"base64": base64.StdEncoding.EncodeToString(t)})
}

func (t *Text) UnmarshalJSON(data []byte) error {
	var text string
	if err := json.Unmarshal(data, &text); err == nil {
		*t = Text(text)
		return nil
	}
	var encoded struct {
		Base64 string `json:"base64"`
	}
	if err := json.Unmarshal(data, &encoded); err != nil {
		return fmt.Errorf("parity text: %w", err)
	}
	decoded, err := base64.StdEncoding.DecodeString(encoded.Base64)
	if err != nil {
		return err
	}
	*t = decoded
	return nil
}

// Normalize rewrites volatile values in place: clock fields in bodies,
// volatile header values, and the order of header removals.
func (r *Record) Normalize() *Record {
	for i := range r.Phases {
		phase := &r.Phases[i]
		phase.Header = NormalizeHeader(phase.Header)
		phase.Body = NormalizeBody(phase.Body)
		if phase.Effect != nil {
			phase.Effect.normalize()
		}
	}
	r.Upstream.normalize()
	r.Response.normalize()
	return r
}

func (e *Effect) normalize() {
	e.Header = NormalizeMutation(e.Header)
	if e.Body != nil {
		e.Body.Body = NormalizeBody(e.Body.Body)
	}
	if e.Immediate != nil {
		e.Immediate.Header = NormalizeMutation(e.Immediate.Header)
		e.Immediate.Body = NormalizeBody(e.Immediate.Body)
	}
}

func (m *Message) normalize() {
	if m == nil {
		return
	}
	m.Header = NormalizeHeader(m.Header)
	m.Body = NormalizeBody(m.Body)
	for i := range m.Chunks {
		m.Chunks[i] = NormalizeBody(m.Chunks[i])
	}
}

// Encode renders the record as indented JSON.
func (r *Record) Encode() ([]byte, error) {
	var out bytes.Buffer
	encoder := json.NewEncoder(&out)
	encoder.SetEscapeHTML(false)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(r); err != nil {
		return nil, err
	}
	return out.Bytes(), nil
}
