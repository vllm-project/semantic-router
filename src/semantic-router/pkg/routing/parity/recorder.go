package parity

import (
	"context"
	"errors"
	"io"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

// Tape records every phase of the sessions opened through it.
type Tape struct {
	mu     sync.Mutex
	phases []PhaseRecord
}

// Processor wraps p so that its sessions record onto the tape.
func (t *Tape) Processor(p routing.Processor) routing.Processor {
	return tapedProcessor{tape: t, next: p}
}

// Phases returns the recorded phases.
func (t *Tape) Phases() []PhaseRecord {
	t.mu.Lock()
	defer t.mu.Unlock()
	return append([]PhaseRecord(nil), t.phases...)
}

// Reset forgets the recorded phases.
func (t *Tape) Reset() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.phases = nil
}

func (t *Tape) add(phase PhaseRecord) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.phases = append(t.phases, phase)
}

type tapedProcessor struct {
	tape *Tape
	next routing.Processor
}

func (p tapedProcessor) Open(ctx context.Context) (routing.Session, error) {
	session, err := p.next.Open(ctx)
	if err != nil {
		return nil, err
	}
	return &tapedSession{tape: p.tape, next: session}, nil
}

type tapedSession struct {
	tape *Tape
	next routing.Session
}

func (s *tapedSession) record(phase PhaseRecord, effect *routing.Effect, err error) (*routing.Effect, error) {
	phase.Effect = EffectRecord(effect)
	if err != nil {
		phase.Error = err.Error()
	}
	s.tape.add(phase)
	return effect, err
}

func (s *tapedSession) RequestHeaders(header routing.Header, endOfStream bool) (*routing.Effect, error) {
	effect, err := s.next.RequestHeaders(header, endOfStream)
	return s.record(PhaseRecord{Phase: routing.PhaseRequestHeaders, EndOfStream: endOfStream, Header: header.Clone()}, effect, err)
}

func (s *tapedSession) RequestBody(body []byte, endOfStream bool) (*routing.Effect, error) {
	effect, err := s.next.RequestBody(body, endOfStream)
	return s.record(PhaseRecord{Phase: routing.PhaseRequestBody, EndOfStream: endOfStream, Body: append(Text(nil), body...)}, effect, err)
}

func (s *tapedSession) ResponseHeaders(header routing.Header, endOfStream bool) (*routing.Effect, error) {
	effect, err := s.next.ResponseHeaders(header, endOfStream)
	return s.record(PhaseRecord{Phase: routing.PhaseResponseHeaders, EndOfStream: endOfStream, Header: header.Clone()}, effect, err)
}

func (s *tapedSession) ResponseBody(body []byte, endOfStream bool) (*routing.Effect, error) {
	effect, err := s.next.ResponseBody(body, endOfStream)
	return s.record(PhaseRecord{Phase: routing.PhaseResponseBody, EndOfStream: endOfStream, Body: append(Text(nil), body...)}, effect, err)
}

func (s *tapedSession) Evidence() routing.Evidence { return s.next.Evidence() }

func (s *tapedSession) Close(err error) { s.next.Close(err) }

// Recorder runs corpus cases through a routing engine the way a gateway does
// and records each one.
type Recorder struct {
	tape   Tape
	engine routing.Engine
}

// NewRecorder returns a Recorder for an engine built on p with opts.
func NewRecorder(p routing.Processor, opts routing.Options) *Recorder {
	r := &Recorder{}
	r.engine = routing.NewEngine(r.tape.Processor(p), opts)
	return r
}

// Run plans c's request, answers a planned call with c's upstream fixture,
// reads the client response to its end, finishes the request, and returns the
// normalized record. Cases run one at a time.
func (r *Recorder) Run(ctx context.Context, c Case) *Record {
	r.tape.Reset()
	record := &Record{Case: c.Name}
	plan, err := r.engine.Plan(ctx, c.GatewayRequest())
	if err != nil {
		record.Error = err.Error()
		record.Phases = r.tape.Phases()
		return record.Normalize()
	}
	finishErr := r.serve(ctx, c, plan, record)
	plan.Finish(finishErr)
	record.Evidence = plan.Evidence
	record.Phases = r.tape.Phases()
	return record.Normalize()
}

var errNoUpstreamFixture = errors.New("the case planned an upstream call but has no upstream fixture")

func (r *Recorder) serve(ctx context.Context, c Case, plan *routing.Plan, record *Record) error {
	if plan.Immediate != nil {
		record.Response = responseMessage(plan.Immediate)
		return nil
	}
	record.Route = plan.Call.Route
	record.Upstream = &Message{Header: plan.Call.Request.Header.Clone(), Body: plan.Call.Request.Body}
	if c.Upstream == nil {
		record.Error = errNoUpstreamFixture.Error()
		return context.Canceled
	}
	resp, err := r.engine.Respond(ctx, plan, c.Upstream.Response())
	if err != nil {
		record.Error = err.Error()
		return err
	}
	record.Response = responseMessage(resp)
	if resp.Stream == nil {
		return nil
	}
	for {
		chunk, err := resp.Stream.Next()
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			record.Error = err.Error()
			return err
		}
		record.Response.Chunks = append(record.Response.Chunks, chunk)
	}
}

func responseMessage(resp *routing.Response) *Message {
	return &Message{Status: resp.Status, Header: resp.Header.Clone(), Body: resp.Body}
}
