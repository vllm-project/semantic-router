//go:build !windows

package apiserver

import (
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

const (
	apiObservabilityToolEventsPath = apiObservabilityPath + "/tool-events"
	toolEventsRequestLimit         = 64 << 10
	maxToolEventsPerBatch          = 64
	maxToolEventIDBytes            = 256
	maxToolEventHarnessBytes       = 64
	// toolEventClockSkew is how far ahead of the Router's clock a reported
	// time may be.
	toolEventClockSkew = 5 * time.Minute
)

// ToolEventsRequest is a batch of tool calls one harness session ran, as the
// harness's vSR plugin reports them for Router Replay's run view: names, call
// ids, times and outcomes, never a tool's input or output. session_id is the
// session the Router files the run under.
type ToolEventsRequest struct {
	Harness   string            `json:"harness,omitempty"`
	SessionID string            `json:"session_id"`
	Events    []ToolEventReport `json:"events"`
}

// ToolEventReport is one tool call. status is succeeded, failed or denied
// (refused, never run). started_at is when the tool began, ended_at when it
// ended or was refused, and approval_requested_at when the harness put it to
// approval: before started_at when the harness reports the start after the
// answer, after it when the start includes the wait.
type ToolEventReport struct {
	ToolCallID          string     `json:"tool_call_id"`
	Tool                string     `json:"tool"`
	Status              string     `json:"status"`
	StartedAt           *time.Time `json:"started_at,omitempty"`
	EndedAt             *time.Time `json:"ended_at,omitempty"`
	ApprovalRequestedAt *time.Time `json:"approval_requested_at,omitempty"`
}

// ToolEventsResponse says how many events the Router kept.
type ToolEventsResponse struct {
	Accepted int `json:"accepted"`
}

// toolEventRecorder is the replay runtime of a generation that keeps tool
// events.
type toolEventRecorder interface {
	RecordToolEvents(events []store.ToolEvent) error
}

func (s *ClassificationAPIServer) handleRouterToolEvents(w http.ResponseWriter, r *http.Request) {
	var req ToolEventsRequest
	if err := s.parseStrictJSONRequest(r, &req); err != nil {
		s.writeJSONRequestError(w, err)
		return
	}
	events, err := toolEventsOf(req, time.Now())
	if err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_TOOL_EVENTS", err.Error())
		return
	}
	runtime, release := s.currentReplayRuntime()
	defer release()
	recorder, ok := runtime.(toolEventRecorder)
	if !ok {
		s.writeErrorResponse(w, http.StatusNotFound, "REPLAY_UNAVAILABLE", "router replay is disabled or unavailable")
		return
	}
	if err := recorder.RecordToolEvents(events); err != nil {
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "TOOL_EVENTS_UNAVAILABLE", "tool events could not be kept")
		return
	}
	s.writeJSONResponse(w, http.StatusOK, ToolEventsResponse{Accepted: len(events)})
}

// toolEventsOf checks a batch and stamps each event with when the Router took
// it.
func toolEventsOf(req ToolEventsRequest, now time.Time) ([]store.ToolEvent, error) {
	session, harness := strings.TrimSpace(req.SessionID), strings.TrimSpace(req.Harness)
	switch {
	case session == "" || len(session) > maxToolEventIDBytes:
		return nil, fmt.Errorf("session_id is required, at most %d bytes", maxToolEventIDBytes)
	case len(harness) > maxToolEventHarnessBytes:
		return nil, fmt.Errorf("harness is at most %d bytes", maxToolEventHarnessBytes)
	case len(req.Events) == 0 || len(req.Events) > maxToolEventsPerBatch:
		return nil, fmt.Errorf("events holds 1 to %d tool calls", maxToolEventsPerBatch)
	}
	events := make([]store.ToolEvent, 0, len(req.Events))
	for index, report := range req.Events {
		if err := checkToolEventReport(report, now); err != nil {
			return nil, fmt.Errorf("events[%d]: %w", index, err)
		}
		events = append(events, store.ToolEvent{
			SessionID: session, Harness: harness, ToolCallID: report.ToolCallID, Tool: report.Tool,
			Status: report.Status, StartedAt: report.StartedAt, EndedAt: report.EndedAt,
			ApprovalRequestedAt: report.ApprovalRequestedAt, ReceivedAt: now,
		})
	}
	return events, nil
}

func checkToolEventReport(report ToolEventReport, now time.Time) error {
	switch {
	case report.ToolCallID == "" || len(report.ToolCallID) > maxToolEventIDBytes:
		return fmt.Errorf("tool_call_id is required, at most %d bytes", maxToolEventIDBytes)
	case report.Tool == "" || len(report.Tool) > maxToolEventIDBytes:
		return fmt.Errorf("tool is required, at most %d bytes", maxToolEventIDBytes)
	case report.Status != store.ToolEventSucceeded && report.Status != store.ToolEventFailed && report.Status != store.ToolEventDenied:
		return fmt.Errorf("status is succeeded, failed or denied")
	}
	return checkToolEventTimes(report, now)
}

func checkToolEventTimes(report ToolEventReport, now time.Time) error {
	for _, at := range []*time.Time{report.StartedAt, report.EndedAt, report.ApprovalRequestedAt} {
		if at != nil && at.After(now.Add(toolEventClockSkew)) {
			return fmt.Errorf("a time is more than %s ahead of the Router's clock", toolEventClockSkew)
		}
	}
	if report.StartedAt != nil && report.EndedAt != nil && report.EndedAt.Before(*report.StartedAt) {
		return fmt.Errorf("ended_at is before started_at")
	}
	return nil
}
