package routerreplay

import (
	"fmt"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func TestRecorderRequestPoliciesKeepConcurrentCaptureIndependent(t *testing.T) {
	owner := NewRecorder(store.NewMemoryStore(100, 0))
	t.Cleanup(func() { require.NoError(t, owner.Close()) })
	capture := owner.WithCapturePolicy(CapturePolicy{
		CaptureRequestBody: true, CaptureResponseBody: true, MaxBodyBytes: 4,
		MaxToolTraceBytes: 3, MaxToolTraceSteps: 1,
	})
	metadata := owner.WithCapturePolicy(CapturePolicy{MaxBodyBytes: 10})
	var workers sync.WaitGroup
	for i := range 20 {
		workers.Add(1)
		go func() {
			defer workers.Done()
			for _, item := range []struct {
				name string
				view *Recorder
				body string
			}{{"capture", capture, "requ"}, {"metadata", metadata, ""}} {
				id, err := item.view.AddRecord(RoutingRecord{
					ID: fmt.Sprintf("%s-%d", item.name, i), RequestBody: "request-body",
				})
				if err != nil {
					t.Error(err)
					return
				}
				if err := item.view.AttachResponse(id, []byte("response-body")); err != nil {
					t.Error(err)
					return
				}
				record, found := owner.GetRecord(id)
				if !found || record.RequestBody != item.body {
					t.Errorf("%s: request=%q found=%t", id, record.RequestBody, found)
				}
				wantResponse := ""
				if item.name == "capture" {
					wantResponse = "resp"
				}
				if record.ResponseBody != wantResponse {
					t.Errorf("%s: response=%q want=%q", id, record.ResponseBody, wantResponse)
				}
			}
		}()
	}
	workers.Wait()
	require.False(t, owner.ShouldCaptureRequest())
	require.False(t, owner.ShouldCaptureResponse())
	traceID, err := capture.AddRecord(RoutingRecord{Prompt: "abcdef", ToolTrace: &ToolTrace{
		Steps: []ToolTraceStep{{Arguments: "first"}, {Arguments: "second"}},
	}})
	require.NoError(t, err)
	record, found := owner.GetRecord(traceID)
	require.True(t, found)
	require.Equal(t, "abc", record.Prompt)
	require.Len(t, record.ToolTrace.Steps, 1)
	require.Equal(t, "sec", record.ToolTrace.Steps[0].Arguments)
}

func TestRecorderRequestViewsShareLifecycleAndCloseOwnership(t *testing.T) {
	storage := &uncancelableOutcomeStore{Storage: store.NewMemoryStore(10, 0), closed: make(chan struct{})}
	owner := NewRecorder(storage)
	first := owner.WithCapturePolicy(CapturePolicy{CaptureRequestBody: true})
	second := first.WithCapturePolicy(CapturePolicy{})
	id, err := first.AddRecord(RoutingRecord{ID: "shared-lifecycle"})
	require.NoError(t, err)
	require.NoError(t, first.FinalizeLifecycle(id, LifecycleCompleted, "response_complete"))
	require.NoError(t, second.FinalizeLifecycle(id, LifecycleFailed, "late_failure"))
	record, found := owner.GetRecord(id)
	require.True(t, found)
	require.Equal(t, LifecycleCompleted, record.LifecycleState)
	require.Same(t, first.recorderState, second.recorderState)
	require.Same(t, owner.outcomes, second.outcomes)
	require.NoError(t, second.Close())
	require.NoError(t, first.Close())
	require.NoError(t, owner.Close())
	require.EqualValues(t, 1, storage.closeCalls.Load())
}

func TestRecorderRequestPolicyPreservesIgnoredLimitsAndExplicitZero(t *testing.T) {
	owner := NewRecorder(store.NewMemoryStore(10, 0))
	t.Cleanup(func() { require.NoError(t, owner.Close()) })
	owner.SetMaxToolTraceBytes(3)
	owner.SetMaxToolTraceSteps(2)
	for _, limit := range []int{-1, 0} {
		view := owner.WithCapturePolicy(CapturePolicy{MaxToolTraceBytes: limit, MaxToolTraceSteps: limit})
		id, err := view.AddRecord(RoutingRecord{Prompt: "abcdef", ToolTrace: &ToolTrace{
			Steps: []ToolTraceStep{{Arguments: "first"}, {Arguments: "second"}, {Arguments: "third"}},
		}})
		require.NoError(t, err)
		record, found := owner.GetRecord(id)
		require.True(t, found)
		if limit < 0 {
			require.Equal(t, "abc", record.Prompt)
			require.Len(t, record.ToolTrace.Steps, 2)
		} else {
			require.Equal(t, "abcdef", record.Prompt)
			require.Len(t, record.ToolTrace.Steps, 3)
		}
	}
}
