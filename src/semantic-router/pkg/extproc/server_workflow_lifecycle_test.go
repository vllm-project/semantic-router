package extproc

import (
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime"
)

func TestReloadRouterFromConfig_WorkflowMemoryStateDoesNotSurvive(t *testing.T) {
	restore := stubReloadSeams(t)
	defer restore()

	upstream, _ := newWorkflowPauseResumeServer(t)
	cfg := newWorkflowLooperConfig(t, upstream.URL, config.WorkflowStateBackendMemory)
	oldRouter := newWorkflowRouter(t, cfg)
	server := &Server{service: NewRouterService(oldRouter)}
	t.Cleanup(func() { _ = server.service.Close() })
	stubWorkflowReloadSeams(t, cfg)

	pauseResp := routeWorkflowRequest(t, oldRouter, workflowPauseChatBody(t))
	if got := immediateStatus(pauseResp); got != 200 {
		t.Fatalf("pause status = %d, body %s", got, immediateBody(pauseResp))
	}

	if err := server.reloadRouterFromConfig("file", "/tmp/unused-workflow.yaml", oldRouter.Config); err != nil {
		t.Fatalf("reloadRouterFromConfig: %v", err)
	}
	newRouter := server.service.GetRouter()
	if newRouter == oldRouter {
		t.Fatal("reload did not swap the router generation")
	}

	resumeResp := routeWorkflowRequest(t, newRouter, workflowResumeChatBody(t, immediateBody(pauseResp)))
	if got := immediateStatus(resumeResp); got != 500 {
		t.Fatalf("memory resume after reload status = %d, want 500, body %s", got, immediateBody(resumeResp))
	}
	if !strings.Contains(string(immediateBody(resumeResp)), "not found or expired") {
		t.Fatalf("memory resume after reload body = %s", immediateBody(resumeResp))
	}
}

func TestReloadRouterFromConfig_WorkflowFileStateSurvives(t *testing.T) {
	restore := stubReloadSeams(t)
	defer restore()

	upstream, tracker := newWorkflowPauseResumeServer(t)
	cfg := newWorkflowLooperConfig(t, upstream.URL, config.WorkflowStateBackendFile)
	oldRouter := newWorkflowRouter(t, cfg)
	server := &Server{service: NewRouterService(oldRouter)}
	t.Cleanup(func() { _ = server.service.Close() })
	stubWorkflowReloadSeams(t, cfg)

	pauseResp := routeWorkflowRequest(t, oldRouter, workflowPauseChatBody(t))
	if got := immediateStatus(pauseResp); got != 200 {
		t.Fatalf("pause status = %d, body %s", got, immediateBody(pauseResp))
	}

	if err := server.reloadRouterFromConfig("file", "/tmp/unused-workflow.yaml", oldRouter.Config); err != nil {
		t.Fatalf("reloadRouterFromConfig: %v", err)
	}

	resumeResp := routeWorkflowRequest(t, server.service.GetRouter(), workflowResumeChatBody(t, immediateBody(pauseResp)))
	if got := immediateStatus(resumeResp); got != 200 {
		t.Fatalf("file resume after reload status = %d, body %s", got, immediateBody(resumeResp))
	}
	if !tracker.sawToolResult() || !tracker.sawFinal() {
		t.Fatal("file resume after reload did not finish the workflow")
	}
}

func TestReloadRouterFromConfig_WorkflowRedisStateSurvives(t *testing.T) {
	restore := stubReloadSeams(t)
	defer restore()

	upstream, tracker := newWorkflowPauseResumeServer(t)
	cfg := newWorkflowLooperConfig(t, upstream.URL, config.WorkflowStateBackendRedis)
	oldRouter := newWorkflowRouter(t, cfg)
	server := &Server{service: NewRouterService(oldRouter)}
	t.Cleanup(func() { _ = server.service.Close() })
	stubWorkflowReloadSeams(t, cfg)

	pauseResp := routeWorkflowRequest(t, oldRouter, workflowPauseChatBody(t))
	if got := immediateStatus(pauseResp); got != 200 {
		t.Fatalf("pause status = %d, body %s", got, immediateBody(pauseResp))
	}

	if err := server.reloadRouterFromConfig("file", "/tmp/unused-workflow.yaml", oldRouter.Config); err != nil {
		t.Fatalf("reloadRouterFromConfig: %v", err)
	}

	resumeResp := routeWorkflowRequest(t, server.service.GetRouter(), workflowResumeChatBody(t, immediateBody(pauseResp)))
	if got := immediateStatus(resumeResp); got != 200 {
		t.Fatalf("redis resume after reload status = %d, body %s", got, immediateBody(resumeResp))
	}
	if !tracker.sawToolResult() || !tracker.sawFinal() {
		t.Fatal("redis resume after reload did not finish the workflow")
	}
}

func TestServerStopClosesWorkflowStateService(t *testing.T) {
	upstream, _ := newWorkflowPauseResumeServer(t)
	router := newWorkflowRouter(t, newWorkflowLooperConfig(t, upstream.URL, config.WorkflowStateBackendMemory))
	svc := router.WorkflowStateService
	server := &Server{service: NewRouterService(router)}
	t.Cleanup(func() { server.Stop() })

	server.Stop()

	if svc.Acquire() {
		t.Fatal("workflow state service accepted Acquire after Server.Stop")
	}
	if err := server.service.Process(NewMockStream(nil)); err == nil || !strings.Contains(err.Error(), "shutting down") {
		t.Fatalf("Process after Stop error = %v, want shutting down", err)
	}
}

func TestRouterServiceCloseDrainsWorkflowGenerationBeforeStoreClose(t *testing.T) {
	upstream, _ := newWorkflowPauseResumeServer(t)
	router := newWorkflowRouter(t, newWorkflowLooperConfig(t, upstream.URL, config.WorkflowStateBackendMemory))
	svc := router.WorkflowStateService
	service := NewRouterService(router)
	generation := service.current.Load()
	generation.refs.Add(1)

	closed := make(chan struct{})
	go func() {
		_ = service.Close()
		close(closed)
	}()

	select {
	case <-closed:
		t.Fatal("Close returned before the leased generation drained")
	case <-time.After(40 * time.Millisecond):
	}

	if !svc.Acquire() {
		t.Fatal("workflow store closed while a generation lease was held")
	}
	svc.Release()

	generation.refs.Done()
	select {
	case <-closed:
	case <-time.After(time.Second):
		t.Fatal("Close did not finish after the generation lease was released")
	}
	if svc.Acquire() {
		t.Fatal("workflow state service still acquirable after Close")
	}
}

func stubWorkflowReloadSeams(t *testing.T, looperCfg config.LooperConfig) {
	t.Helper()
	ensureReloadConfigModels = func(*config.RouterConfig) error { return nil }
	prepareReloadRuntime = func(*config.RouterConfig) (modelruntime.EmbeddingRuntimeState, error) {
		return modelruntime.EmbeddingRuntimeState{}, nil
	}
	warmupReloadRouter = func(*OpenAIRouter, modelruntime.EmbeddingRuntimeState) error { return nil }
	replaceReloadConfig = func(*config.RouterConfig) {}
	buildReloadRouter = func(*config.RouterConfig) (*OpenAIRouter, error) {
		return newWorkflowRouter(t, looperCfg), nil
	}
}
