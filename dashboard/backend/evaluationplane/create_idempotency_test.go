package evaluationplane

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"sync"
	"testing"
)

const stableClientRequestID = "4d0b4f2c-1fc5-40b0-b04e-876ad9d4d8e2"

func TestCreateRunUsesClientRequestAsAtomicBundleIdentity(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	request := validCreateRequest()
	request.ClientRequestID = stableClientRequestID

	created, err := service.CreateRunAs(context.Background(), SystemActor(), request)
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	if created.ID != request.ClientRequestID || created.ClientRequestID != request.ClientRequestID {
		t.Fatalf("run identity=%+v, want client request UUID", created)
	}
	manifest, _, err := service.readDurableManifest(created.ID)
	if err != nil {
		t.Fatalf("readDurableManifest: %v", err)
	}
	if manifest.RunID != request.ClientRequestID || !createRequestMatchesRun(request, created) {
		t.Fatalf("bundle identity=%+v, want run=%s exact request", manifest, request.ClientRequestID)
	}
	if _, err := os.Stat(filepath.Join(root, "index")); !os.IsNotExist(err) {
		t.Fatalf("fresh store retained a secondary index root: %v", err)
	}
}

func TestCreateRunIdempotencySurvivesRestartAndRejectsChangedPayload(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	request := validCreateRequest()
	request.ClientRequestID = stableClientRequestID
	created, err := service.CreateRunAs(context.Background(), SystemActor(), request)
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	if closeErr := service.Close(); closeErr != nil {
		t.Fatalf("Close original service: %v", closeErr)
	}

	restarted := reopenTestService(t, root)
	replayed, err := restarted.CreateRunAs(context.Background(), SystemActor(), request)
	if err != nil || replayed.ID != created.ID {
		t.Fatalf("idempotent retry returned run=%+v err=%v", replayed, err)
	}
	changed := request
	changed.Description = "different payload"
	if _, retryErr := restarted.CreateRunAs(context.Background(), SystemActor(), changed); !errors.Is(retryErr, ErrConflict) {
		t.Fatalf("changed retry error=%v, want ErrConflict", retryErr)
	}
	whitespaceRepair := request
	whitespaceRepair.Name = " " + request.Name
	if _, retryErr := restarted.CreateRunAs(context.Background(), SystemActor(), whitespaceRepair); !errors.Is(retryErr, ErrInvalid) {
		t.Fatalf("non-canonical retry error=%v, want ErrInvalid before idempotency resolution", retryErr)
	}
	entries, err := os.ReadDir(filepath.Join(root, "runs"))
	if err != nil || len(entries) != 1 {
		t.Fatalf("retries left bundles=%v err=%v, want one", entries, err)
	}
}

func TestCreateRunIdempotencyIsAtomicAcrossServices(t *testing.T) {
	seed, root := newTestService(t, &controlledProcess{}, 1)
	if err := seed.Close(); err != nil {
		t.Fatalf("Close seed service: %v", err)
	}
	first := reopenTestService(t, root)
	second := reopenTestService(t, root)
	request := validCreateRequest()
	request.ClientRequestID = stableClientRequestID

	type result struct {
		run Run
		err error
	}
	results := make(chan result, 2)
	start := make(chan struct{})
	var workers sync.WaitGroup
	for _, service := range []*Service{first, second} {
		workers.Add(1)
		go func(service *Service) {
			defer workers.Done()
			<-start
			run, err := service.CreateRunAs(context.Background(), SystemActor(), request)
			results <- result{run: run, err: err}
		}(service)
	}
	close(start)
	workers.Wait()
	close(results)
	for result := range results {
		if result.err != nil || result.run.ID != request.ClientRequestID {
			t.Fatalf("concurrent create returned run=%+v err=%v", result.run, result.err)
		}
	}
	entries, err := os.ReadDir(filepath.Join(root, "runs"))
	if err != nil || len(entries) != 1 || entries[0].Name() != request.ClientRequestID {
		t.Fatalf("concurrent create bundles=%v err=%v", entries, err)
	}
}

func TestCreateRunIdempotencyRejectsConcurrentDifferentPayloadAcrossServices(t *testing.T) {
	seed, root := newTestService(t, &controlledProcess{}, 1)
	if err := seed.Close(); err != nil {
		t.Fatalf("Close seed service: %v", err)
	}
	first := reopenTestService(t, root)
	second := reopenTestService(t, root)
	firstRequest := validCreateRequest()
	firstRequest.ClientRequestID = stableClientRequestID
	firstRequest.Name = "first payload"
	secondRequest := firstRequest
	secondRequest.Name = "second payload"

	type result struct {
		run Run
		err error
	}
	results := make(chan result, 2)
	start := make(chan struct{})
	var workers sync.WaitGroup
	for _, candidate := range []struct {
		service *Service
		request CreateRunRequest
	}{{first, firstRequest}, {second, secondRequest}} {
		workers.Add(1)
		go func(service *Service, request CreateRunRequest) {
			defer workers.Done()
			<-start
			run, err := service.CreateRunAs(context.Background(), SystemActor(), request)
			results <- result{run: run, err: err}
		}(candidate.service, candidate.request)
	}
	close(start)
	workers.Wait()
	close(results)

	succeeded, conflicted := 0, 0
	for result := range results {
		switch {
		case result.err == nil && result.run.ID == stableClientRequestID:
			succeeded++
		case errors.Is(result.err, ErrConflict):
			conflicted++
		default:
			t.Fatalf("concurrent different create returned run=%+v err=%v", result.run, result.err)
		}
	}
	if succeeded != 1 || conflicted != 1 {
		t.Fatalf("concurrent different create outcomes succeeded=%d conflicted=%d, want 1/1", succeeded, conflicted)
	}
	persisted, err := first.GetRunAs(SystemActor(), stableClientRequestID)
	if err != nil || (persisted.Name != firstRequest.Name && persisted.Name != secondRequest.Name) {
		t.Fatalf("published bundle run=%+v err=%v", persisted, err)
	}
}

func TestCreateRunFailsClosedForCorruptExistingBundle(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	request := validCreateRequest()
	request.ClientRequestID = stableClientRequestID
	created, err := service.CreateRunAs(context.Background(), SystemActor(), request)
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	if err := os.WriteFile(
		filepath.Join(root, "runs", created.ID, runFileName),
		[]byte("{not-json\n"),
		0o600,
	); err != nil {
		t.Fatalf("corrupt status: %v", err)
	}
	if _, err := service.CreateRunAs(context.Background(), SystemActor(), request); err == nil {
		t.Fatal("retry silently replaced a corrupt current-contract bundle")
	}
}

func reopenTestService(t *testing.T, root string) *Service {
	t.Helper()
	service, err := NewService(Options{
		DataDir: root, PythonPath: "python3", ConfigPath: filepath.Join(root, "config.yaml"),
		RouterAPIURL: "http://router.invalid", EnvoyURL: "http://envoy.invalid",
		CodeRevision: testSourceRevision, MaxConcurrent: 1, Process: &controlledProcess{},
	})
	if err != nil {
		t.Fatalf("restart NewService: %v", err)
	}
	t.Cleanup(func() { _ = service.Close() })
	return service
}
