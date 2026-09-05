package main

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestRunRouterProcessShutsDownManagementServerWhenStartupIsCancelled(t *testing.T) {
	apiPort := freePort(t)
	routerPort := freePort(t)
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	configYAML := fmt.Sprintf(`version: v0.3
global:
  services:
    management_api:
      bind_address: 127.0.0.1
      port: %d
      remote_exposure: false
      auth:
        mode: disabled
`, apiPort)
	if err := os.WriteFile(configPath, []byte(configYAML), 0o600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	done := make(chan error, 1)
	go func() {
		done <- runRouterProcess(ctx, runtimeOptions{
			configPath:  configPath,
			port:        routerPort,
			apiPort:     apiPort,
			apiBind:     "127.0.0.1",
			metricsPort: 0,
			enableAPI:   true,
		})
	}()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("runRouterProcess() error = %v, want context canceled", err)
		}
	case <-time.After(time.Second):
		t.Fatal("runRouterProcess() did not finish after startup cancellation")
	}

	requireListenerReleased(t, "management", fmt.Sprintf("127.0.0.1:%d", apiPort))
}

func TestRunRouterProcessLoadedShutdownIsBoundedAndReleasesListener(t *testing.T) {
	routerPort := freePort(t)
	metricsPort := freePort(t)
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	configYAML, err := os.ReadFile(filepath.Join("..", "..", "..", "e2e", "config", "config.agent-smoke.cpu.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(configPath, configYAML, 0o600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		done <- runRouterProcess(ctx, runtimeOptions{
			configPath:  configPath,
			port:        routerPort,
			metricsPort: metricsPort,
			enableAPI:   false,
		})
	}()

	address := fmt.Sprintf("127.0.0.1:%d", routerPort)
	waitForProcessListener(t, done, address)

	cancel()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("runRouterProcess() error = %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("loaded Router shutdown exceeded its bound")
	}

	requireListenerReleased(t, "ExtProc", address)
	requireListenerReleased(t, "metrics", fmt.Sprintf("127.0.0.1:%d", metricsPort))
}

func TestShutdownRouterProcessBoundsSlowHookAndPreservesErrors(t *testing.T) {
	hookErr := errors.New("hook failed")
	tracingErr := errors.New("tracing failed")
	hooks := []func(context.Context) error{
		func(ctx context.Context) error {
			<-ctx.Done()
			return errors.Join(hookErr, ctx.Err())
		},
	}
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	started := time.Now()
	err := shutdownRouterProcess(ctx, nil, nil, nil, &hooks, func(context.Context) error {
		return tracingErr
	})
	if elapsed := time.Since(started); elapsed > time.Second {
		t.Fatalf("shutdownRouterProcess() took %s, want at most 1s", elapsed)
	}
	for _, want := range []error{hookErr, context.DeadlineExceeded, tracingErr} {
		if !errors.Is(err, want) {
			t.Errorf("shutdownRouterProcess() error = %v, want errors.Is(_, %v)", err, want)
		}
	}
}

func TestShutdownRouterComponentsDrainsAcceptedManagementRequestBeforeResources(t *testing.T) {
	requestAccepted := make(chan struct{})
	releaseRequest := make(chan struct{})
	managementServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		close(requestAccepted)
		<-releaseRequest
		w.WriteHeader(http.StatusNoContent)
	}))
	defer managementServer.Close()
	defer closeTestSignal(releaseRequest)
	shutdownStarted := make(chan struct{})
	managementServer.Config.RegisterOnShutdown(func() { close(shutdownStarted) })

	requestDone := startManagementRequest(managementServer)
	waitForTestSignal(t, requestAccepted, "management request was not accepted")

	servingStopped := make(chan struct{})
	resourcesClosed := make(chan struct{})
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	shutdownDone := make(chan error, 1)
	go func() {
		shutdownDone <- shutdownRouterComponents(
			ctx,
			managementServer.Config.Shutdown,
			func(context.Context) error {
				close(resourcesClosed)
				return nil
			},
			nil,
			func(context.Context) error { return nil },
			func(context.Context) error {
				close(servingStopped)
				return nil
			},
		)
	}()

	waitForTestSignal(t, shutdownStarted, "management shutdown did not start")
	waitForTestSignal(t, servingStopped, "serving endpoint shutdown did not start")
	requireNoTestSignal(
		t,
		resourcesClosed,
		50*time.Millisecond,
		"resources closed before the accepted management request completed",
	)
	close(releaseRequest)

	if requestErr := waitForTestError(t, requestDone, "management request did not complete after release"); requestErr != nil {
		t.Fatalf("management request failed: %v", requestErr)
	}
	if shutdownErr := waitForTestError(t, shutdownDone, "shutdown did not complete after the management request was released"); shutdownErr != nil {
		t.Fatalf("shutdownRouterComponents() error = %v", shutdownErr)
	}
	waitForTestSignal(t, resourcesClosed, "resources remained open after management drain")
}

func TestShutdownRouterComponentsDoesNotCloseResourcesUnderTimedOutManagementRequest(t *testing.T) {
	requestAccepted := make(chan struct{})
	releaseRequest := make(chan struct{})
	managementServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		close(requestAccepted)
		<-releaseRequest
		w.WriteHeader(http.StatusNoContent)
	}))
	defer managementServer.Close()
	defer closeTestSignal(releaseRequest)

	requestDone := startManagementRequest(managementServer)
	waitForTestSignal(t, requestAccepted, "management request was not accepted")

	servingStopped := make(chan struct{})
	resourcesClosed := make(chan struct{})
	shutdownHookRan := make(chan struct{})
	shutdownHooks := []func(context.Context) error{
		func(context.Context) error {
			close(shutdownHookRan)
			return nil
		},
	}
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	shutdownDone := make(chan error, 1)
	go func() {
		shutdownDone <- shutdownRouterComponents(
			ctx,
			managementServer.Config.Shutdown,
			func(context.Context) error {
				close(resourcesClosed)
				return nil
			},
			&shutdownHooks,
			func(context.Context) error { return nil },
			func(context.Context) error {
				close(servingStopped)
				return nil
			},
		)
	}()

	waitForTestSignal(t, servingStopped, "serving endpoint shutdown did not start")
	shutdownErr := waitForTestError(t, shutdownDone, "shutdown did not return after its deadline")
	if !errors.Is(shutdownErr, context.DeadlineExceeded) {
		t.Fatalf("shutdownRouterComponents() error = %v, want deadline exceeded", shutdownErr)
	}
	requireNoTestSignal(
		t,
		resourcesClosed,
		50*time.Millisecond,
		"resources closed while the timed-out management request was still running",
	)
	requireNoTestSignal(
		t,
		shutdownHookRan,
		50*time.Millisecond,
		"shutdown hook ran while the timed-out management request was still running",
	)
	close(releaseRequest)
	_ = waitForTestError(t, requestDone, "management request did not exit after release")
}

func startManagementRequest(server *httptest.Server) <-chan error {
	done := make(chan error, 1)
	go func() {
		response, err := server.Client().Get(server.URL)
		if err == nil {
			_ = response.Body.Close()
		}
		done <- err
	}()
	return done
}

func closeTestSignal(signal chan struct{}) {
	select {
	case <-signal:
	default:
		close(signal)
	}
}

func waitForTestSignal(t *testing.T, signal <-chan struct{}, failure string) {
	t.Helper()
	select {
	case <-signal:
	case <-time.After(time.Second):
		t.Fatal(failure)
	}
}

func requireNoTestSignal(t *testing.T, signal <-chan struct{}, duration time.Duration, failure string) {
	t.Helper()
	select {
	case <-signal:
		t.Fatal(failure)
	case <-time.After(duration):
	}
}

func waitForTestError(t *testing.T, result <-chan error, failure string) error {
	t.Helper()
	select {
	case err := <-result:
		return err
	case <-time.After(time.Second):
		t.Fatal(failure)
		return nil
	}
}

func waitForProcessListener(t *testing.T, done <-chan error, address string) {
	t.Helper()
	startupDeadline := time.Now().Add(5 * time.Second)
	for {
		select {
		case err := <-done:
			t.Fatalf("Router exited before ExtProc became ready: %v", err)
		default:
		}
		connection, err := net.DialTimeout("tcp", address, 25*time.Millisecond)
		if err == nil {
			_ = connection.Close()
			return
		}
		if time.Now().After(startupDeadline) {
			t.Fatal("ExtProc listener did not become ready")
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func requireListenerReleased(t *testing.T, name, address string) {
	t.Helper()
	listener, err := net.Listen("tcp", address)
	if err != nil {
		t.Fatalf("%s listener remained open after shutdown: %v", name, err)
	}
	_ = listener.Close()
}
