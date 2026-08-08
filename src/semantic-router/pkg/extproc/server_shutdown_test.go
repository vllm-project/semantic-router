package extproc

import (
	"fmt"
	"net"
	"sync"
	"testing"
	"time"
)

// TestServerStopWaitsForInFlightRequestBeforeReturning pins the guarantee the
// process-level signal handler (cmd/runtime_bootstrap.go) is built on: it runs
// Stop first and only then releases shared resources such as the vector store,
// which is only safe because Stop does not return while a request is still
// using them.
func TestServerStopWaitsForInFlightRequestBeforeReturning(t *testing.T) {
	server, startErrCh := startTestServer(t)

	// Stands in for an in-flight Process call: RouterService.Process holds
	// exactly this lease for the duration of the call.
	lease, err := server.service.acquireCurrentLease()
	if err != nil {
		t.Fatalf("acquireCurrentLease() error = %v", err)
	}

	stopped := make(chan struct{})
	go func() {
		server.Stop()
		close(stopped)
	}()

	select {
	case <-stopped:
		t.Fatal("Stop() returned while a request was still in flight")
	case <-time.After(100 * time.Millisecond):
	}

	lease.release()

	select {
	case <-stopped:
	case <-time.After(10 * time.Second):
		t.Fatal("Stop() did not return after the in-flight request finished")
	}

	// Start owns no signal handler anymore; Stop is the only thing that ends
	// it, so Start returning is the proof that contract holds.
	select {
	case err := <-startErrCh:
		if err != nil {
			t.Fatalf("Start() error = %v, want nil after Stop()", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("Start() did not return after Stop()")
	}
}

// TestServerStopIsIdempotentAndBlocksLateCallers covers the two callers the
// new ownership split creates: the signal handler's hook and Start's own
// post-serve path. A second call must neither re-run the teardown nor return
// before the first one has finished, since the signal handler treats Stop
// returning as "the drain is complete" before releasing shared resources.
func TestServerStopIsIdempotentAndBlocksLateCallers(t *testing.T) {
	server, startErrCh := startTestServer(t)

	lease, err := server.service.acquireCurrentLease()
	if err != nil {
		t.Fatalf("acquireCurrentLease() error = %v", err)
	}

	var wg sync.WaitGroup
	returned := make(chan int, 3)
	for i := 0; i < 3; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			server.Stop()
			returned <- id
		}(i)
	}

	select {
	case id := <-returned:
		t.Fatalf("Stop() call %d returned while a request was still in flight", id)
	case <-time.After(100 * time.Millisecond):
	}

	lease.release()

	allReturned := make(chan struct{})
	go func() {
		wg.Wait()
		close(allReturned)
	}()
	select {
	case <-allReturned:
	case <-time.After(10 * time.Second):
		t.Fatal("concurrent Stop() callers did not all return after the drain finished")
	}

	// A further sequential Stop must be a no-op rather than a second teardown
	// of an already-retired router.
	server.Stop()

	select {
	case err := <-startErrCh:
		if err != nil {
			t.Fatalf("Start() error = %v, want nil after Stop()", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("Start() did not return after Stop()")
	}
}

// TestServerStartDoesNotServeAfterStop covers the startup-window ordering the
// signal handler makes reachable: Stop can land between NewServer and Start,
// since warmup sits between them in main. Start must then decline to serve
// rather than open a socket shutdown has already finished with.
func TestServerStartDoesNotServeAfterStop(t *testing.T) {
	port, err := freeTCPPort()
	if err != nil {
		t.Fatalf("failed to allocate a free port: %v", err)
	}
	server := &Server{service: NewRouterService(&OpenAIRouter{}), port: port}

	server.Stop()

	if err := server.Start(); err != nil {
		t.Fatalf("Start() error = %v, want nil after a completed Stop()", err)
	}
	if portListening(port) {
		t.Fatal("Start() began serving after Stop() had already completed")
	}
}

// startTestServer runs a real gRPC ExtProc server on a free port and returns
// once it is accepting connections, along with the channel carrying Start's
// eventual return value.
func startTestServer(t *testing.T) (*Server, <-chan error) {
	t.Helper()

	port, err := freeTCPPort()
	if err != nil {
		t.Fatalf("failed to allocate a free port: %v", err)
	}

	server := &Server{service: NewRouterService(&OpenAIRouter{}), port: port}
	startErrCh := make(chan error, 1)
	go func() { startErrCh <- server.Start() }()
	t.Cleanup(server.Stop)

	if !waitForPortListening(port, 10*time.Second) {
		t.Fatalf("server did not start listening on port %d", port)
	}
	return server, startErrCh
}

func freeTCPPort() (int, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	defer func() { _ = l.Close() }()
	return l.Addr().(*net.TCPAddr).Port, nil
}

// waitForPortListening polls a real, observable condition (the port accepting
// connections) instead of sleeping a guessed duration.
func waitForPortListening(port int, timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if portListening(port) {
			return true
		}
		time.Sleep(10 * time.Millisecond)
	}
	return false
}

func portListening(port int) bool {
	conn, err := net.DialTimeout("tcp", fmt.Sprintf("127.0.0.1:%d", port), 100*time.Millisecond)
	if err != nil {
		return false
	}
	_ = conn.Close()
	return true
}
