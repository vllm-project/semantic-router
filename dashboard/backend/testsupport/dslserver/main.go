// The browser fixture serves the production compiler handlers without starting
// Dashboard services or loading a developer's configuration or credentials.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/handlers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

func main() {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		log.Fatal(err)
	}
	mux := http.NewServeMux()
	for _, operation := range []string{"compile", "validate", "parse", "decompile", "format"} {
		mux.HandleFunc("/api/dsl/"+operation, handlers.DSLEditorHandler(operation))
	}
	// Structural signals need no model deployment. Publish the real task
	// registry without inventing deployment readiness or model capabilities.
	mux.HandleFunc("GET /api/decision-model/tasks", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(modelservice.ProjectTaskCatalog(nil, nil))
	})
	server := &http.Server{Handler: mux, ReadHeaderTimeout: 5 * time.Second}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	go func() {
		<-ctx.Done()
		shutdown, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = server.Shutdown(shutdown)
	}()
	fmt.Println("http://" + listener.Addr().String())
	if err := server.Serve(listener); err != nil && err != http.ErrServerClosed {
		log.Fatal(err)
	}
}
