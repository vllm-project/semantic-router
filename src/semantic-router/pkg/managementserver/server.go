package managementserver

import (
	"context"
	"fmt"
	"net/http"
	"reflect"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
)

// Runtime is the lifecycle seam owned by the managed control plane.
type Runtime interface {
	Run(context.Context) error
	Ready(context.Context) error
}

// RouteRegistrar mounts one narrow Management domain without making Server a
// dependency container for that domain.
type RouteRegistrar interface {
	Register(*http.ServeMux)
}

type readinessProbe interface {
	Ready(context.Context) error
}

// Server is the narrow managed-API composition injected into the Router's
// existing management listener.
type Server struct {
	routes  []RouteRegistrar
	runtime Runtime
}

func NewServer(runtime Runtime, routes ...RouteRegistrar) (*Server, error) {
	if runtime == nil || len(routes) == 0 {
		return nil, fmt.Errorf("management runtime and at least one route registrar are required")
	}
	for index, registrar := range routes {
		if nilRouteRegistrar(registrar) {
			return nil, fmt.Errorf("management route registrar %d is nil", index)
		}
	}
	return &Server{routes: append([]RouteRegistrar(nil), routes...), runtime: runtime}, nil
}

func nilRouteRegistrar(registrar RouteRegistrar) bool {
	if registrar == nil {
		return true
	}
	value := reflect.ValueOf(registrar)
	switch value.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return value.IsNil()
	default:
		return false
	}
}

func (server *Server) Register(mux *http.ServeMux) {
	if server == nil || len(server.routes) == 0 || mux == nil {
		panic("Management server is not initialized")
	}
	managementMux := server.domainMux()
	mux.Handle(managementapi.BasePath+"/", newManagementTransport(managementMux))
}

func (server *Server) domainMux() *http.ServeMux {
	if server == nil || len(server.routes) == 0 {
		panic("Management server is not initialized")
	}
	mux := http.NewServeMux()
	for _, registrar := range server.routes {
		registrar.Register(mux)
	}
	return mux
}

func (server *Server) Ready(ctx context.Context) error {
	if server == nil || server.runtime == nil {
		return fmt.Errorf("provider Management runtime is unavailable")
	}
	if err := server.runtime.Ready(ctx); err != nil {
		return err
	}
	for index, registrar := range server.routes {
		probe, ok := registrar.(readinessProbe)
		if !ok {
			continue
		}
		if err := probe.Ready(ctx); err != nil {
			return fmt.Errorf("management route registrar %d is not ready: %w", index, err)
		}
	}
	return nil
}

func (server *Server) Run(ctx context.Context) error {
	if server == nil || server.runtime == nil {
		return fmt.Errorf("provider Management runtime is unavailable")
	}
	return server.runtime.Run(ctx)
}
