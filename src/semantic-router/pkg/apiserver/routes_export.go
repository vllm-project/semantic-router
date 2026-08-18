//go:build !windows && cgo

package apiserver

// This file exposes read-only exports over the route catalog for tooling that
// needs API surface information without starting the router (e.g. docs
// generation in tools/openapi-gen). The runtime endpoint at /openapi.json and
// this export share the same generateOpenAPISpec(), so generated docs cannot
// diverge from what the server serves.

// ExportedRoute is the public, documentation-facing view of one API route.
type ExportedRoute struct {
	Path        string
	Method      string
	Description string
}

// ExportedRoutes returns the route catalog metadata in registration order.
func ExportedRoutes() []ExportedRoute {
	routes := apiRoutes()
	out := make([]ExportedRoute, 0, len(routes))
	for _, route := range routes {
		out = append(out, ExportedRoute{
			Path:        route.Path,
			Method:      route.Method,
			Description: route.Description,
		})
	}
	return out
}

// ExportOpenAPISpec generates the OpenAPI 3.0 specification from the route
// catalog without requiring a running server.
func ExportOpenAPISpec() OpenAPISpec {
	return (&ClassificationAPIServer{}).generateOpenAPISpec()
}
