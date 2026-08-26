//go:build !windows && cgo

package apiserver

import (
	"context"
	"net/http"
)

// ManagementAPI is the explicit composition seam for Router-native Management
// resources. The process composition root owns storage and background
// lifecycles; this listener only mounts routes and includes their aggregate
// readiness in the Router readiness contract.
type ManagementAPI interface {
	Register(*http.ServeMux)
	Ready(context.Context) error
}

// RuntimeReadiness is the process-owned serving contract behind /ready. It is
// independent of whether the optional Management API surface is enabled.
type RuntimeReadiness interface {
	Ready(context.Context) error
}
