//go:build !windows && cgo

package apiserver

import (
	"context"
	"net/http"
)

// ManagedAPI is the explicit composition seam for Router-native control-plane
// resources. The process composition root owns storage and background
// lifecycles; this listener only mounts routes and includes their aggregate
// readiness in the Router readiness contract.
type ManagedAPI interface {
	Register(*http.ServeMux)
	Ready(context.Context) error
}
