// Package principal says on whose behalf an upstream call is made. A call
// made for a client request acts as the requesting user and may present the
// delegated credential the authenticator issued for that request; every other
// call acts as the Router. An upstream whose auth is credential: delegated
// takes only the first kind, and only with a credential.
package principal

import (
	"context"
	"errors"
	"sync/atomic"
)

// ErrCredentialMissing refuses a call made for a client request to a
// delegated upstream when the request carries no delegated credential. The
// caller answers the client 401 instead of making the call.
var ErrCredentialMissing = errors.New("the upstream takes the requesting user's delegated credential, which the request does not carry")

// ErrRouterCall refuses a Router-own call to a delegated upstream: it acts as
// the Router and has no user's credential to present.
var ErrRouterCall = errors.New("the upstream takes the requesting user's delegated credential, which a Router-own call does not carry")

type key struct{}

// request is the principal of the calls made for one client request.
type request struct {
	credential string
	// refused records that a call of the request was refused for want of a
	// credential, so the request answers 401 even where the call's own error
	// is absorbed, as a signal's is.
	refused atomic.Bool
}

// ForRequest marks the calls made with ctx as made for a client request whose
// delegated credential is credential; empty when the request carries none.
func ForRequest(ctx context.Context, credential string) context.Context {
	return context.WithValue(ctx, key{}, &request{credential: credential})
}

// ForRouter marks the calls made with ctx as the Router's own, whatever
// request ctx derives from.
func ForRouter(ctx context.Context) context.Context {
	return context.WithValue(ctx, key{}, (*request)(nil))
}

// Inherit gives detached the principal of from, for a call detached from the
// request's lifetime that is still made for it.
func Inherit(detached, from context.Context) context.Context {
	if detached == nil || from == nil {
		return detached
	}
	if value, ok := from.Value(key{}).(*request); ok {
		return context.WithValue(detached, key{}, value)
	}
	return detached
}

func requestOf(ctx context.Context) *request {
	if ctx == nil {
		return nil
	}
	value, _ := ctx.Value(key{}).(*request)
	return value
}

// ForUser reports whether the calls made with ctx are made for a client request.
func ForUser(ctx context.Context) bool {
	return requestOf(ctx) != nil
}

// RequestCredential returns the delegated credential of the request ctx's
// calls are made for, and whether they are made for one, without refusing
// anything: a request hands it to the calls it dispatches itself.
func RequestCredential(ctx context.Context) (string, bool) {
	value := requestOf(ctx)
	if value == nil {
		return "", false
	}
	return value.credential, true
}

// Delegated returns the credential a call made with ctx presents to a
// delegated upstream: the request's, ErrCredentialMissing when the request
// carries none, or ErrRouterCall for a Router-own call.
func Delegated(ctx context.Context) (string, error) {
	value := requestOf(ctx)
	if value == nil {
		return "", ErrRouterCall
	}
	if value.credential == "" {
		value.refused.Store(true)
		return "", ErrCredentialMissing
	}
	return value.credential, nil
}

// Refused reports whether a call made for ctx's request was refused because
// the request carries no delegated credential.
func Refused(ctx context.Context) bool {
	value := requestOf(ctx)
	return value != nil && value.refused.Load()
}
