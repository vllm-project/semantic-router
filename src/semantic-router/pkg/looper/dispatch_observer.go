/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package looper

import "context"

// DispatchObserver is the accounting boundary around every physical model
// call made by a Mixture-of-Models algorithm. Started must complete before the
// HTTP request can leave the process; Completed is invoked exactly once for
// every successful Started call.
type DispatchObserver interface {
	Started(context.Context, DispatchStart) (DispatchAuthorization, error)
	Completed(context.Context, DispatchCompletion)
}

type DispatchStart struct {
	Model     string
	Iteration int
}

// DispatchAuthorization contains only the opaque values an authenticated
// internal Router hop needs before it can leave the process.
type DispatchAuthorization struct {
	DispatchID string
	Grant      string
	RequestID  string
}

type DispatchCompletion struct {
	DispatchID  string
	Model       string
	Iteration   int
	HTTPStarted bool
	FailureCode string
}

type dispatchObserverContextKey struct{}

// WithDispatchObserver binds accounting to one outer inference request. The
// value is process-local and is never serialized into internal request headers.
func WithDispatchObserver(ctx context.Context, observer DispatchObserver) context.Context {
	if observer == nil {
		return ctx
	}
	return context.WithValue(ctx, dispatchObserverContextKey{}, observer)
}

func dispatchObserverFromContext(ctx context.Context) DispatchObserver {
	if ctx == nil {
		return nil
	}
	observer, _ := ctx.Value(dispatchObserverContextKey{}).(DispatchObserver)
	return observer
}
