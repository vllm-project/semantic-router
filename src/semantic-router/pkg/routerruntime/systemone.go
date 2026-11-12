package routerruntime

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/systemone"
)

// AcquireSystemOne retains the frontend snapshot and native routing pipeline
// atomically. A reload cannot mix listener grants, classifiers and model clients.
func (r *Registry) AcquireSystemOne() (*configsnapshot.Snapshot, systemone.Router, func(), bool) {
	if r == nil {
		return nil, nil, nil, false
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	if r.configSnapshot == nil || r.acquireGeneration == nil {
		return nil, nil, nil, false
	}
	release, ok := r.acquireGeneration()
	if !ok {
		return nil, nil, nil, false
	}
	return r.configSnapshot, r.nativeRouter, release, true
}
