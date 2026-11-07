package classification

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/admission"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

func remoteTaskBinding[I, O any](ctx context.Context, models *classifierModelRuntime, spec config.ResolvedModelBinding, closer io.Closer, infer func(context.Context, I) (O, error), validate func(I, O) error) (*binding.Resolved[I, O], error) {
	if models == nil {
		models = standaloneModelRuntime()
	}
	if spec.Deployment.Input.MaxTokens != 0 || (spec.Deployment.Input.Overflow != "" && spec.Deployment.Input.Overflow != "reject") {
		_ = closer.Close()
		return nil, fmt.Errorf("remote adapter cannot enforce a local tokenizer input budget")
	}
	// Per-binding timeouts, byte limits and catalog aliases do not create a
	// second physical deployment or a second admission allowance; credentials
	// participate without appearing in the resource key or diagnostics.
	endpoint, selectedModel, identityErr := remoteOperationIdentity(spec.Binding.Adapter, spec.Deployment)
	if identityErr != nil {
		_ = closer.Close()
		return nil, identityErr
	}
	description, _ := json.Marshal(struct {
		Endpoint   string
		Model      string
		Credential string
	}{endpoint, selectedModel, credentialFingerprint(spec.Deployment)})
	fingerprint := sha256.Sum256(description)
	// A declared local revision does not change the remote deployment actually
	// addressed by the connector, so it cannot create another admission gate.
	identity := binding.ResourceIdentity{Artifact: hex.EncodeToString(fingerprint[:]), Provider: "http", Device: "external", Precision: "external"}
	budget, _ := json.Marshal(spec.Admission)
	var gate admission.Admissioner = admission.Noop{}
	if spec.Admission.MaxConcurrency > 0 {
		gate = admission.NewSemaphore(spec.Admission.MaxConcurrency, spec.Admission.MaxQueue, time.Duration(spec.Admission.QueueTimeoutMs)*time.Millisecond, admission.Overflow(spec.Admission.OnOverflow))
	}
	// The reference shares only the deployment's admission gate; the binding
	// owns its connector.
	resource, err := models.runtime.Pool.Admit(ctx, identity, string(budget), gate)
	if err != nil {
		_ = closer.Close()
		return nil, err
	}
	if err = resource.Own(closer); err != nil {
		_ = closer.Close()
		_ = resource.Close()
		return nil, err
	}
	registry := binding.NewRegistry(models.runtime.ObserveBinding)
	task, err := binding.Register(registry, spec.Binding.Contract, func(I) error { return nil }, validate)
	if err != nil {
		_ = resource.Close()
		return nil, err
	}
	bound, err := task.Resolve(binding.Identity{Recipe: string(spec.Recipe), Name: spec.Name, Deployment: spec.Binding.Deployment, Contract: spec.Binding.Contract, Adapter: spec.Binding.Adapter, Head: spec.Binding.Head}, binding.Capability{Contract: spec.Binding.Contract, Provider: "http", Device: "external", Precision: "external"}, resource, func(ctx context.Context, _ io.Closer, input I) (O, error) { return infer(ctx, input) })
	if err != nil {
		_ = resource.Close()
		return nil, err
	}
	bound.Ready()
	return bound, nil
}
