// Package binding prepares typed task handles and owns the model resources
// they use. Providers interpret artifacts; recipes only see resolved tasks.
package binding

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
)

var (
	ErrClosed        = errors.New("model binding is closed")
	ErrCapability    = errors.New("model capability mismatch")
	ErrInputLimit    = errors.New("model input exceeds task budget")
	ErrInvalidInput  = errors.New("model task input is invalid")
	ErrInvalidResult = errors.New("model returned an invalid task result")
)

// ResourceIdentity describes physical execution, independently of a recipe or
// task head. A provider may omit a head only when it actually shares the same
// immutable backbone. Execution must include any weight-changing adapters and
// other settings that affect resource compatibility. Paths alone are not keys.
type ResourceIdentity struct {
	Artifact  string
	Revision  string
	Provider  string
	Device    string
	Precision string
	Execution string
}

func (i ResourceIdentity) Key() (string, error) {
	if i.Artifact == "" || i.Provider == "" || i.Device == "" || i.Precision == "" {
		return "", fmt.Errorf("%w: artifact, provider, device and precision are required", ErrCapability)
	}
	data, err := json.Marshal(i)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), nil
}

// Identity names one task binding. Recipe and Name are never part of the
// physical resource key, and sharing a resource grants no symbol visibility.
type Identity struct {
	Recipe     string
	Name       string
	Deployment string
	Contract   string
	Adapter    string
	Head       string
}

// Limits separates architectural capacity, the implemented task limit and
// the operator's budget. Zero means unknown/unset, never unlimited capability.
type Limits struct {
	ModelTokens      int
	TaskTokens       int
	DeploymentTokens int
	Overflow         string
}

func (l Limits) EffectiveTokens() int {
	limit := 0
	for _, n := range []int{l.ModelTokens, l.TaskTokens, l.DeploymentTokens} {
		if n > 0 && (limit == 0 || n < limit) {
			limit = n
		}
	}
	return limit
}

func (l Limits) Validate() error {
	if l.ModelTokens < 0 || l.TaskTokens < 0 || l.DeploymentTokens < 0 {
		return fmt.Errorf("%w: token limits must not be negative", ErrCapability)
	}
	switch l.Overflow {
	case "", "reject", "truncate", "window":
	default:
		return fmt.Errorf("%w: unsupported overflow policy %q", ErrCapability, l.Overflow)
	}
	if l.DeploymentTokens > 0 && l.TaskTokens > 0 && l.DeploymentTokens > l.TaskTokens {
		return fmt.Errorf("%w: deployment budget %d exceeds task limit %d", ErrCapability, l.DeploymentTokens, l.TaskTokens)
	}
	if l.DeploymentTokens > 0 && l.ModelTokens > 0 && l.DeploymentTokens > l.ModelTokens {
		return fmt.Errorf("%w: deployment budget exceeds model capacity", ErrCapability)
	}
	return nil
}

// CheckInput receives the provider's actual tokenizer count, including its
// template and reserved tokens. It does not estimate tokens from text length.
func (l Limits) CheckInput(tokens int) error {
	if tokens < 0 {
		return fmt.Errorf("%w: negative token count", ErrInputLimit)
	}
	if max := l.EffectiveTokens(); max > 0 && tokens > max && (l.Overflow == "" || l.Overflow == "reject") {
		return fmt.Errorf("%w: got %d tokens, maximum %d", ErrInputLimit, tokens, max)
	}
	return nil
}

// Capability describes what a loaded provider can actually execute. Requested
// hardware must agree with effective hardware; CPU fallback is not implicit.
type Capability struct {
	Contract  string
	Provider  string
	Device    string
	Precision string
	Limits    Limits
	Labels    []string
	Embedding *EmbeddingCapability
}

// EmbeddingCapability describes actual vector semantics. Empty strings and
// zero dimension/layer mean unknown, not inferred from a checkpoint name.
type EmbeddingCapability struct {
	Dimension     int
	Layer         int
	Pooling       string
	Normalization string
	Modalities    []string
}

func cloneCapability(capability Capability) Capability {
	capability.Labels = append([]string(nil), capability.Labels...)
	if capability.Embedding != nil {
		value := *capability.Embedding
		value.Modalities = append([]string(nil), value.Modalities...)
		capability.Embedding = &value
	}
	return capability
}

func (c Capability) Validate(id Identity) error {
	if id.Recipe == "" || id.Name == "" || id.Deployment == "" || id.Adapter == "" {
		return fmt.Errorf("%w: binding recipe, name, deployment and adapter are required", ErrCapability)
	}
	if c.Contract == "" || c.Contract != id.Contract || c.Provider == "" || c.Device == "" || c.Precision == "" {
		return fmt.Errorf("%w: task contract and effective execution configuration must be explicit", ErrCapability)
	}
	return c.Limits.Validate()
}
