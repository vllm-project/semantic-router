package agentmanagement

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

type RegistrySource interface {
	Current(context.Context, string) (*ToolRegistry, error)
	Load(context.Context, string, string) (*ToolRegistry, error)
}

type ServiceOptions struct {
	Store            Store
	Notifier         TurnNotifier
	SessionAuthority SessionAuthority
	TargetVisibility TargetVisibility
	Definitions      DefinitionValidator
	SourcePolicies   ToolSourcePolicyValidator
	ToolSources      ToolSourceDiscoverer
	Registries       RegistrySource
	SecretCodec      SecretCodec
	CommandCodec     *managementcommand.Codec
	CursorKeyring    securitykeyring.Symmetric
	SessionTTL       time.Duration
	Now              func() time.Time
}

type Service struct {
	store            Store
	notifier         TurnNotifier
	sessionAuthority SessionAuthority
	targetVisibility TargetVisibility
	definitions      DefinitionValidator
	sourcePolicies   ToolSourcePolicyValidator
	toolSources      ToolSourceDiscoverer
	registries       RegistrySource
	secretCodec      SecretCodec
	commandCodec     *managementcommand.Codec
	codec            signedCodec
	sessionTTL       time.Duration
	now              func() time.Time
	closeOnce        sync.Once
}

const agentResourceCommandTTL = 7 * 24 * time.Hour

const (
	agentProfileResourceType = "agent_profile"
	agentSkillResourceType   = "agent_skill"
	// #nosec G101 -- this is a resource-type identifier, not a credential value.
	agentToolCredentialResourceType = "agent_tool_credential"
	agentToolSourceResourceType     = "agent_tool_source"
)

func NewService(options ServiceOptions) (*Service, error) {
	if options.Store == nil || options.SessionAuthority == nil || options.TargetVisibility == nil || options.Definitions == nil ||
		options.SourcePolicies == nil || options.ToolSources == nil ||
		options.Registries == nil || options.SecretCodec == nil || options.CommandCodec == nil {
		return nil, fmt.Errorf("%w: Agent service dependencies are incomplete", ErrInvalid)
	}
	codec, err := newSignedCodec(options.CursorKeyring)
	if err != nil {
		return nil, err
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	sessionTTL := options.SessionTTL
	if sessionTTL == 0 {
		sessionTTL = 8 * time.Hour
	}
	if sessionTTL < 5*time.Minute || sessionTTL > 24*time.Hour {
		codec.close()
		return nil, fmt.Errorf("%w: Agent session TTL is outside the supported range", ErrInvalid)
	}
	return &Service{
		store: options.Store, notifier: options.Notifier, sessionAuthority: options.SessionAuthority,
		targetVisibility: options.TargetVisibility,
		definitions:      options.Definitions, sourcePolicies: options.SourcePolicies, toolSources: options.ToolSources,
		registries:  options.Registries,
		secretCodec: options.SecretCodec, commandCodec: options.CommandCodec,
		codec: codec, sessionTTL: sessionTTL, now: now,
	}, nil
}

func (service *Service) Close() {
	if service != nil {
		service.closeOnce.Do(func() { service.codec.close() })
	}
}

func (service *Service) Ready(ctx context.Context) error {
	if service == nil || service.store == nil || service.registries == nil {
		return errors.New("agent service is unavailable")
	}
	return service.store.Ready(ctx)
}
