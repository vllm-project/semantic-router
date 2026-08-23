package accessruntime

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
)

// Runtime owns the process-local identity and stable service seams for one
// inference access runtime. Operation-specific behavior lives in the adjacent
// authentication, authorization, admission, dispatch, and settlement modules.
type Runtime struct {
	reader             ProjectionReader
	engine             AtomicEngine
	peppers            map[accesscredential.Kind]accesscredential.PepperKeyring
	keyPrefix          string
	delegationAudience string
	delegationBarriers managementauth.DelegationRevocationBarrierStore
	identity           *runtimeIdentity
}

func New(options RuntimeOptions) (*Runtime, error) {
	if options.Reader == nil {
		return nil, fmt.Errorf("access projection reader is required")
	}
	if options.Engine == nil {
		return nil, fmt.Errorf("atomic access and quota engine is required")
	}
	if err := options.APIKeyPeppers.Validate(); err != nil {
		return nil, fmt.Errorf("API-key pepper keyring: %w", err)
	}
	if err := options.DelegationPeppers.Validate(); err != nil {
		return nil, fmt.Errorf("delegation pepper keyring: %w", err)
	}
	if strings.TrimSpace(options.DelegationAudience) == "" || options.DelegationAudience != strings.TrimSpace(options.DelegationAudience) {
		return nil, fmt.Errorf("delegation audience is required and must be canonical")
	}
	if options.DelegationBarriers == nil {
		return nil, fmt.Errorf("delegation revocation barriers are required")
	}
	if _, err := quotaruntime.NewAccessProjectionKeyspaceWithPrefix(options.KeyPrefix, "validation"); err != nil {
		return nil, fmt.Errorf("access runtime key prefix: %w", err)
	}
	return &Runtime{
		reader: options.Reader, engine: options.Engine,
		peppers: map[accesscredential.Kind]accesscredential.PepperKeyring{
			accesscredential.KindAPIKey:     options.APIKeyPeppers,
			accesscredential.KindDelegation: options.DelegationPeppers,
		},
		keyPrefix: options.KeyPrefix, delegationAudience: options.DelegationAudience,
		delegationBarriers: options.DelegationBarriers,
		identity:           &runtimeIdentity{marker: 1},
	}, nil
}
