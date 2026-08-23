package managementauthorization

import (
	"context"
	"errors"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementpermission"
)

type Snapshot struct {
	Principal       accesscontrol.ManagementPrincipal
	RoleGrants      []RoleGrant
	TeamGrants      []TeamGrant
	AuthorityDigest string
}

type SnapshotLoader interface {
	Load(
		context.Context,
		accesscontrol.ManagementPrincipalID,
		accesscontrol.NamespaceID,
	) (Snapshot, error)
}

type Request struct {
	PrincipalID   accesscontrol.ManagementPrincipalID
	NamespaceID   accesscontrol.NamespaceID
	Permission    managementpermission.Expression
	Targets       map[string][]accesscontrol.ScopedTarget
	Conditions    map[string]bool
	SpecialAuth   map[string]bool
	Recorded      map[string]bool
	Authenticated bool
}

type Decision struct {
	AuthorityDigest string
}

// Runtime is the shared authorization core used by every Management domain.
// Operation-specific route adapters resolve resource ownership into Targets;
// this layer loads current authority and evaluates the registered permission
// expression without trusting caller-provided roles or claims.
type Runtime struct {
	Loader SnapshotLoader
}

func (runtime Runtime) Authorize(ctx context.Context, request Request) (Decision, error) {
	if runtime.Loader == nil {
		return Decision{}, errors.New("management authorization loader is unavailable")
	}
	snapshot, err := runtime.Loader.Load(ctx, request.PrincipalID, request.NamespaceID)
	if err != nil {
		return Decision{}, err
	}
	if snapshot.Principal.ID != request.PrincipalID || snapshot.AuthorityDigest == "" {
		return Decision{}, ErrInvalidContext
	}
	err = Evaluate(request.Permission, EvaluationContext{
		Authenticated: request.Authenticated,
		RoleGrants:    snapshot.RoleGrants,
		TeamGrants:    snapshot.TeamGrants,
		Targets:       request.Targets,
		Conditions:    request.Conditions,
		SpecialAuth:   request.SpecialAuth,
		Recorded:      request.Recorded,
	})
	if err != nil {
		return Decision{}, err
	}
	return Decision{AuthorityDigest: snapshot.AuthorityDigest}, nil
}
