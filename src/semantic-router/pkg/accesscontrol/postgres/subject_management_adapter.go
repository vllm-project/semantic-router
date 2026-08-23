package postgres

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/subjectmanagement"
)

var _ subjectmanagement.Repository = (*subjectRepositoryAdapter)(nil)

// subjectRepositoryAdapter keeps the public Store surface stable while the
// application depends only on its narrow Repository contract.
type subjectRepositoryAdapter struct{ store *Store }

func NewSubjectRepository(store *Store) (subjectmanagement.Repository, error) {
	if store == nil || store.db == nil {
		return nil, subjectmanagement.ErrUnavailable
	}
	return &subjectRepositoryAdapter{store: store}, nil
}

func (adapter *subjectRepositoryAdapter) Ready(ctx context.Context, codec *managementcommand.Codec) error {
	return adapter.store.ReadySubjectManagement(ctx, codec)
}

func (adapter *subjectRepositoryAdapter) Replay(ctx context.Context, command managementcommand.Command) (subjectmanagement.MutationResult, bool, error) {
	return adapter.store.ReplaySubjectCommand(ctx, command)
}

func (adapter *subjectRepositoryAdapter) GetUser(ctx context.Context, namespaceID, userID string) (subjectmanagement.User, error) {
	return adapter.store.GetSubjectUser(ctx, namespaceID, userID)
}

func (adapter *subjectRepositoryAdapter) ListUsers(ctx context.Context, query subjectmanagement.UserQuery) (subjectmanagement.RepositoryPage[subjectmanagement.User], error) {
	return adapter.store.ListSubjectUsers(ctx, query)
}

func (adapter *subjectRepositoryAdapter) CreateUser(ctx context.Context, mutation subjectmanagement.CreateUserMutation) (subjectmanagement.MutationResult, error) {
	return adapter.store.CreateSubjectUser(ctx, mutation)
}

func (adapter *subjectRepositoryAdapter) UpdateUser(ctx context.Context, user subjectmanagement.User, expected uint64, actor subjectmanagement.Actor) (subjectmanagement.MutationResult, error) {
	return adapter.store.UpdateSubjectUser(ctx, user, expected, actor)
}

func (adapter *subjectRepositoryAdapter) DeleteUser(ctx context.Context, namespaceID, userID string, expected uint64, actor subjectmanagement.Actor) (subjectmanagement.MutationResult, error) {
	return adapter.store.DeleteSubjectUser(ctx, namespaceID, userID, expected, actor)
}

func (adapter *subjectRepositoryAdapter) ResolveTeamDefaults(ctx context.Context, namespaceID string) (subjectmanagement.TeamDefaults, error) {
	return adapter.store.ResolveSubjectTeamDefaults(ctx, namespaceID)
}

func (adapter *subjectRepositoryAdapter) GetTeam(ctx context.Context, namespaceID, teamID string) (subjectmanagement.Team, error) {
	return adapter.store.GetSubjectTeam(ctx, namespaceID, teamID)
}

func (adapter *subjectRepositoryAdapter) ListTeams(ctx context.Context, query subjectmanagement.TeamQuery) (subjectmanagement.RepositoryPage[subjectmanagement.Team], error) {
	return adapter.store.ListSubjectTeams(ctx, query)
}

func (adapter *subjectRepositoryAdapter) CreateTeam(ctx context.Context, mutation subjectmanagement.CreateTeamMutation) (subjectmanagement.MutationResult, error) {
	return adapter.store.CreateSubjectTeam(ctx, mutation)
}

func (adapter *subjectRepositoryAdapter) UpdateTeam(ctx context.Context, team subjectmanagement.Team, expected uint64, actor subjectmanagement.Actor) (subjectmanagement.MutationResult, error) {
	return adapter.store.UpdateSubjectTeam(ctx, team, expected, actor)
}

func (adapter *subjectRepositoryAdapter) DeleteTeam(ctx context.Context, namespaceID, teamID string, expected uint64, actor subjectmanagement.Actor) (subjectmanagement.MutationResult, error) {
	return adapter.store.DeleteSubjectTeam(ctx, namespaceID, teamID, expected, actor)
}

func (adapter *subjectRepositoryAdapter) GetMembership(ctx context.Context, namespaceID, teamID, userID string) (subjectmanagement.Membership, error) {
	return adapter.store.GetSubjectMembership(ctx, namespaceID, teamID, userID)
}

func (adapter *subjectRepositoryAdapter) ListUserMemberships(ctx context.Context, query subjectmanagement.MembershipQuery) (subjectmanagement.RepositoryPage[subjectmanagement.UserMembership], error) {
	return adapter.store.ListSubjectUserMemberships(ctx, query)
}

func (adapter *subjectRepositoryAdapter) ListTeamMembers(ctx context.Context, query subjectmanagement.MembershipQuery) (subjectmanagement.RepositoryPage[subjectmanagement.TeamMember], error) {
	return adapter.store.ListSubjectTeamMembers(ctx, query)
}

func (adapter *subjectRepositoryAdapter) PutMembership(ctx context.Context, mutation subjectmanagement.PutMembershipMutation) (subjectmanagement.MutationResult, error) {
	return adapter.store.PutSubjectMembership(ctx, mutation)
}

func (adapter *subjectRepositoryAdapter) UpdateMembership(ctx context.Context, membership subjectmanagement.Membership, expected uint64, actor subjectmanagement.Actor) (subjectmanagement.MutationResult, error) {
	return adapter.store.UpdateSubjectMembership(ctx, membership, expected, actor)
}

func (adapter *subjectRepositoryAdapter) DeleteMembership(ctx context.Context, namespaceID, teamID, userID string, expected uint64, actor subjectmanagement.Actor) (subjectmanagement.MutationResult, error) {
	return adapter.store.DeleteSubjectMembership(ctx, namespaceID, teamID, userID, expected, actor)
}
