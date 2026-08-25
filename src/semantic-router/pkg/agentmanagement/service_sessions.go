package agentmanagement

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
)

func (service *Service) PrepareSession(
	ctx context.Context, namespaceID, principalID string, input SessionInput,
) (SessionAuthorization, error) {
	profile, err := service.sessionProfile(ctx, namespaceID, principalID, input, AccessContext{
		PrincipalID: principalID,
		Scope:       accesscontrol.ResultScope{NamespaceID: accesscontrol.NamespaceID(namespaceID), All: true},
	})
	if err != nil {
		return SessionAuthorization{}, err
	}
	return service.sessionAuthority.Prepare(ctx, SessionAuthorizationRequest{
		NamespaceID: namespaceID, PrincipalID: principalID,
		KeyID: input.KeyID, EffectiveTeamID: input.EffectiveTeamID, Profile: profile, Target: input.Target,
	})
}

func (service *Service) ResolveSessionAccess(
	ctx context.Context, namespaceID, sessionID string,
) (SessionAccess, error) {
	if uuid.Validate(namespaceID) != nil || uuid.Validate(sessionID) != nil {
		return SessionAccess{}, ErrInvalid
	}
	session, err := service.store.GetSession(ctx, namespaceID, sessionID)
	if err != nil {
		return SessionAccess{}, err
	}
	return SessionAccess{
		ID: session.ID, OwnerPrincipalID: session.OwnerPrincipalID,
		EffectiveUserID: session.EffectiveUserID, EffectiveTeamID: session.EffectiveTeamID,
	}, nil
}

func (service *Service) CreateSession(
	ctx context.Context, namespaceID, principalID, idempotencyKey string, input SessionInput,
	mutation MutationContext, access AccessContext,
) (Session, bool, error) {
	profile, err := service.sessionProfile(ctx, namespaceID, principalID, input, access)
	if err != nil {
		return Session{}, false, err
	}
	encoded, err := json.Marshal(input)
	if err != nil {
		return Session{}, false, ErrInvalid
	}
	now := service.now().UTC()
	command, err := service.commandCodec.Bind(
		managementcommand.NamespaceCommandScope(namespaceID), principalID,
		"/management/v1/agent-sessions", idempotencyKey, encoded, now, now.Add(7*24*time.Hour),
	)
	if err != nil {
		return Session{}, false, ErrInvalid
	}
	return service.sessionAuthority.Bootstrap(ctx, SessionBootstrapRequest{
		SessionID: uuid.NewString(), NamespaceID: namespaceID, PrincipalID: principalID,
		KeyID: input.KeyID, EffectiveTeamID: input.EffectiveTeamID, Profile: profile, Target: input.Target,
		Mode: input.Mode, Title: strings.TrimSpace(input.Title), SessionTTL: service.sessionTTL,
		Mutation: mutation, Command: command,
	})
}

func (service *Service) sessionProfile(
	ctx context.Context, namespaceID, principalID string, input SessionInput, access AccessContext,
) (Profile, error) {
	if uuid.Validate(namespaceID) != nil || uuid.Validate(principalID) != nil ||
		(input.ProfileID != "" && uuid.Validate(input.ProfileID) != nil) ||
		uuid.Validate(input.KeyID) != nil ||
		(input.EffectiveTeamID != "" && uuid.Validate(input.EffectiveTeamID) != nil) ||
		validateTarget(input.Target) != nil ||
		(input.Mode != SessionChat && input.Mode != SessionBuilder) || len(input.Title) > 256 {
		return Profile{}, ErrInvalid
	}
	if err := validateAccessContext(namespaceID, access); err != nil || access.PrincipalID != principalID {
		return Profile{}, ErrDenied
	}
	var (
		profile Profile
		err     error
	)
	if input.ProfileID == "" {
		profile, err = service.store.GetDefaultProfile(ctx, namespaceID, input.Mode)
	} else {
		profile, err = service.store.GetProfile(ctx, namespaceID, input.ProfileID)
	}
	if err != nil {
		return Profile{}, err
	}
	if profile.Status != StatusActive || !supportsMode(profile.SupportedModes, input.Mode) {
		return Profile{}, ErrNotFound
	}
	return profile, nil
}

func (service *Service) CreateTurn(
	ctx context.Context, namespaceID, sessionID, idempotencyKey string, input TurnInput, access AccessContext,
) (Turn, bool, error) {
	if uuid.Validate(namespaceID) != nil || uuid.Validate(sessionID) != nil ||
		ValidateTurnInput(input) != nil {
		return Turn{}, false, ErrInvalid
	}
	original, createTurnErr := json.Marshal(input)
	if createTurnErr != nil {
		return Turn{}, false, ErrInvalid
	}
	input, createTurnErr = NormalizeTurnInput(input)
	if createTurnErr != nil {
		return Turn{}, false, createTurnErr
	}
	if err := validateAccessContext(namespaceID, access); err != nil {
		return Turn{}, false, err
	}
	session, createTurnErr := service.store.GetSession(ctx, namespaceID, sessionID)
	if createTurnErr != nil {
		return Turn{}, false, createTurnErr
	}
	if session.Status != SessionActive || !accessCanReadSession(access, session) {
		return Turn{}, false, ErrDenied
	}
	currentProfile, createTurnErr := service.store.GetProfile(ctx, namespaceID, session.ProfileID)
	if createTurnErr != nil || currentProfile.Status != StatusActive {
		if createTurnErr != nil {
			return Turn{}, false, createTurnErr
		}
		return Turn{}, false, ErrDenied
	}
	profile, createTurnErr := service.store.GetProfileRevision(ctx, namespaceID, session.ProfileID, session.ProfileRevision)
	if createTurnErr != nil {
		return Turn{}, false, createTurnErr
	}
	if err := service.sessionAuthority.Reauthorize(ctx, session, profile.MinimumTargetCapabilities); err != nil {
		return Turn{}, false, err
	}
	registry, createTurnErr := service.registries.Current(ctx, namespaceID)
	if createTurnErr != nil {
		return Turn{}, false, createTurnErr
	}
	now := service.now().UTC()
	command, createTurnErr := service.commandCodec.Bind(
		managementcommand.NamespaceCommandScope(namespaceID), access.PrincipalID,
		"/management/v1/agent-sessions/"+sessionID+"/turns", idempotencyKey,
		original, now, now.Add(7*24*time.Hour),
	)
	if createTurnErr != nil {
		return Turn{}, false, ErrInvalid
	}
	turn := Turn{
		ID: uuid.NewString(), SessionID: sessionID, Status: TurnQueued,
		RegistryRevision: registry.Revision(), Input: input, Revision: 1, CreatedAt: now, UpdatedAt: now,
	}
	created, replayed, createTurnErr := service.store.CreateTurn(ctx, CreateTurnRequest{
		Turn: turn, NamespaceID: namespaceID, ActorPrincipalID: access.PrincipalID, Command: command,
	})
	if createTurnErr != nil {
		return Turn{}, false, createTurnErr
	}
	if !replayed && service.notifier != nil {
		if wakeErr := service.notifier.Wake(ctx, namespaceID, created.ID); wakeErr != nil {
			// The durable queued row remains discoverable by polling workers.
			_ = wakeErr
		}
	}
	return created, replayed, nil
}

func (service *Service) ListSessions(
	ctx context.Context, namespaceID string, request PageRequest, access AccessContext,
) (Page[Session], error) {
	if err := validateAccessContext(namespaceID, access); err != nil {
		return Page[Session]{}, err
	}
	if !access.Scope.All && request.OwnerPrincipalID == "" {
		request.OwnerPrincipalID = access.PrincipalID
	}
	query, err := service.listQuery(namespaceID, "sessions", request)
	if err != nil {
		return Page[Session]{}, err
	}
	result, err := service.store.ListSessions(ctx, namespaceID, query)
	return makePage(service, namespaceID, "sessions", query, request.PageSize,
		result.Items, result.HasMore, err)
}

func (service *Service) GetSession(
	ctx context.Context, namespaceID, sessionID string, access AccessContext,
) (Session, error) {
	if uuid.Validate(namespaceID) != nil || uuid.Validate(sessionID) != nil {
		return Session{}, ErrInvalid
	}
	if err := validateAccessContext(namespaceID, access); err != nil {
		return Session{}, err
	}
	session, err := service.store.GetSession(ctx, namespaceID, sessionID)
	if err != nil {
		return Session{}, err
	}
	if !accessCanReadSession(access, session) {
		return Session{}, ErrNotFound
	}
	return session, nil
}

func (service *Service) PatchSession(
	ctx context.Context, namespaceID, sessionID string, expected int64,
	patch SessionPatch, mutation MutationContext, access AccessContext,
) (Session, error) {
	if uuid.Validate(namespaceID) != nil || uuid.Validate(sessionID) != nil || expected < 1 ||
		(patch.Title == nil && patch.Status == nil) {
		return Session{}, ErrInvalid
	}
	if err := validateAccessContext(namespaceID, access); err != nil {
		return Session{}, err
	}
	session, err := service.store.GetSession(ctx, namespaceID, sessionID)
	if err != nil {
		return Session{}, err
	}
	if !accessCanReadSession(access, session) {
		return Session{}, ErrNotFound
	}
	if patch.Status == nil {
		return service.store.PatchSession(ctx, namespaceID, sessionID, expected, patch, mutation)
	}
	if *patch.Status != SessionClosed || session.Status != SessionActive {
		return Session{}, ErrInvalid
	}
	return service.sessionAuthority.Close(ctx, session, expected, patch, mutation)
}

func (service *Service) DeleteSession(
	ctx context.Context, namespaceID, sessionID string, expected int64,
	mutation MutationContext, access AccessContext,
) (int64, error) {
	if uuid.Validate(namespaceID) != nil || uuid.Validate(sessionID) != nil || expected < 1 {
		return 0, ErrInvalid
	}
	if err := validateAccessContext(namespaceID, access); err != nil {
		return 0, err
	}
	session, err := service.store.GetSession(ctx, namespaceID, sessionID)
	if err != nil {
		return 0, err
	}
	if !accessCanReadSession(access, session) {
		return 0, ErrNotFound
	}
	if session.Status != SessionClosed {
		return 0, ErrConflict
	}
	return service.store.DeleteSession(ctx, namespaceID, sessionID, expected, mutation)
}

func (service *Service) ListTurns(
	ctx context.Context, namespaceID, sessionID string, request PageRequest, access AccessContext,
) (Page[Turn], error) {
	if _, err := service.GetSession(ctx, namespaceID, sessionID, access); err != nil {
		return Page[Turn]{}, err
	}
	query, err := service.listQuery(namespaceID, "turns:"+sessionID, request)
	if err != nil {
		return Page[Turn]{}, err
	}
	result, err := service.store.ListTurns(ctx, namespaceID, sessionID, query)
	return makePage(service, namespaceID, "turns:"+sessionID, query, request.PageSize,
		result.Items, result.HasMore, err)
}

func (service *Service) ResumeEvents(
	ctx context.Context, namespaceID, sessionID string, after int64, limit int, access AccessContext,
) ([]Event, bool, error) {
	if uuid.Validate(namespaceID) != nil || uuid.Validate(sessionID) != nil ||
		after < 0 || limit < 1 || limit > 1000 {
		return nil, false, ErrInvalid
	}
	if err := validateAccessContext(namespaceID, access); err != nil {
		return nil, false, err
	}
	session, err := service.store.GetSession(ctx, namespaceID, sessionID)
	if err != nil {
		return nil, false, err
	}
	if !accessCanReadSession(access, session) {
		return nil, false, ErrDenied
	}
	oldest, err := service.store.OldestEventSequence(ctx, namespaceID, sessionID)
	if err != nil && !errors.Is(err, ErrNotFound) {
		return nil, false, err
	}
	if oldest > 1 && after+1 < oldest {
		checkpoint, checkpointErr := service.store.LatestCheckpoint(ctx, namespaceID, sessionID)
		if checkpointErr != nil {
			return nil, false, HistoryExpiredError{Recovery: HistoryRecovery{ThroughSequence: oldest - 1}}
		}
		return nil, false, HistoryExpiredError{Recovery: HistoryRecovery{
			CheckpointID: checkpoint.ID, ThroughSequence: checkpoint.ThroughSequence,
		}}
	}
	return service.store.ListEventsAfter(ctx, namespaceID, sessionID, after, limit)
}

// ListEventHistory returns the newest retained page on the first request and
// older pages thereafter. Items are always ascending so a client can append
// them to a transcript without reordering. The opaque cursor is bound to the
// namespace and evaluated result scope.
func (service *Service) ListEventHistory(
	ctx context.Context,
	namespaceID string,
	sessionID string,
	request EventPageRequest,
	access AccessContext,
) (Page[Event], error) {
	if uuid.Validate(namespaceID) != nil || uuid.Validate(sessionID) != nil ||
		request.PageSize < 1 || request.PageSize > 1000 {
		return Page[Event]{}, ErrInvalid
	}
	if err := validateAccessContext(namespaceID, access); err != nil {
		return Page[Event]{}, err
	}
	canonicalScope, err := request.Scope.Canonical()
	if err != nil || string(canonicalScope.NamespaceID) != namespaceID {
		return Page[Event]{}, ErrDenied
	}
	session, err := service.store.GetSession(ctx, namespaceID, sessionID)
	if err != nil {
		return Page[Event]{}, err
	}
	if !accessCanReadSession(access, session) {
		return Page[Event]{}, ErrDenied
	}
	scopeDigest, err := canonicalScope.Digest()
	if err != nil {
		return Page[Event]{}, ErrDenied
	}
	before := int64(0)
	if request.Cursor != "" {
		cursor, decodeErr := service.codec.decodeCursor(request.Cursor)
		if decodeErr != nil || cursor.NamespaceID != namespaceID || cursor.Kind != "events" ||
			cursor.ScopeDigest != scopeDigest {
			return Page[Event]{}, ErrInvalid
		}
		before = cursor.Sequence
	}
	items, hasMore, err := service.store.ListEventHistory(ctx, namespaceID, sessionID,
		EventHistoryQuery{BeforeSequence: before, Limit: request.PageSize})
	if err != nil {
		return Page[Event]{}, err
	}
	page := Page[Event]{Items: items, HasMore: hasMore}
	if hasMore && len(items) > 0 {
		page.NextCursor, err = service.codec.encodeCursor(cursorPayload{
			NamespaceID: namespaceID,
			Kind:        "events",
			ScopeDigest: scopeDigest,
			Sequence:    items[0].Sequence,
		})
	}
	return page, err
}
