package agentmanagement

import (
	"context"

	"github.com/google/uuid"
)

func (service *Service) GetArtifactMetadata(
	ctx context.Context, namespaceID, artifactID string, access AccessContext,
) (Artifact, error) {
	artifact, err := service.getAuthorizedArtifact(ctx, namespaceID, artifactID, access)
	if err != nil {
		return Artifact{}, err
	}
	clear(artifact.Content)
	artifact.Content = nil
	return artifact, nil
}

func (service *Service) ResolveArtifactAccess(
	ctx context.Context, namespaceID, artifactID string,
) (ArtifactAccess, error) {
	if uuid.Validate(namespaceID) != nil || uuid.Validate(artifactID) != nil {
		return ArtifactAccess{}, ErrInvalid
	}
	artifact, err := service.store.GetArtifact(ctx, namespaceID, artifactID)
	if err != nil || !service.now().UTC().Before(artifact.ExpiresAt) {
		if err != nil {
			return ArtifactAccess{}, err
		}
		return ArtifactAccess{}, ErrNotFound
	}
	session, err := service.store.GetSession(ctx, namespaceID, artifact.SessionID)
	if err != nil {
		return ArtifactAccess{}, err
	}
	return ArtifactAccess{ID: artifact.ID, Session: SessionAccess{
		ID: session.ID, OwnerPrincipalID: session.OwnerPrincipalID,
		EffectiveUserID: session.EffectiveUserID, EffectiveTeamID: session.EffectiveTeamID,
	}}, nil
}

func (service *Service) GetArtifactContent(
	ctx context.Context, namespaceID, artifactID string, access AccessContext,
) (ArtifactContent, error) {
	artifact, err := service.getAuthorizedArtifact(ctx, namespaceID, artifactID, access)
	if err != nil {
		return ArtifactContent{}, err
	}
	return ArtifactContent{
		ID: artifact.ID, MediaType: artifact.MediaType,
		Encoding: "base64", Content: append([]byte(nil), artifact.Content...), Digest: artifact.Digest,
	}, nil
}

func (service *Service) getAuthorizedArtifact(
	ctx context.Context, namespaceID, artifactID string, access AccessContext,
) (Artifact, error) {
	if uuid.Validate(namespaceID) != nil || uuid.Validate(artifactID) != nil {
		return Artifact{}, ErrInvalid
	}
	if err := validateAccessContext(namespaceID, access); err != nil {
		return Artifact{}, err
	}
	artifact, err := service.store.GetArtifact(ctx, namespaceID, artifactID)
	if err != nil {
		return Artifact{}, err
	}
	if !service.now().UTC().Before(artifact.ExpiresAt) {
		return Artifact{}, ErrNotFound
	}
	session, err := service.store.GetSession(ctx, namespaceID, artifact.SessionID)
	if err != nil {
		return Artifact{}, err
	}
	if !accessCanReadSession(access, session) {
		return Artifact{}, ErrDenied
	}
	return artifact, nil
}

func (service *Service) RequestCancellation(
	ctx context.Context, namespaceID, sessionID, turnID string, access AccessContext,
) (Turn, bool, error) {
	if uuid.Validate(namespaceID) != nil || uuid.Validate(sessionID) != nil ||
		uuid.Validate(turnID) != nil {
		return Turn{}, false, ErrInvalid
	}
	if err := validateAccessContext(namespaceID, access); err != nil {
		return Turn{}, false, err
	}
	session, err := service.store.GetSession(ctx, namespaceID, sessionID)
	if err != nil || !accessCanReadSession(access, session) {
		if err != nil {
			return Turn{}, false, err
		}
		return Turn{}, false, ErrDenied
	}
	turn, replayed, err := service.store.RequestCancellation(
		ctx, namespaceID, sessionID, turnID, service.now().UTC(),
	)
	if err != nil || replayed {
		return turn, replayed, err
	}
	if service.notifier != nil {
		// Notification is acceleration only. The durable cancellation flag was
		// committed above and a worker observes it even when fan-out fails.
		_ = service.notifier.NotifyCancellation(ctx, namespaceID, turnID)
	}
	return turn, false, nil
}

func (service *Service) ResolvePublicationAccess(
	ctx context.Context, namespaceID, planID string,
) (PublicationAccess, error) {
	if uuid.Validate(namespaceID) != nil || uuid.Validate(planID) != nil {
		return PublicationAccess{}, ErrInvalid
	}
	plan, err := service.store.GetPublicationPlan(ctx, namespaceID, planID)
	if err != nil {
		return PublicationAccess{}, err
	}
	session, err := service.store.GetSession(ctx, namespaceID, plan.SessionID)
	if err != nil {
		return PublicationAccess{}, err
	}
	models, err := service.store.GetPublicationModelIDs(ctx, namespaceID, planID)
	if err != nil {
		return PublicationAccess{}, err
	}
	return PublicationAccess{
		PlanID: plan.ID, SessionID: plan.SessionID, RecipeID: plan.RecipeID,
		EntrypointID: plan.EntrypointID, ModelIDs: models, Revision: plan.Revision,
		Digest: plan.Digest, ExpiresAt: plan.ExpiresAt,
		Session: SessionAccess{
			ID: session.ID, OwnerPrincipalID: session.OwnerPrincipalID,
			EffectiveUserID: session.EffectiveUserID, EffectiveTeamID: session.EffectiveTeamID,
		},
	}, nil
}
