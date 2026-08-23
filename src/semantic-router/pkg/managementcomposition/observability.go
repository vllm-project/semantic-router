package managementcomposition

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"errors"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/auditlog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managedruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementserver"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/subjectmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/usageledger"
)

const observabilityCursorKDFContext = "vllm-sr/management-observability-cursor/v1"

// observabilityComposition owns only read-side codecs and routes. PostgreSQL
// and authorization dependencies remain process-owned and borrowed.
type observabilityComposition struct {
	routes      *managementserver.ObservabilityRoutes
	logCursor   *usageledger.LogCursorCodec
	auditCursor *auditlog.CursorCodec
}

func composeObservability(
	dependencies managedruntime.ManagementDependencies,
	authorization managementauthorization.Runtime,
	namespaces managementserver.NamespaceResolver,
	sessions managementserver.SessionAuthenticator,
	authorizer managementserver.Authorizer,
	subjects managementserver.SubjectManagementService,
	apiKeys managementserver.APIKeyManagementService,
	now func() time.Time,
) (*observabilityComposition, error) {
	if dependencies.Database == nil || authorization.Loader == nil || namespaces == nil || sessions == nil ||
		authorizer == nil || subjects == nil || apiKeys == nil {
		return nil, errors.New("management observability dependencies are incomplete")
	}
	root := dependencies.Keyrings.ControlPlane.ManagementCursor.Symmetric()
	defer zeroSymmetricKeyring(&root)
	logKey, err := deriveObservabilityCursorKey(root, "request-log")
	if err != nil {
		return nil, fmt.Errorf("derive request-log cursor key: %w", err)
	}
	defer erase(logKey)
	auditKey, err := deriveObservabilityCursorKey(root, "audit-event")
	if err != nil {
		return nil, fmt.Errorf("derive audit cursor key: %w", err)
	}
	defer erase(auditKey)

	logCursor, err := usageledger.NewLogCursorCodec(logKey)
	if err != nil {
		return nil, err
	}
	auditCursor, err := auditlog.NewCursorCodec(auditKey)
	if err != nil {
		logCursor.Close()
		return nil, err
	}
	routes, err := managementserver.NewObservabilityRoutes(managementserver.ObservabilityRoutesOptions{
		Queries: usageledger.PostgresQueries{DB: dependencies.Database}, LogCursors: logCursor,
		Audit: auditlog.PostgresQueries{DB: dependencies.Database}, AuditCursors: auditCursor,
		Resources:     observabilityResources{subjects: subjects, apiKeys: apiKeys},
		Authorization: authorizer, Scopes: authorization, Namespaces: namespaces, Sessions: sessions, Now: now,
	})
	if err != nil {
		auditCursor.Close()
		logCursor.Close()
		return nil, err
	}
	return &observabilityComposition{routes: routes, logCursor: logCursor, auditCursor: auditCursor}, nil
}

type observabilityResources struct {
	subjects managementserver.SubjectManagementService
	apiKeys  managementserver.APIKeyManagementService
}

func (resources observabilityResources) GetUser(ctx context.Context, namespaceID, userID string) (subjectmanagement.User, error) {
	return resources.subjects.GetUser(ctx, namespaceID, userID)
}

func (resources observabilityResources) GetTeam(ctx context.Context, namespaceID, teamID string) (subjectmanagement.Team, error) {
	return resources.subjects.GetTeam(ctx, namespaceID, teamID)
}

func (resources observabilityResources) GetAPIKey(ctx context.Context, namespaceID, keyID string) (accesscontrol.APIKey, error) {
	return resources.apiKeys.Get(ctx, namespaceID, keyID)
}

func deriveObservabilityCursorKey(keyring securitykeyring.Symmetric, domain string) ([]byte, error) {
	if domain == "" || keyring.ActiveVersion == "" {
		return nil, errors.New("cursor key domain and active key version are required")
	}
	key, found := keyring.Keys[keyring.ActiveVersion]
	if !found || len(key) < sha256.Size {
		return nil, errors.New("active Management cursor key is unavailable")
	}
	mac := hmac.New(sha256.New, key)
	_, _ = mac.Write([]byte(observabilityCursorKDFContext))
	_, _ = mac.Write([]byte{0})
	_, _ = mac.Write([]byte(domain))
	_, _ = mac.Write([]byte{0})
	_, _ = mac.Write([]byte(keyring.ActiveVersion))
	return mac.Sum(nil), nil
}

func (composition *observabilityComposition) Close() error {
	if composition == nil {
		return nil
	}
	if composition.auditCursor != nil {
		composition.auditCursor.Close()
		composition.auditCursor = nil
	}
	if composition.logCursor != nil {
		composition.logCursor.Close()
		composition.logCursor = nil
	}
	composition.routes = nil
	return nil
}

func erase(value []byte) {
	for index := range value {
		value[index] = 0
	}
}

func zeroSymmetricKeyring(keyring *securitykeyring.Symmetric) {
	if keyring == nil {
		return
	}
	for _, key := range keyring.Keys {
		erase(key)
	}
	keyring.Keys = nil
	keyring.ActiveVersion = ""
}
