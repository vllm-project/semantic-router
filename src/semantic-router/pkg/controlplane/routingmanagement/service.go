package routingmanagement

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providerdiscovery"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

type ServiceOptions struct {
	Store              Store
	ModelCompiler      providercatalog.ModelCompiler
	DiscoveryClaims    providerdiscovery.ClaimCodec
	CredentialVersions CredentialVersionReader
	Prober             Prober
	CursorKeyring      securitykeyring.Symmetric
	Now                func() time.Time
	ManifestCodec      ManifestCodec
}

type Service struct {
	store              Store
	modelCompiler      providercatalog.ModelCompiler
	discoveryClaims    providerdiscovery.ClaimCodec
	credentialVersions CredentialVersionReader
	prober             Prober
	cursors            routingCursorCodec
	now                func() time.Time
	manifests          ManifestCodec
}

func NewService(options ServiceOptions) (*Service, error) {
	if options.Store == nil || options.ModelCompiler.Catalog == nil || options.ModelCompiler.Registry == nil {
		return nil, fmt.Errorf("%w: routing store and Model compiler are required", ErrInvalid)
	}
	cursors, err := newRoutingCursorCodec(options.CursorKeyring)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrInvalid, err)
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	return &Service{
		store: options.Store, modelCompiler: options.ModelCompiler,
		discoveryClaims: options.DiscoveryClaims, credentialVersions: options.CredentialVersions,
		prober: options.Prober, cursors: cursors, now: now,
		manifests: options.ManifestCodec,
	}, nil
}

func (service *Service) ExportCurrentManifest(ctx context.Context, namespaceID string) ([]byte, int64, error) {
	return service.exportCurrentManifest(ctx, namespaceID)
}

func (service *Service) Close() {
	if service != nil {
		service.cursors.close()
	}
}

func (service *Service) GetModel(ctx context.Context, namespaceID, id string) (Model, error) {
	return service.store.GetModel(ctx, namespaceID, id)
}

func (service *Service) CreateModel(
	ctx context.Context, namespaceID string, input ModelInput, mutation MutationContext,
) (Model, RevisionReceipt, error) {
	model, err := service.compileModel(ctx, namespaceID, input, 1)
	if err != nil {
		return Model{}, RevisionReceipt{}, err
	}
	models, receipt, err := service.store.CreateModels(ctx, namespaceID, []routingsnapshot.Model{model}, mutation)
	if err != nil {
		return Model{}, RevisionReceipt{}, err
	}
	return models[0], receipt, nil
}

func (service *Service) PatchModel(
	ctx context.Context, namespaceID, id string, expected int64, patch ModelPatch, mutation MutationContext,
) (Model, RevisionReceipt, error) {
	current, err := service.store.GetModel(ctx, namespaceID, id)
	if err != nil {
		return Model{}, RevisionReceipt{}, err
	}
	if current.Revision != expected {
		return Model{}, RevisionReceipt{}, ErrConflict
	}
	if patch.Empty() {
		return Model{}, RevisionReceipt{}, fmt.Errorf("%w: Model patch is empty", ErrInvalid)
	}

	candidate := current.Current
	candidate.Revision++
	if patch.Name != nil {
		candidate.Name = *patch.Name
	}
	if patch.Aliases != nil {
		candidate.Aliases = append([]string(nil), (*patch.Aliases)...)
	}
	if patch.ParamSize != nil {
		candidate.ParamSize = *patch.ParamSize
	}
	if patch.ContextWindowSize != nil {
		candidate.ContextWindowSize = *patch.ContextWindowSize
	}
	if patch.Description != nil {
		candidate.Description = *patch.Description
	}
	if patch.Capabilities != nil {
		candidate.Capabilities = append([]string(nil), (*patch.Capabilities)...)
	}
	if patch.Reasoning != nil {
		candidate.Reasoning = *patch.Reasoning
		candidate.Reasoning.Efforts = append([]string(nil), patch.Reasoning.Efforts...)
	}
	if patch.LoRAs != nil {
		candidate.LoRAs = append([]string(nil), (*patch.LoRAs)...)
	}
	if patch.QualityScore != nil {
		candidate.QualityScore = *patch.QualityScore
	}
	if patch.Modality != nil {
		candidate.Modality = *patch.Modality
	}
	if patch.Tags != nil {
		candidate.Tags = append([]string(nil), (*patch.Tags)...)
	}
	if patch.Execution != nil {
		candidate.Execution = *patch.Execution
	}
	if patch.Pricing != nil {
		candidate.Pricing = cloneModelPricing(*patch.Pricing)
	}

	var model routingsnapshot.Model
	if patch.Backends != nil {
		model, err = service.compileModel(ctx, namespaceID, ModelInput{
			ID: candidate.ID, Name: candidate.Name, Aliases: candidate.Aliases,
			ParamSize: candidate.ParamSize, ContextWindowSize: candidate.ContextWindowSize,
			Description:  candidate.Description,
			Capabilities: candidate.Capabilities, Reasoning: candidate.Reasoning, LoRAs: candidate.LoRAs,
			QualityScore: candidate.QualityScore, Modality: candidate.Modality, Tags: candidate.Tags,
			Execution: candidate.Execution, Pricing: candidate.Pricing, Backends: *patch.Backends,
		}, candidate.Revision)
	} else {
		model, err = service.normalizeCompiledModel(ctx, namespaceID, candidate)
	}
	if err != nil {
		return Model{}, RevisionReceipt{}, err
	}
	return service.store.UpdateModel(ctx, namespaceID, id, expected, model, mutation)
}

func cloneModelPricing(value routingsnapshot.ModelPricing) routingsnapshot.ModelPricing {
	clone := func(input *string) *string {
		if input == nil {
			return nil
		}
		value := *input
		return &value
	}
	return routingsnapshot.ModelPricing{
		InputCostPerMillionTokens:      clone(value.InputCostPerMillionTokens),
		OutputCostPerMillionTokens:     clone(value.OutputCostPerMillionTokens),
		CacheReadCostPerMillionTokens:  clone(value.CacheReadCostPerMillionTokens),
		CacheWriteCostPerMillionTokens: clone(value.CacheWriteCostPerMillionTokens),
	}
}

func (service *Service) DeleteModel(
	ctx context.Context, namespaceID, id string, expected int64, mutation MutationContext,
) (RevisionReceipt, error) {
	return service.store.DeleteModel(ctx, namespaceID, id, expected, mutation)
}

func (service *Service) BulkImport(
	ctx context.Context, request BulkImportRequest, mutation MutationContext,
) ([]Model, RevisionReceipt, error) {
	if len(request.Selections) == 0 || len(request.Selections) > 200 ||
		validateCatalogRevision(request.CatalogRevision) != nil {
		return nil, RevisionReceipt{}, fmt.Errorf("%w: bulk selection is invalid", ErrInvalid)
	}
	itemIDs := make([]string, len(request.Selections))
	for index := range request.Selections {
		itemIDs[index] = request.Selections[index].CatalogItemID
	}
	verified, err := service.discoveryClaims.VerifySelection(request.DiscoveryClaim, providerdiscovery.ClaimExpectation{
		NamespaceID: request.NamespaceID, AuthorityDigest: request.AuthorityDigest,
		CatalogRevision: request.CatalogRevision, ProviderID: request.ProviderID,
	}, itemIDs, service.now().UTC())
	if err != nil {
		return nil, RevisionReceipt{}, fmt.Errorf("%w: %w", ErrClaim, err)
	}
	if verified.Binding.Origin == "" || verified.Binding.CredentialID != request.CredentialID {
		return nil, RevisionReceipt{}, fmt.Errorf("%w: discovery backend binding changed", ErrClaim)
	}
	if request.CredentialID != "" {
		if service.credentialVersions == nil {
			return nil, RevisionReceipt{}, fmt.Errorf("%w: credential version service is unavailable", ErrClaim)
		}
		version, pinErr := service.credentialVersions.Pin(
			ctx, request.CredentialID, request.ProviderID, verified.Binding.Origin,
		)
		if pinErr != nil || version != verified.Binding.CredentialVersion {
			return nil, RevisionReceipt{}, fmt.Errorf("%w: discovered credential version is stale", ErrClaim)
		}
	} else if verified.Binding.CredentialVersion != "" {
		return nil, RevisionReceipt{}, fmt.Errorf("%w: discovery credential binding is inconsistent", ErrClaim)
	}
	models := make([]routingsnapshot.Model, len(request.Selections))
	for index, selection := range request.Selections {
		if verified.Models[index].CatalogItemID != selection.CatalogItemID {
			return nil, RevisionReceipt{}, fmt.Errorf("%w: discovery selection order changed", ErrClaim)
		}
		input := ModelInput{
			ID: selection.ID, Name: selection.Name, Aliases: selection.Aliases,
			ParamSize: selection.ParamSize, ContextWindowSize: selection.ContextWindowSize,
			Description:  selection.Description,
			Capabilities: selection.Capabilities, Reasoning: selection.Reasoning, LoRAs: selection.LoRAs,
			QualityScore: selection.QualityScore, Modality: selection.Modality, Tags: selection.Tags,
			Execution: selection.Execution, Pricing: selection.Pricing,
			Backends: []ModelBackendInput{{
				ProviderID: request.ProviderID, InterfaceID: request.InterfaceID,
				ProviderModelID: verified.Models[index].ProviderModelID,
				CredentialID:    request.CredentialID, Origin: request.Origin,
				ConnectionFields: request.ConnectionFields, Weight: request.Weight,
			}},
		}
		var bindings []providercatalog.CompiledModelBackend
		models[index], bindings, err = service.compileModelWithBindings(ctx, request.NamespaceID, input, 1)
		if err != nil {
			return nil, RevisionReceipt{}, fmt.Errorf("selection %d: %w", index, err)
		}
		if len(bindings) != 1 || bindings[0].ConnectionDigest != verified.Binding.ConnectionDigest ||
			models[index].CatalogRevision != verified.Binding.CatalogRevision ||
			models[index].Backends[0].Origin != verified.Binding.Origin ||
			models[index].Backends[0].ProviderCredentialID != verified.Binding.CredentialID {
			return nil, RevisionReceipt{}, fmt.Errorf("%w: compiled backend differs from discovery binding", ErrClaim)
		}
	}
	return service.store.CreateModels(ctx, request.NamespaceID, models, mutation)
}

func (service *Service) ProbeModel(
	ctx context.Context, namespaceID, id string, timeout time.Duration,
) (ProbeResult, error) {
	if service.prober == nil {
		return ProbeResult{}, ErrProbeUnavailable
	}
	model, err := service.store.GetModel(ctx, namespaceID, id)
	if err != nil {
		return ProbeResult{}, err
	}
	timeout, err = resolveModelProbeTimeout(model.Current.Execution.RequestTimeout, timeout)
	if err != nil {
		return ProbeResult{}, err
	}
	probeCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	return service.prober.Probe(probeCtx, ProbeRequest{NamespaceID: namespaceID, Model: model.Current, Timeout: timeout})
}

func (service *Service) compileModel(
	ctx context.Context, namespaceID string, input ModelInput, revision int64,
) (routingsnapshot.Model, error) {
	model, _, err := service.compileModelWithBindings(ctx, namespaceID, input, revision)
	return model, err
}

func (service *Service) compileModelWithBindings(
	ctx context.Context, namespaceID string, input ModelInput, revision int64,
) (routingsnapshot.Model, []providercatalog.CompiledModelBackend, error) {
	if input.ID == "" {
		input.ID = generatedID("mdl")
	}
	if err := validateIdentity(input.ID, input.Name); err != nil || revision <= 0 || len(input.Backends) == 0 || len(input.Backends) > 32 {
		return routingsnapshot.Model{}, nil, fmt.Errorf("%w: Model identity, revision, or backends are invalid", ErrInvalid)
	}
	aliases, err := uniqueCanonical(input.Aliases, 64)
	if err != nil {
		return routingsnapshot.Model{}, nil, err
	}
	capabilities, err := uniqueCanonical(input.Capabilities, 64)
	if err != nil {
		return routingsnapshot.Model{}, nil, err
	}
	loras, err := uniqueCanonical(input.LoRAs, 64)
	if err != nil {
		return routingsnapshot.Model{}, nil, err
	}
	tags, err := uniqueCanonical(input.Tags, 64)
	if err != nil {
		return routingsnapshot.Model{}, nil, err
	}
	model := routingsnapshot.Model{
		ID: input.ID, Revision: revision, Name: input.Name, Aliases: aliases,
		ParamSize: input.ParamSize, ContextWindowSize: input.ContextWindowSize,
		Description:  input.Description,
		Capabilities: capabilities, Reasoning: input.Reasoning, LoRAs: loras,
		QualityScore: input.QualityScore, Modality: input.Modality, Tags: tags,
		Execution: input.Execution, Pricing: input.Pricing,
	}
	bindings := make([]providercatalog.CompiledModelBackend, 0, len(input.Backends))
	for index, backendInput := range input.Backends {
		if backendInput.ID == "" {
			backendInput.ID = uuid.NewString()
		}
		compiled, compileErr := service.modelCompiler.CompileBackend(ctx, providercatalog.ModelBackendRequest{
			NamespaceID: namespaceID, BackendID: backendInput.ID,
			ProviderID: backendInput.ProviderID, InterfaceID: backendInput.InterfaceID,
			ProviderModelID: backendInput.ProviderModelID,
			CredentialID:    backendInput.CredentialID, Origin: backendInput.Origin,
			ConnectionFields: backendInput.ConnectionFields, Weight: backendInput.Weight,
		})
		if compileErr != nil {
			return routingsnapshot.Model{}, nil, fmt.Errorf("%w: backend %d: %w", ErrInvalid, index, compileErr)
		}
		if model.CatalogRevision != "" && model.CatalogRevision != compiled.CatalogRevision {
			return routingsnapshot.Model{}, nil, fmt.Errorf("%w: one Model revision cannot mix Provider catalog revisions", ErrInvalid)
		}
		model.CatalogRevision = compiled.CatalogRevision
		model.Backends = append(model.Backends, compiled.Backend)
		bindings = append(bindings, compiled)
	}
	normalized, err := service.normalizeCompiledModel(ctx, namespaceID, model)
	return normalized, bindings, err
}

func (service *Service) normalizeCompiledModel(
	ctx context.Context, namespaceID string, model routingsnapshot.Model,
) (routingsnapshot.Model, error) {
	if err := validateIdentity(model.ID, model.Name); err != nil {
		return routingsnapshot.Model{}, err
	}
	var err error
	if model.Aliases, err = uniqueCanonical(model.Aliases, 64); err != nil {
		return routingsnapshot.Model{}, err
	}
	if model.Capabilities, err = uniqueCanonical(model.Capabilities, 64); err != nil {
		return routingsnapshot.Model{}, err
	}
	if model.LoRAs, err = uniqueCanonical(model.LoRAs, 64); err != nil {
		return routingsnapshot.Model{}, err
	}
	if model.Tags, err = uniqueCanonical(model.Tags, 64); err != nil {
		return routingsnapshot.Model{}, err
	}
	currency, err := service.store.NamespaceCurrency(ctx, namespaceID)
	if err != nil {
		return routingsnapshot.Model{}, err
	}
	validated, err := routingsnapshot.Compile(routingsnapshot.Bundle{
		NamespaceID: namespaceID, Revision: 1, Currency: currency, Models: []routingsnapshot.Model{model},
	})
	if err != nil {
		return routingsnapshot.Model{}, fmt.Errorf("%w: %w", ErrInvalid, err)
	}
	// Persist the normalized value returned by the single runtime compiler so
	// configured defaults and canonical decimal/duration forms cannot drift
	// between Management reads and the published data-plane snapshot.
	return validated.Models[0], nil
}

func generatedID(prefix string) string {
	return prefix + "_" + strings.ReplaceAll(uuid.NewString(), "-", "")[:20]
}
