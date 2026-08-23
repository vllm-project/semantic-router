package managementcomposition

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendegress"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managedruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth/issuerverifier"
)

type assertionVerifierComposition struct {
	verifier *issuerverifier.Verifier
	keys     *issuerverifier.RemoteKeySource
}

func composeAssertionVerifier(
	dependencies managedruntime.ManagementDependencies,
) (*assertionVerifierComposition, error) {
	repository, err := issuerverifier.NewPostgresRepository(dependencies.Database)
	if err != nil {
		return nil, fmt.Errorf("compose trusted-issuer repository: %w", err)
	}
	transport, err := backendegress.NewTransport(backendegress.TransportOptions{
		Guard: backendegress.Guard{Policy: dependencies.EgressPolicy},
	})
	if err != nil {
		return nil, fmt.Errorf("compose trusted-issuer egress transport: %w", err)
	}
	keys, err := issuerverifier.NewRemoteKeySource(issuerverifier.RemoteKeySourceOptions{
		Transport: transport,
	})
	if err != nil {
		transport.CloseIdleConnections()
		return nil, fmt.Errorf("compose trusted-issuer key source: %w", err)
	}
	verifier, err := issuerverifier.New(issuerverifier.Options{
		Repository: repository,
		Keys:       keys,
	})
	if err != nil {
		_ = keys.Close()
		return nil, fmt.Errorf("compose trusted-issuer assertion verifier: %w", err)
	}
	return &assertionVerifierComposition{verifier: verifier, keys: keys}, nil
}

func (composition *assertionVerifierComposition) Verifier() managementauth.SubjectAssertionVerifier {
	if composition == nil {
		return nil
	}
	return composition.verifier
}

func (composition *assertionVerifierComposition) Close() error {
	if composition == nil || composition.keys == nil {
		return nil
	}
	err := composition.keys.Close()
	composition.keys = nil
	composition.verifier = nil
	return err
}
