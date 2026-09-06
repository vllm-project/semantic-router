package evaluationplane

import (
	"fmt"
	"strings"
)

type serviceRuntimeSetup struct {
	codeRevision   string
	registrySource runtimeRegistrySource
	process        Process
}

func prepareServiceRuntime(
	options *Options,
	store *Store,
	constructor registryConstructor,
) (serviceRuntimeSetup, error) {
	codeRevision := strings.TrimSpace(options.CodeRevision)
	if !sourceRevisionPattern.MatchString(codeRevision) {
		return serviceRuntimeSetup{}, fmt.Errorf(
			"%w: evaluation source revision must be an immutable git commit or source-tree digest",
			ErrInvalid,
		)
	}
	if options.EnvoyAPIKeyEnv != "" && !secretEnvPattern.MatchString(options.EnvoyAPIKeyEnv) {
		return serviceRuntimeSetup{}, fmt.Errorf("evaluation Envoy credential reference must be an uppercase environment variable name")
	}
	routerAuthRequired, err := resolveRouterAuthentication(options.RouterAPIKeyEnv, options.CredentialProvider)
	if err != nil {
		return serviceRuntimeSetup{}, err
	}
	registrySource := newRuntimeRegistrySource(
		options, store.SuiteRoot(), routerAuthRequired, constructor,
	)
	_, err = registrySource.snapshot()
	if err != nil {
		return serviceRuntimeSetup{}, err
	}
	process, err := configureServiceProcess(options, store)
	if err != nil {
		return serviceRuntimeSetup{}, err
	}
	return serviceRuntimeSetup{
		codeRevision: codeRevision, registrySource: registrySource, process: process,
	}, nil
}
