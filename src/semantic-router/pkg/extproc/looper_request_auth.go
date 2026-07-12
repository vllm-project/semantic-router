/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package extproc

import (
	"fmt"
	"os"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
)

const looperSharedSecretEnvironmentVariable = "VLLM_SR_LOOPER_SHARED_SECRET" // #nosec G101 -- environment variable name, not a credential

// newLooperRequestAuthenticatorFromEnvironment uses a deployment-owned shared
// secret when configured so any router replica can authenticate Looper
// reentry. A missing variable keeps the single-process random-token default;
// a present but invalid variable is a startup error.
func newLooperRequestAuthenticatorFromEnvironment() (*looper.RequestAuthenticator, error) {
	sharedSecret, configured := os.LookupEnv(looperSharedSecretEnvironmentVariable)
	if !configured {
		return looper.NewRequestAuthenticator()
	}

	authenticator, err := looper.NewRequestAuthenticatorFromSharedSecret(sharedSecret)
	if err != nil {
		return nil, fmt.Errorf("invalid %s: %w", looperSharedSecretEnvironmentVariable, err)
	}
	return authenticator, nil
}
