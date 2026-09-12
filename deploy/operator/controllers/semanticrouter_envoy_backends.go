/*
Copyright 2026 vLLM Semantic Router Contributors.

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

package controllers

import (
	"bytes"
	"fmt"
	"net"
	"strconv"
	"strings"
	"text/template"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type operatorEnvoyBackend struct {
	Name      string
	Host      string
	Port      int
	Authority string
	Weight    int
	TLS       bool
}

type operatorEnvoyModel struct {
	Name     string
	Backends []operatorEnvoyBackend
}

// The Operator's supported discoveries produce host:port HTTP(S) backends.
// Resolve them into model routes, rather than forwarding to the client's Host:
// ExtProc communicates the selected model but does not rewrite :authority.
func generateStandaloneEnvoyConfig(canonical *routerconfig.CanonicalConfig) (string, error) {
	data := struct {
		Models          []operatorEnvoyModel
		DefaultBackends []operatorEnvoyBackend
	}{}
	for _, model := range canonical.Providers.Models {
		projected := operatorEnvoyModel{Name: model.Name}
		for index, backend := range model.BackendRefs {
			endpoint, err := operatorEnvoyBackendFromRef(model.Name, index, backend)
			if err != nil {
				return "", err
			}
			projected.Backends = append(projected.Backends, endpoint)
		}
		if len(projected.Backends) == 0 {
			continue
		}
		data.Models = append(data.Models, projected)
		if model.Name == canonical.Providers.Defaults.DefaultModel {
			data.DefaultBackends = projected.Backends
		}
	}
	tmpl, err := template.New("operator-envoy").Funcs(template.FuncMap{"quote": strconv.Quote}).Parse(standaloneEnvoyConfigYAML)
	if err != nil {
		return "", fmt.Errorf("parse standalone Envoy template: %w", err)
	}
	var output bytes.Buffer
	if err := tmpl.Execute(&output, data); err != nil {
		return "", fmt.Errorf("render standalone Envoy config: %w", err)
	}
	return output.String(), nil
}

func operatorEnvoyBackendFromRef(model string, index int, ref routerconfig.CanonicalBackendRef) (operatorEnvoyBackend, error) {
	host, portText, err := net.SplitHostPort(ref.Endpoint)
	if err != nil || host == "" {
		return operatorEnvoyBackend{}, fmt.Errorf("model %q backend %d requires a discovered host:port endpoint", model, index)
	}
	port, err := strconv.Atoi(portText)
	if err != nil || port < 1 || port > 65535 {
		return operatorEnvoyBackend{}, fmt.Errorf("model %q backend %d has an invalid port", model, index)
	}
	protocol := strings.ToLower(ref.Protocol)
	if protocol != "http" && protocol != "https" {
		return operatorEnvoyBackend{}, fmt.Errorf("model %q backend %d requires HTTP or HTTPS", model, index)
	}
	weight := ref.Weight
	if weight == 0 {
		weight = 1
	}
	if weight < 0 {
		return operatorEnvoyBackend{}, fmt.Errorf("model %q backend %d has a negative weight", model, index)
	}
	return operatorEnvoyBackend{
		Name:      fmt.Sprintf("model_%x_backend_%d", []byte(model), index),
		Host:      host,
		Port:      port,
		Authority: ref.Endpoint,
		Weight:    weight,
		TLS:       protocol == "https",
	}, nil
}
