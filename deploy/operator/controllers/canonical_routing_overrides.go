package controllers

import (
	"encoding/json"
	"fmt"

	"gopkg.in/yaml.v3"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type canonicalRoutingOverrideFields struct {
	modelBindings bool
	modelCards    bool
	signals       bool
	projections   bool
	decisions     bool
}

func canonicalRoutingFromKubernetesJSON(raw *apiextensionsv1.JSON) (routerconfig.CanonicalRouting, canonicalRoutingOverrideFields, error) {
	var routing routerconfig.CanonicalRouting
	var fields canonicalRoutingOverrideFields

	if raw == nil || len(raw.Raw) == 0 {
		return routing, fields, nil
	}

	var object map[string]interface{}
	if err := json.Unmarshal(raw.Raw, &object); err != nil {
		return routing, fields, err
	}
	if object == nil {
		return routing, fields, nil
	}

	for key := range object {
		switch key {
		case "model_bindings":
			fields.modelBindings = true
		case "modelCards":
			fields.modelCards = true
		case "signals":
			fields.signals = true
		case "projections":
			fields.projections = true
		case "decisions":
			fields.decisions = true
		}
	}

	data, err := yaml.Marshal(object)
	if err != nil {
		return routing, fields, err
	}
	if err := yaml.Unmarshal(data, &routing); err != nil {
		return routing, fields, err
	}
	if fields.modelBindings {
		bindingsJSON, err := json.Marshal(object["model_bindings"])
		if err != nil {
			return routing, fields, err
		}
		bindings, err := decodeCanonicalModelObject[map[string]routerconfig.ModelBinding](&apiextensionsv1.JSON{Raw: bindingsJSON})
		if err != nil {
			return routing, fields, fmt.Errorf("model_bindings: %w", err)
		}
		routing.ModelBindings = bindings
	}

	return routing, fields, nil
}

func applyCanonicalRoutingOverrides(
	canonical *routerconfig.CanonicalConfig,
	routing routerconfig.CanonicalRouting,
	fields canonicalRoutingOverrideFields,
) {
	if fields.modelBindings {
		canonical.Routing.ModelBindings = routing.ModelBindings
	}
	if fields.modelCards {
		canonical.Routing.ModelCards = routing.ModelCards
	}
	if fields.signals {
		canonical.Routing.Signals = routing.Signals
	}
	if fields.projections {
		canonical.Routing.Projections = routing.Projections
	}
	if fields.decisions {
		canonical.Routing.Decisions = routing.Decisions
	}
}

func formatCanonicalRoutingOverrideError(err error) error {
	if err == nil {
		return nil
	}
	return fmt.Errorf("config.routing: %w", err)
}
