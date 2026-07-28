package controllers

import (
	"encoding/json"
	"fmt"
	"reflect"

	"gopkg.in/yaml.v3"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type canonicalRoutingOverrideFields struct {
	modelCards  bool
	signals     bool
	projections bool
	decisions   bool
}

func canonicalRoutingFromKubernetesJSON(raw *apiextensionsv1.JSON) (routerconfig.CanonicalRouting, canonicalRoutingOverrideFields, error) {
	var routing routerconfig.CanonicalRouting
	object, err := decodeCanonicalRoutingObject(raw)
	if err != nil || object == nil {
		return routing, canonicalRoutingOverrideFields{}, err
	}
	fields := canonicalRoutingFields(object)

	data, err := yaml.Marshal(object)
	if err != nil {
		return routing, fields, err
	}
	if err := yaml.Unmarshal(data, &routing); err != nil {
		return routing, fields, err
	}

	return routing, fields, nil
}

func decodeCanonicalRoutingObject(raw *apiextensionsv1.JSON) (map[string]interface{}, error) {
	if raw == nil || len(raw.Raw) == 0 {
		return nil, nil
	}

	var object map[string]interface{}
	if err := json.Unmarshal(raw.Raw, &object); err != nil {
		return nil, err
	}
	if object == nil {
		return nil, nil
	}
	if err := routerconfig.RejectUnknownConfigValue(
		object,
		reflect.TypeOf(routerconfig.CanonicalRouting{}),
		"config.routing",
	); err != nil {
		return nil, err
	}
	return object, nil
}

func canonicalRoutingFields(object map[string]interface{}) canonicalRoutingOverrideFields {
	return canonicalRoutingOverrideFields{
		modelCards:  hasCanonicalRoutingField(object, "modelCards"),
		signals:     hasCanonicalRoutingField(object, "signals"),
		projections: hasCanonicalRoutingField(object, "projections"),
		decisions:   hasCanonicalRoutingField(object, "decisions"),
	}
}

func hasCanonicalRoutingField(object map[string]interface{}, field string) bool {
	_, ok := object[field]
	return ok
}

func applyCanonicalRoutingOverrides(
	canonical *routerconfig.CanonicalConfig,
	routing routerconfig.CanonicalRouting,
	fields canonicalRoutingOverrideFields,
) {
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
