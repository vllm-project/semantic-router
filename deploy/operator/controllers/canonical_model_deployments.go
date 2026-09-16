package controllers

import (
	"bytes"
	"encoding/json"
	"fmt"

	"gopkg.in/yaml.v3"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// The CRD preserves router-owned fields; decode them against the canonical
// types so typos fail reconciliation instead of disappearing from the ConfigMap.
func decodeCanonicalModelObject[T any](raw *apiextensionsv1.JSON) (T, error) {
	var result T
	var object map[string]interface{}
	if err := json.Unmarshal(raw.Raw, &object); err != nil {
		return result, err
	}
	if object == nil {
		return result, fmt.Errorf("must be an object")
	}
	data, err := yaml.Marshal(object)
	if err != nil {
		return result, err
	}
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true)
	if err := decoder.Decode(&result); err != nil {
		return result, err
	}
	return result, nil
}

func applyOperatorModelDeployments(canonical *routerconfig.CanonicalConfig, spec vllmv1alpha1.ConfigSpec) error {
	if spec.ModelDeployments != nil {
		deployments, err := decodeCanonicalModelObject[map[string]routerconfig.ModelDeployment](spec.ModelDeployments)
		if err != nil {
			return fmt.Errorf("config.model_deployments: %w", err)
		}
		canonical.Global.ModelCatalog.Deployments = deployments
	}
	if spec.ModelAdmission != nil {
		admission, err := decodeCanonicalModelObject[map[string]routerconfig.AdmissionConfig](spec.ModelAdmission)
		if err != nil {
			return fmt.Errorf("config.model_admission: %w", err)
		}
		canonical.Global.ModelCatalog.Admission = admission
	}
	return nil
}
