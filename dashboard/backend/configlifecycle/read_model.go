package configlifecycle

import (
	"context"
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

func (s *Service) currentConfigYAML() ([]byte, error) {
	if revisionYAML, found, err := s.activeRevisionYAML(context.Background()); err != nil {
		return nil, fmt.Errorf("failed to load active config revision: %w", err)
	} else if found {
		return revisionYAML, nil
	}

	data, err := os.ReadFile(s.ConfigPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read config: %w", err)
	}
	return data, nil
}

func (s *Service) loadConfigMap() (map[string]interface{}, error) {
	data, err := s.currentConfigYAML()
	if err != nil {
		return nil, err
	}

	result := make(map[string]interface{})
	if err := yaml.Unmarshal(data, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *Service) activeRevisionYAML(ctx context.Context) ([]byte, bool, error) {
	revision, err := s.latestActiveRevision(ctx)
	if err != nil {
		return nil, false, err
	}
	if revision == nil {
		return nil, false, nil
	}
	yamlData, err := revisionYAMLBytes(*revision)
	if err != nil {
		return nil, false, err
	}
	return yamlData, true, nil
}
