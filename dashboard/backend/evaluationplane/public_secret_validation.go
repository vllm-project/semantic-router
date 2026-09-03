package evaluationplane

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"
)

func (s *Service) rejectConfiguredSecretBytes(data []byte) error {
	for _, pattern := range s.configuredPublicSecretPatterns() {
		if bytes.Contains(data, pattern) {
			return fmt.Errorf("%w: public evaluation evidence contains a configured credential", ErrInvalid)
		}
	}
	return s.rejectConfiguredSecretJSONReader(bytes.NewReader(data))
}

func (s *Service) rejectConfiguredSecretReader(reader io.Reader) error {
	patterns := s.configuredPublicSecretPatterns()
	if len(patterns) == 0 {
		return nil
	}
	maxPattern := 0
	for _, pattern := range patterns {
		if len(pattern) > maxPattern {
			maxPattern = len(pattern)
		}
	}
	buffer := make([]byte, 32*1024)
	tail := make([]byte, 0, maxPattern-1)
	for {
		read, readErr := reader.Read(buffer)
		if read > 0 {
			window := make([]byte, 0, len(tail)+read)
			window = append(window, tail...)
			window = append(window, buffer[:read]...)
			for _, pattern := range patterns {
				if bytes.Contains(window, pattern) {
					return fmt.Errorf("%w: public evaluation evidence contains a configured credential", ErrInvalid)
				}
			}
			keep := maxPattern - 1
			if keep > len(window) {
				keep = len(window)
			}
			tail = append(tail[:0], window[len(window)-keep:]...)
		}
		if readErr != nil {
			if readErr == io.EOF {
				return nil
			}
			return fmt.Errorf("scan public evaluation evidence: %w", readErr)
		}
	}
}

func (s *Service) rejectConfiguredSecretArtifact(reader io.ReadSeeker, mediaType string) error {
	if err := s.rejectConfiguredSecretReader(reader); err != nil {
		return err
	}
	if mediaType != "application/json" && mediaType != "application/x-ndjson" {
		return nil
	}
	if _, err := reader.Seek(0, io.SeekStart); err != nil {
		return fmt.Errorf("rewind public evaluation evidence for credential scan: %w", err)
	}
	return s.rejectConfiguredSecretJSONReader(reader)
}

// JSON permits many byte encodings for the same logical string (for example,
// `secret` and `secr\u0065t`). Raw-byte scanning alone is therefore not a
// confidentiality boundary. Token scanning compares decoded string values and
// also covers newline-delimited streams without materializing the artifact.
func (s *Service) rejectConfiguredSecretJSONReader(reader io.Reader) error {
	secrets := s.configuredPublicSecrets()
	if len(secrets) == 0 {
		return nil
	}
	decoder := json.NewDecoder(reader)
	for {
		token, err := decoder.Token()
		if err != nil {
			if err == io.EOF {
				return nil
			}
			return fmt.Errorf("%w: public evaluation JSON is invalid during credential scan", ErrInvalid)
		}
		value, ok := token.(string)
		if ok {
			for _, secret := range secrets {
				if strings.Contains(value, secret) {
					return fmt.Errorf("%w: public evaluation evidence contains a configured credential", ErrInvalid)
				}
			}
		}
	}
}

func (s *Service) configuredPublicSecretPatterns() [][]byte {
	secrets := s.configuredPublicSecrets()
	patterns := make([][]byte, 0, len(secrets)*2)
	for _, secret := range secrets {
		raw := []byte(secret)
		patterns = append(patterns, raw)
		encoded, err := json.Marshal(secret)
		if err == nil && len(encoded) >= 2 {
			escaped := encoded[1 : len(encoded)-1]
			if len(escaped) > 0 && !bytes.Equal(escaped, raw) {
				patterns = append(patterns, escaped)
			}
		}
	}
	return patterns
}

func (s *Service) configuredPublicSecrets() []string {
	source := s.registrySource
	envNames := []string{source.routerAPIKeyEnv, source.envoyAPIKeyEnv}
	for _, endpoint := range []*ServiceEndpoint{
		source.agentTaskLedger,
		source.faultRecoveryLedger,
		source.hardPolicyLedger,
		source.productionExperimentLedger,
	} {
		if endpoint != nil && endpoint.APIKey != nil {
			envNames = append(envNames, endpoint.APIKey.Env)
		}
	}
	seen := make(map[string]struct{}, len(envNames))
	secrets := make([]string, 0, len(envNames))
	for _, envName := range envNames {
		envName = strings.TrimSpace(envName)
		if envName == "" {
			continue
		}
		if _, duplicate := seen[envName]; duplicate {
			continue
		}
		seen[envName] = struct{}{}
		if secret, present := os.LookupEnv(envName); present && secret != "" {
			secrets = append(secrets, secret)
		}
	}
	return secrets
}
