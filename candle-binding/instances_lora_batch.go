package candle_binding

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// LoRABatchInstanceOptions names the three existing merged classifier artifacts.
// Each task retains its own tokenizer, labels, budget, device and precision.
// An omitted ModelType reads the artifact architecture; BERT uses the existing
// merged LoRA loader, while ModernBERT/mmBERT use their maintained loaders.
type LoRABatchInstanceOptions struct {
	Intent   InstanceOptions
	PII      InstanceOptions
	Security InstanceOptions
}

type LoRABatchInstanceInfo struct{ Intent, PII, Security InstanceInfo }

// LoRABatchOutput contains one independent result per input for every task.
// Scores are genuine softmax distributions; token offsets are UTF-8 bytes.
// This retains the existing parallel engine's separate task forwards. It does
// not claim a joint forward or copy an aggregate prediction across the batch.
type LoRABatchOutput struct {
	Intent   []DistributionOutput
	PII      []TokenOutput
	Security []DistributionOutput
}

type LoRABatchClassifier struct {
	mu       sync.RWMutex
	intent   *SequenceClassifier
	pii      *TokenClassifier
	security *SequenceClassifier
	closed   bool
}

func mergedLoRAOptions(options InstanceOptions) (InstanceOptions, error) {
	if options.ModelType != "" {
		return options, nil
	}
	data, err := os.ReadFile(filepath.Join(options.ModelPath, "config.json"))
	if err != nil {
		return options, fmt.Errorf("read merged classifier architecture: %w", err)
	}
	var config struct {
		ModelType string `json:"model_type"`
	}
	if err = json.Unmarshal(data, &config); err != nil {
		return options, err
	}
	kind := strings.ToLower(config.ModelType)
	switch kind {
	case "modernbert", "mmbert", "mmbert32k":
		options.ModelType = kind
	case "bert":
		options.ModelType = "bert_lora"
	default:
		return options, &InstanceError{Code: "capability", Message: fmt.Sprintf("unsupported merged classifier architecture %q", kind)}
	}
	return options, nil
}

func LoadLoRABatchClassifier(options LoRABatchInstanceOptions) (*LoRABatchClassifier, error) {
	var err error
	options.Intent, err = mergedLoRAOptions(options.Intent)
	if err != nil {
		return nil, err
	}
	options.PII, err = mergedLoRAOptions(options.PII)
	if err != nil {
		return nil, err
	}
	options.Security, err = mergedLoRAOptions(options.Security)
	if err != nil {
		return nil, err
	}
	m := &LoRABatchClassifier{}
	m.intent, err = LoadSequenceClassifier(options.Intent)
	if err != nil {
		return nil, err
	}
	m.pii, err = LoadTokenClassifier(options.PII)
	if err != nil {
		_ = m.Close()
		return nil, err
	}
	m.security, err = LoadSequenceClassifier(options.Security)
	if err != nil {
		_ = m.Close()
		return nil, err
	}
	return m, nil
}

// ComposeLoRABatchClassifier clones independent references to prepared heads.
// Callers may first BindSequenceHead/BindTokenHead to explicitly share a real
// backbone. The supplied handles remain owned by the caller.
func ComposeLoRABatchClassifier(intent *SequenceClassifier, pii *TokenClassifier, security *SequenceClassifier) (*LoRABatchClassifier, error) {
	if intent == nil || pii == nil || security == nil {
		return nil, &InstanceError{Code: "configuration", Message: "all three merged classifier heads are required"}
	}
	m := &LoRABatchClassifier{}
	var err error
	m.intent, err = intent.Clone()
	if err != nil {
		return nil, err
	}
	m.pii, err = pii.Clone()
	if err != nil {
		_ = m.Close()
		return nil, err
	}
	m.security, err = security.Clone()
	if err != nil {
		_ = m.Close()
		return nil, err
	}
	return m, nil
}

func (m *LoRABatchClassifier) ClassifyBatch(texts []string) (LoRABatchOutput, error) {
	if m == nil {
		return LoRABatchOutput{}, ErrInstanceClosed
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.closed {
		return LoRABatchOutput{}, ErrInstanceClosed
	}
	if len(texts) == 0 {
		return LoRABatchOutput{}, &InstanceError{Code: "configuration", Message: "batch must contain at least one input"}
	}
	output := LoRABatchOutput{Intent: make([]DistributionOutput, len(texts)), PII: make([]TokenOutput, len(texts)), Security: make([]DistributionOutput, len(texts))}
	var failures [3]error
	var calls sync.WaitGroup
	calls.Add(3)
	go func() {
		defer calls.Done()
		for index, text := range texts {
			output.Intent[index], failures[0] = m.intent.Classify(text)
			if failures[0] != nil {
				return
			}
		}
	}()
	go func() {
		defer calls.Done()
		for index, text := range texts {
			output.PII[index], failures[1] = m.pii.ClassifyTokens(text)
			if failures[1] != nil {
				return
			}
		}
	}()
	go func() {
		defer calls.Done()
		for index, text := range texts {
			output.Security[index], failures[2] = m.security.Classify(text)
			if failures[2] != nil {
				return
			}
		}
	}()
	calls.Wait()
	if err := errors.Join(failures[:]...); err != nil {
		return LoRABatchOutput{}, err
	}
	return output, nil
}

func (m *LoRABatchClassifier) Info() (LoRABatchInstanceInfo, error) {
	if m == nil {
		return LoRABatchInstanceInfo{}, ErrInstanceClosed
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.closed {
		return LoRABatchInstanceInfo{}, ErrInstanceClosed
	}
	intent, e1 := m.intent.Info()
	pii, e2 := m.pii.Info()
	security, e3 := m.security.Info()
	return LoRABatchInstanceInfo{Intent: intent, PII: pii, Security: security}, errors.Join(e1, e2, e3)
}

func (m *LoRABatchClassifier) Clone() (*LoRABatchClassifier, error) {
	if m == nil {
		return nil, ErrInstanceClosed
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.closed {
		return nil, ErrInstanceClosed
	}
	return ComposeLoRABatchClassifier(m.intent, m.pii, m.security)
}

func (m *LoRABatchClassifier) Close() error {
	if m == nil {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil
	}
	m.closed = true
	var err error
	if m.intent != nil {
		err = errors.Join(err, m.intent.Close())
	}
	if m.pii != nil {
		err = errors.Join(err, m.pii.Close())
	}
	if m.security != nil {
		err = errors.Join(err, m.security.Close())
	}
	return err
}
