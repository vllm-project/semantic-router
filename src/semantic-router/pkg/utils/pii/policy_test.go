package pii

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestIsPIIEnabled(t *testing.T) {
	tests := []struct {
		name         string
		decisionName string
		setupConfig  func() *config.RouterConfig
		expected     bool
	}{
		{
			name:         "Empty decision name",
			decisionName: "",
			setupConfig: func() *config.RouterConfig {
				return &config.RouterConfig{}
			},
			expected: false,
		},
		{
			name:         "Decision not found",
			decisionName: "nonexistent",
			setupConfig: func() *config.RouterConfig {
				return &config.RouterConfig{
					IntelligentRouting: config.IntelligentRouting{
						Decisions: []config.Decision{},
					},
				}
			},
			expected: false,
		},
		{
			name:         "PII enabled for decision",
			decisionName: "finance",
			setupConfig: func() *config.RouterConfig {
				return &config.RouterConfig{
					IntelligentRouting: config.IntelligentRouting{
						Decisions: []config.Decision{
							{
								Name: "finance",
								Plugins: []config.DecisionPlugin{
									{
										Type: "pii",
										Configuration: map[string]interface{}{
											"enabled": true,
										},
									},
								},
							},
						},
					},
				}
			},
			expected: true,
		},
		{
			name:         "PII disabled for decision",
			decisionName: "general",
			setupConfig: func() *config.RouterConfig {
				return &config.RouterConfig{
					IntelligentRouting: config.IntelligentRouting{
						Decisions: []config.Decision{
							{
								Name: "general",
								Plugins: []config.DecisionPlugin{
									{
										Type: "pii",
										Configuration: map[string]interface{}{
											"enabled": false,
										},
									},
								},
							},
						},
					},
				}
			},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := tt.setupConfig()
			checker := NewPolicyChecker(cfg)
			result := checker.IsPIIEnabled(tt.decisionName)
			if result != tt.expected {
				t.Errorf("IsPIIEnabled() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestCheckPolicy(t *testing.T) {
	tests := []struct {
		name          string
		decisionName  string
		detectedPII   []string
		setupConfig   func() *config.RouterConfig
		expectAllowed bool
		expectDenied  []string
	}{
		{
			name:         "PII disabled - allow all",
			decisionName: "general",
			detectedPII:  []string{"PERSON", "EMAIL_ADDRESS"},
			setupConfig: func() *config.RouterConfig {
				return &config.RouterConfig{
					IntelligentRouting: config.IntelligentRouting{
						Decisions: []config.Decision{
							{
								Name: "general",
								Plugins: []config.DecisionPlugin{
									{
										Type: "pii",
										Configuration: map[string]interface{}{
											"enabled": false,
										},
									},
								},
							},
						},
					},
				}
			},
			expectAllowed: true,
			expectDenied:  nil,
		},
		{
			name:         "PII enabled with allowed types",
			decisionName: "public",
			detectedPII:  []string{"PERSON", "ORGANIZATION"},
			setupConfig: func() *config.RouterConfig {
				return &config.RouterConfig{
					IntelligentRouting: config.IntelligentRouting{
						Decisions: []config.Decision{
							{
								Name: "public",
								Plugins: []config.DecisionPlugin{
									{
										Type: "pii",
										Configuration: map[string]interface{}{
											"enabled": true,
											"pii_types_allowed": []interface{}{
												"PERSON",
												"ORGANIZATION",
											},
										},
									},
								},
							},
						},
					},
				}
			},
			expectAllowed: true,
			expectDenied:  nil,
		},
		{
			name:         "Deny by default with allowed types",
			decisionName: "restricted",
			detectedPII:  []string{"PERSON", "EMAIL_ADDRESS", "CREDIT_CARD"},
			setupConfig: func() *config.RouterConfig {
				return &config.RouterConfig{
					IntelligentRouting: config.IntelligentRouting{
						Decisions: []config.Decision{
							{
								Name: "restricted",
								Plugins: []config.DecisionPlugin{
									{
										Type: "pii",
										Configuration: map[string]interface{}{
											"enabled":          true,
											"allow_by_default": false,
											"pii_types_allowed": []interface{}{
												"PERSON",
												"EMAIL_ADDRESS",
											},
										},
									},
								},
							},
						},
					},
				}
			},
			expectAllowed: false,
			expectDenied:  []string{"CREDIT_CARD"},
		},
		{
			name:         "NO_PII should be skipped",
			decisionName: "restricted",
			detectedPII:  []string{"NO_PII", "PERSON"},
			setupConfig: func() *config.RouterConfig {
				return &config.RouterConfig{
					IntelligentRouting: config.IntelligentRouting{
						Decisions: []config.Decision{
							{
								Name: "restricted",
								Plugins: []config.DecisionPlugin{
									{
										Type: "pii",
										Configuration: map[string]interface{}{
											"enabled":           true,
											"allow_by_default":  false,
											"pii_types_allowed": []interface{}{"PERSON"},
										},
									},
								},
							},
						},
					},
				}
			},
			expectAllowed: true,
			expectDenied:  nil,
		},
		{
			name:         "All PII types allowed",
			decisionName: "restricted",
			detectedPII:  []string{"PERSON", "EMAIL_ADDRESS"},
			setupConfig: func() *config.RouterConfig {
				return &config.RouterConfig{
					IntelligentRouting: config.IntelligentRouting{
						Decisions: []config.Decision{
							{
								Name: "restricted",
								Plugins: []config.DecisionPlugin{
									{
										Type: "pii",
										Configuration: map[string]interface{}{
											"enabled":          true,
											"allow_by_default": false,
											"pii_types_allowed": []interface{}{
												"PERSON",
												"EMAIL_ADDRESS",
											},
										},
									},
								},
							},
						},
					},
				}
			},
			expectAllowed: true,
			expectDenied:  nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := tt.setupConfig()
			checker := NewPolicyChecker(cfg)
			allowed, denied, err := checker.CheckPolicy(tt.decisionName, tt.detectedPII)
			if err != nil {
				t.Errorf("CheckPolicy() returned unexpected error: %v", err)
			}

			if allowed != tt.expectAllowed {
				t.Errorf("CheckPolicy() allowed = %v, want %v", allowed, tt.expectAllowed)
			}

			if len(denied) != len(tt.expectDenied) {
				t.Errorf("CheckPolicy() denied = %v, want %v", denied, tt.expectDenied)
			}

			// Check if denied types match
			if tt.expectDenied != nil {
				for _, expectedDenied := range tt.expectDenied {
					found := false
					for _, d := range denied {
						if d == expectedDenied {
							found = true
							break
						}
					}
					if !found {
						t.Errorf("Expected denied PII type %s not found in result", expectedDenied)
					}
				}
			}
		})
	}
}

func TestIsPIITypeAllowed(t *testing.T) {
	tests := []struct {
		name         string
		piiType      string
		allowedTypes []string
		expected     bool
	}{
		{
			name:         "Exact match",
			piiType:      "PERSON",
			allowedTypes: []string{"PERSON", "ORGANIZATION"},
			expected:     true,
		},
		{
			name:         "Not in allowed list",
			piiType:      "CREDIT_CARD",
			allowedTypes: []string{"PERSON", "ORGANIZATION"},
			expected:     false,
		},
		{
			name:         "BIO tag - B prefix",
			piiType:      "B-ORGANIZATION",
			allowedTypes: []string{"ORGANIZATION"},
			expected:     true,
		},
		{
			name:         "BIO tag - I prefix",
			piiType:      "I-PERSON",
			allowedTypes: []string{"PERSON"},
			expected:     true,
		},
		{
			name:         "BIO tag - O prefix",
			piiType:      "O-EMAIL_ADDRESS",
			allowedTypes: []string{"EMAIL_ADDRESS"},
			expected:     true,
		},
		{
			name:         "BIO tag - E prefix",
			piiType:      "E-PHONE_NUMBER",
			allowedTypes: []string{"PHONE_NUMBER"},
			expected:     true,
		},
		{
			name:         "BIO tag not allowed",
			piiType:      "B-CREDIT_CARD",
			allowedTypes: []string{"PERSON"},
			expected:     false,
		},
		{
			name:         "Invalid BIO format",
			piiType:      "X-PERSON",
			allowedTypes: []string{"PERSON"},
			expected:     false,
		},
		{
			name:         "Empty allowed list",
			piiType:      "PERSON",
			allowedTypes: []string{},
			expected:     false,
		},
		{
			name:         "BIO tag with exact match also in list",
			piiType:      "B-PERSON",
			allowedTypes: []string{"B-PERSON", "PERSON"},
			expected:     true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := isPIITypeAllowed(tt.piiType, tt.allowedTypes)
			if result != tt.expected {
				t.Errorf("isPIITypeAllowed(%s, %v) = %v, want %v",
					tt.piiType, tt.allowedTypes, result, tt.expected)
			}
		})
	}
}

func TestExtractAllContent(t *testing.T) {
	tests := []struct {
		name              string
		userContent       string
		nonUserMessages   []string
		expectedLength    int
		expectedFirstItem string
	}{
		{
			name:              "Both user and non-user content",
			userContent:       "What is the capital of France?",
			nonUserMessages:   []string{"Paris is the capital", "Population is 2.1M"},
			expectedLength:    3,
			expectedFirstItem: "What is the capital of France?",
		},
		{
			name:              "Only user content",
			userContent:       "Hello world",
			nonUserMessages:   []string{},
			expectedLength:    1,
			expectedFirstItem: "Hello world",
		},
		{
			name:              "Only non-user messages",
			userContent:       "",
			nonUserMessages:   []string{"System message 1", "System message 2"},
			expectedLength:    2,
			expectedFirstItem: "System message 1",
		},
		{
			name:              "Empty content",
			userContent:       "",
			nonUserMessages:   []string{},
			expectedLength:    0,
			expectedFirstItem: "",
		},
		{
			name:            "Multiple non-user messages",
			userContent:     "User query",
			nonUserMessages: []string{"Msg1", "Msg2", "Msg3", "Msg4"},
			expectedLength:  5,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ExtractAllContent(tt.userContent, tt.nonUserMessages)

			if len(result) != tt.expectedLength {
				t.Errorf("ExtractAllContent() length = %d, want %d", len(result), tt.expectedLength)
			}

			if tt.expectedLength > 0 && tt.expectedFirstItem != "" {
				if result[0] != tt.expectedFirstItem {
					t.Errorf("ExtractAllContent() first item = %s, want %s",
						result[0], tt.expectedFirstItem)
				}
			}
		})
	}
}

func TestNewPolicyChecker(t *testing.T) {
	cfg := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{
				{
					Name: "test-decision",
				},
			},
		},
	}

	checker := NewPolicyChecker(cfg)

	if checker == nil {
		t.Fatal("NewPolicyChecker() returned nil")
	}

	if checker.Config == nil {
		t.Error("PolicyChecker.Config is nil")
	}

	if len(checker.Config.Decisions) != 1 {
		t.Errorf("Expected 1 decision, got %d", len(checker.Config.Decisions))
	}
}

func TestCheckPolicy_NilDecision(t *testing.T) {
	cfg := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{},
		},
	}

	checker := NewPolicyChecker(cfg)
	allowed, denied, err := checker.CheckPolicy("nonexistent", []string{"PERSON"})
	if err != nil {
		t.Errorf("CheckPolicy() returned unexpected error: %v", err)
	}

	if !allowed {
		t.Error("CheckPolicy() should allow when decision not found")
	}

	if len(denied) > 0 {
		t.Errorf("CheckPolicy() should return nil denied list when decision not found, got %v", denied)
	}
}

func TestCheckPolicy_ComplexScenario(t *testing.T) {
	cfg := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{
				{
					Name: "banking",
					Plugins: []config.DecisionPlugin{
						{
							Type: "pii",
							Configuration: map[string]interface{}{
								"enabled":          true,
								"allow_by_default": false,
								"pii_types_allowed": []interface{}{
									"PERSON",
									"DATE_TIME",
									"ORGANIZATION",
								},
							},
						},
					},
				},
			},
		},
	}

	checker := NewPolicyChecker(cfg)

	testCases := []struct {
		detected      []string
		expectAllowed bool
		expectDenied  []string
	}{
		{
			detected:      []string{"PERSON", "DATE_TIME"},
			expectAllowed: true,
			expectDenied:  nil,
		},
		{
			detected:      []string{"PERSON", "CREDIT_CARD", "US_SSN"},
			expectAllowed: false,
			expectDenied:  []string{"CREDIT_CARD", "US_SSN"},
		},
		{
			detected:      []string{"NO_PII"},
			expectAllowed: true,
			expectDenied:  nil,
		},
		{
			detected:      []string{"B-PERSON", "I-ORGANIZATION"},
			expectAllowed: true,
			expectDenied:  nil,
		},
		{
			detected:      []string{"B-PERSON", "CREDIT_CARD", "I-ORGANIZATION"},
			expectAllowed: false,
			expectDenied:  []string{"CREDIT_CARD"},
		},
	}

	for i, tc := range testCases {
		allowed, denied, err := checker.CheckPolicy("banking", tc.detected)
		if err != nil {
			t.Errorf("Test case %d: unexpected error: %v", i, err)
		}
		if allowed != tc.expectAllowed {
			t.Errorf("Test case %d: allowed = %v, want %v", i, allowed, tc.expectAllowed)
		}
		if len(denied) != len(tc.expectDenied) {
			t.Errorf("Test case %d: denied length = %d, want %d", i, len(denied), len(tc.expectDenied))
		}
	}
}
