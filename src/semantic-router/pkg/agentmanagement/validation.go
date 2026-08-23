package agentmanagement

import (
	"fmt"
	"net/url"
	"regexp"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/google/uuid"
)

var transcriptCredentialPattern = regexp.MustCompile(`(?i)(bearer|basic)[ \t]+[^ \t\r\n]+|\b(?:sk-|vsr_|vsd_)[A-Za-z0-9._-]{8,}`)

const (
	defaultMaximumTurnSeconds = int64(1800)
	defaultMaximumToolSteps   = 48
	defaultContextTokenBudget = int64(65536)
)

func NormalizeProfileInput(input ProfileInput) (ProfileInput, error) {
	input.Name = strings.TrimSpace(input.Name)
	input.Description = strings.TrimSpace(input.Description)
	if input.ApprovalPolicy == "" {
		input.ApprovalPolicy = "required"
	}
	if input.MaximumTurnSeconds == 0 {
		input.MaximumTurnSeconds = defaultMaximumTurnSeconds
	}
	if input.MaximumToolSteps == 0 {
		input.MaximumToolSteps = defaultMaximumToolSteps
	}
	if input.ContextTokenBudget == 0 {
		input.ContextTokenBudget = defaultContextTokenBudget
	}
	if len(input.SupportedModes) == 0 {
		input.SupportedModes = []SessionMode{SessionChat, SessionBuilder}
	}
	input.ToolPolicy.Allow = canonicalStrings(input.ToolPolicy.Allow)
	input.ToolPolicy.Deny = canonicalStrings(input.ToolPolicy.Deny)
	if duplicateSkillReference(input.Skills) {
		return ProfileInput{}, fmt.Errorf("%w: Agent Profile contains a duplicate Skill reference", ErrInvalid)
	}
	input.Skills = canonicalSkillReferences(input.Skills)
	input.MinimumTargetCapabilities = canonicalStrings(input.MinimumTargetCapabilities)
	input.SupportedModes = canonicalModes(input.SupportedModes)
	input.DefaultForModes = canonicalModes(input.DefaultForModes)
	if input.DefaultTarget != nil && validateTarget(*input.DefaultTarget) != nil {
		return ProfileInput{}, fmt.Errorf("%w: Agent Profile default target is invalid", ErrInvalid)
	}
	if !validName(input.Name) || len(input.Description) > 4096 ||
		input.ApprovalPolicy != "required" || input.MaximumTurnSeconds < 10 || input.MaximumTurnSeconds > 86400 ||
		input.MaximumToolSteps < 1 || input.MaximumToolSteps > 256 ||
		input.ContextTokenBudget < 1024 || input.ContextTokenBudget > 1048576 ||
		len(input.ToolPolicy.Allow) == 0 || len(input.SupportedModes) == 0 ||
		!modesContained(input.DefaultForModes, input.SupportedModes) {
		return ProfileInput{}, fmt.Errorf("%w: Agent Profile is invalid", ErrInvalid)
	}
	for _, pattern := range append(append([]string(nil), input.ToolPolicy.Allow...), input.ToolPolicy.Deny...) {
		if !canonicalToolPattern(pattern) {
			return ProfileInput{}, fmt.Errorf("%w: invalid tool policy pattern", ErrInvalid)
		}
	}
	for _, skill := range input.Skills {
		if uuid.Validate(skill.ID) != nil || skill.Revision < 1 {
			return ProfileInput{}, fmt.Errorf("%w: invalid Skill reference", ErrInvalid)
		}
	}
	return input, nil
}

func NormalizeSkillInput(input SkillInput) (SkillInput, error) {
	input.Name = strings.TrimSpace(input.Name)
	input.Description = strings.TrimSpace(input.Description)
	input.Instructions = strings.TrimSpace(input.Instructions)
	input.RequiredTools = canonicalStrings(input.RequiredTools)
	input.MinimumCapabilities = canonicalStrings(input.MinimumCapabilities)
	if !validName(input.Name) || len(input.Description) > 4096 ||
		len(input.Instructions) == 0 || len(input.Instructions) > 262144 {
		return SkillInput{}, fmt.Errorf("%w: Agent Skill is invalid", ErrInvalid)
	}
	for _, tool := range input.RequiredTools {
		if !canonicalToolName(tool) {
			return SkillInput{}, fmt.Errorf("%w: Skill requires an invalid tool", ErrInvalid)
		}
	}
	return input, nil
}

// NormalizeToolSourceInput validates the transport-independent syntax only.
// Service mutations additionally require ToolSourcePolicyValidator, which
// compiles and authorizes this policy through the shared egress boundary.
func NormalizeToolSourceInput(input ToolSourceInput) (ToolSourceInput, error) {
	input.Name = strings.TrimSpace(input.Name)
	input.Description = strings.TrimSpace(input.Description)
	input.Endpoint = strings.TrimSpace(input.Endpoint)
	input.CredentialID = strings.TrimSpace(input.CredentialID)
	parsed, err := url.Parse(input.Endpoint)
	if !validName(input.Name) || len(input.Description) > 4096 || input.Kind != "remote" ||
		input.Transport != "streamable_http" ||
		err != nil || parsed.Scheme != "https" || parsed.Host == "" || parsed.User != nil ||
		parsed.RawQuery != "" || parsed.Fragment != "" {
		return ToolSourceInput{}, fmt.Errorf("%w: Agent Tool Source is invalid", ErrInvalid)
	}
	input.Endpoint = parsed.String()
	if len(input.EgressPolicy.AllowedHosts) == 0 || len(input.EgressPolicy.AllowedHosts) > 64 ||
		len(input.EgressPolicy.AllowedPorts) > 32 || len(input.EgressPolicy.AllowedPrivateCIDRs) > 32 {
		return ToolSourceInput{}, fmt.Errorf("%w: Tool Source egress policy is invalid", ErrInvalid)
	}
	if input.CredentialID != "" && uuid.Validate(input.CredentialID) != nil {
		return ToolSourceInput{}, fmt.Errorf("%w: Tool credential reference is invalid", ErrInvalid)
	}
	return input, nil
}

func ValidateTurnInput(input TurnInput) error {
	if len(input.Content) == 0 || len(input.Content) > 64 {
		return fmt.Errorf("%w: turn content must contain 1-64 blocks", ErrInvalid)
	}
	totalBytes := 0
	for _, block := range input.Content {
		switch block.Type {
		case "text":
			if strings.TrimSpace(block.Text) == "" || block.URL != "" || block.FileID != "" {
				return fmt.Errorf("%w: invalid text block", ErrInvalid)
			}
			totalBytes += len(block.Text)
		case "image_url":
			parsed, err := url.Parse(block.URL)
			if err != nil || parsed.Scheme != "https" || parsed.Host == "" || parsed.User != nil ||
				parsed.RawQuery != "" || parsed.Fragment != "" ||
				(block.Detail != "" && block.Detail != "auto" && block.Detail != "low" && block.Detail != "high") {
				return fmt.Errorf("%w: invalid image URL block", ErrInvalid)
			}
			totalBytes += len(block.URL)
		case "file_reference":
			if uuid.Validate(block.FileID) != nil || block.Text != "" || block.URL != "" {
				return fmt.Errorf("%w: invalid file reference block", ErrInvalid)
			}
		default:
			return fmt.Errorf("%w: unsupported content block %q", ErrUnsupported, block.Type)
		}
	}
	if totalBytes > 1<<20 {
		return fmt.Errorf("%w: turn content exceeds one MiB", ErrInvalid)
	}
	return nil
}

// NormalizeTurnInput is the single durable input boundary. User-provided
// credentials are replaced before either the Turn row or transcript event is
// persisted; signed/query-bearing image URLs are rejected because their query
// cannot be safely distinguished from an access token.
func NormalizeTurnInput(input TurnInput) (TurnInput, error) {
	result := TurnInput{Content: append([]ContentBlock(nil), input.Content...)}
	if err := ValidateTurnInput(result); err != nil {
		return TurnInput{}, err
	}
	for index := range result.Content {
		if result.Content[index].Type == "text" {
			result.Content[index].Text = transcriptCredentialPattern.ReplaceAllString(
				result.Content[index].Text, "[redacted credential]",
			)
		}
	}
	return result, nil
}

func validateTarget(target Target) error {
	if (target.Kind != TargetModel && target.Kind != TargetEntrypoint) || !validRequestFacingTargetID(target.ID) {
		return fmt.Errorf("%w: Agent target is invalid", ErrInvalid)
	}
	return nil
}

func validRequestFacingTargetID(value string) bool {
	if len(value) < 1 || len(value) > 256 || strings.TrimSpace(value) != value || strings.ContainsAny(value, "\x00\r\n\t ") {
		return false
	}
	for _, character := range value {
		if character > 127 || (character < 'a' || character > 'z') &&
			(character < 'A' || character > 'Z') && (character < '0' || character > '9') &&
			!strings.ContainsRune("._:/-", character) {
			return false
		}
	}
	return true
}

func validName(value string) bool {
	if value == "" || len(value) > 160 || strings.TrimSpace(value) != value || !utf8.ValidString(value) {
		return false
	}
	for _, character := range value {
		if !unicode.IsGraphic(character) {
			return false
		}
	}
	return true
}

func canonicalStrings(values []string) []string {
	seen := make(map[string]bool, len(values))
	result := make([]string, 0, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value != "" && !seen[value] {
			seen[value] = true
			result = append(result, value)
		}
	}
	sort.Strings(result)
	return result
}

func canonicalSkillReferences(values []SkillReference) []SkillReference {
	result := append([]SkillReference(nil), values...)
	sort.Slice(result, func(i, j int) bool {
		if result[i].ID == result[j].ID {
			return result[i].Revision < result[j].Revision
		}
		return result[i].ID < result[j].ID
	})
	compacted := result[:0]
	for _, value := range result {
		if len(compacted) == 0 || compacted[len(compacted)-1].ID != value.ID {
			compacted = append(compacted, value)
		}
	}
	return compacted
}

func duplicateSkillReference(values []SkillReference) bool {
	seen := make(map[string]struct{}, len(values))
	for _, value := range values {
		if _, duplicate := seen[value.ID]; duplicate {
			return true
		}
		seen[value.ID] = struct{}{}
	}
	return false
}

func canonicalModes(values []SessionMode) []SessionMode {
	seen := make(map[SessionMode]struct{}, len(values))
	result := make([]SessionMode, 0, len(values))
	for _, value := range values {
		if value != SessionChat && value != SessionBuilder {
			return nil
		}
		if _, duplicate := seen[value]; duplicate {
			continue
		}
		seen[value] = struct{}{}
		result = append(result, value)
	}
	sort.Slice(result, func(i, j int) bool { return result[i] < result[j] })
	return result
}

func modesContained(values, container []SessionMode) bool {
	allowed := make(map[SessionMode]struct{}, len(container))
	for _, value := range container {
		allowed[value] = struct{}{}
	}
	for _, value := range values {
		if _, found := allowed[value]; !found {
			return false
		}
	}
	return true
}

func canonicalToolPattern(value string) bool {
	if canonicalToolName(value) {
		return true
	}
	return strings.HasSuffix(value, ".*") && canonicalToolName(strings.TrimSuffix(value, ".*"))
}
