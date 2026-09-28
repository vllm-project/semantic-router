// Package shadowdataset turns captured primary and shadow observations into a
// versioned comparison dataset.
//
// A manifest holds identity and digests, never payload text, so it can be
// published without carrying the prompts and responses it describes. It is
// content addressed: Build derives Digest from the policy and the examples, and
// the same records under the same policy produce the same manifest, including
// the split every example lands in. Nothing here reads a live route or writes
// one, and a manifest is never repaired in place. A different rule set is a
// different manifest.
//
// Both sides of a comparison arrive the same way, as an outcome the router
// wrote carrying a digest of the assistant text. Neither side is hashed here
// from a stored body, because the stored body is the encoded protocol response
// and hashing it on one side only would give two models different digests for
// the same answer.
//
// An observation enters the dataset whole or not at all. A request that failed,
// is unfinished, lost its input to truncation, or never had a digest recorded
// is excluded under a counted reason, so a manifest needs no per-example
// capture state to qualify one.
package shadowdataset

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

// ManifestVersion identifies the manifest shape. A change to what Build emits
// or to how it assigns splits changes this string, because consumers compare
// numbers across manifests and have to know when that comparison stopped being
// meaningful.
const ManifestVersion = "shadow-dataset.v1"

// shadowDispatchSource is the Outcome.Source that the shadow dispatch plugin
// writes. Outcomes from anything else describe a different experiment.
const shadowDispatchSource = "shadow_dispatch"

// primaryResponseSource is the Outcome.Source the router writes for the answer
// the selected model gave. Both sides of a comparison therefore arrive as an
// outcome carrying a digest of the same thing, the assistant text of the
// decoded response. The stored response body is not that: it is the encoded
// protocol envelope with an id and a usage block, so two models answering
// identically would hash differently if this read the body instead.
const primaryResponseSource = "primary_response"

// shadowVerdictCompleted marks a shadow arm that answered. A failed or dropped
// arm is evidence about dispatch, not about the model, so it is not comparable
// output and does not enter the dataset.
const shadowVerdictCompleted = "completed"

// Exclusion reasons. Every record Build leaves out is counted under one of
// these, so a manifest states what it dropped rather than only what it kept.
const (
	ExcludeNoShadowArm         = "no_completed_shadow_arm"
	ExcludeNoPrimaryArm        = "no_primary_response_digest"
	ExcludePrimaryUnsuccessful = "primary_request_unsuccessful"
	ExcludeRequestMissing      = "request_body_missing"
	ExcludeRequestTruncated    = "request_body_truncated"
	ExcludePrimaryModelMissing = "primary_model_missing"
	ExcludeDuplicateInput      = "duplicate_input"
)

// Split is one named part of the dataset. Weights are whole numbers so a split
// plan has no rounding to argue about.
type Split struct {
	Name   string `json:"name"`
	Weight int    `json:"weight"`
}

// Policy is the rule set a manifest was built under. It is covered by the
// digest, so a manifest cannot be reproduced under a policy it was not built
// with.
type Policy struct {
	// Seed fixes the split assignment. Two manifests with different seeds put
	// the same input in different splits, which is the point of recording it.
	Seed   string  `json:"seed"`
	Splits []Split `json:"splits"`
}

// Arm is one model's side of a comparison. OutputDigest is the hash of the text
// that model produced, never the text.
type Arm struct {
	Model        string `json:"model"`
	Backend      string `json:"backend,omitempty"`
	OutputDigest string `json:"output_digest"`
	SampleRate   string `json:"sample_rate,omitempty"`
}

// Lineage points back at the observation an example came from. It carries no
// payload, so it is safe to publish next to the digests.
type Lineage struct {
	ReplayID  string `json:"replay_id"`
	RequestID string `json:"request_id,omitempty"`
	Recipe    string `json:"recipe,omitempty"`
	Decision  string `json:"decision,omitempty"`
}

// Example is one input with the primary answer and every shadow answer to it.
//
// There is no field for capture or redaction state because an example is built
// only from a successful observation whose input survived intact and whose
// arms each recorded a digest. Everything else is excluded and counted, so a
// digest in a manifest always stands for the whole answer a model gave.
type Example struct {
	ID          string  `json:"id"`
	InputDigest string  `json:"input_digest"`
	Split       string  `json:"split"`
	Primary     Arm     `json:"primary"`
	Shadows     []Arm   `json:"shadows"`
	Lineage     Lineage `json:"lineage"`
}

// Counts reports what Build kept and what it dropped, by reason.
type Counts struct {
	Records  int            `json:"records"`
	Examples int            `json:"examples"`
	Excluded map[string]int `json:"excluded,omitempty"`
}

// Manifest is the published artifact.
type Manifest struct {
	Version  string    `json:"version"`
	Digest   string    `json:"digest"`
	Policy   Policy    `json:"policy"`
	Counts   Counts    `json:"counts"`
	Examples []Example `json:"examples"`
}

// Build derives a manifest from captured observations. It does not read the
// store, so the caller decides which records are approved for export.
func Build(records []store.Record, policy Policy) (Manifest, error) {
	if err := policy.validate(); err != nil {
		return Manifest{}, err
	}

	counts := Counts{Records: len(records), Excluded: map[string]int{}}
	examples := make([]Example, 0, len(records))
	for _, rec := range sortedRecords(records) {
		example, reason := buildExample(rec)
		if reason != "" {
			counts.Excluded[reason]++
			continue
		}
		examples = append(examples, example)
	}

	examples = dropDuplicateInputs(examples, counts.Excluded)
	for i := range examples {
		examples[i].Split = policy.splitFor(examples[i].ID)
	}
	sort.Slice(examples, func(i, j int) bool { return examples[i].ID < examples[j].ID })
	counts.Examples = len(examples)
	if len(counts.Excluded) == 0 {
		counts.Excluded = nil
	}

	manifest := Manifest{Version: ManifestVersion, Policy: policy, Counts: counts, Examples: examples}
	digest, err := manifest.contentDigest()
	if err != nil {
		return Manifest{}, err
	}
	manifest.Digest = digest
	return manifest, nil
}

// Validate rejects a manifest that cannot be trusted as evidence. It runs
// against a manifest read back from wherever it was published, so it repeats
// the checks Build already satisfied rather than assuming Build produced it.
func Validate(m Manifest) error {
	if m.Version != ManifestVersion {
		return fmt.Errorf("manifest version %q, want %q", m.Version, ManifestVersion)
	}
	if err := m.Policy.validate(); err != nil {
		return err
	}

	declared := make(map[string]struct{}, len(m.Policy.Splits))
	for _, split := range m.Policy.Splits {
		declared[split.Name] = struct{}{}
	}

	seenInput := make(map[string]string, len(m.Examples))
	for _, example := range m.Examples {
		if err := example.validate(declared); err != nil {
			return err
		}
		if previous, ok := seenInput[example.InputDigest]; ok {
			return fmt.Errorf("input %s appears in splits %q and %q", example.InputDigest, previous, example.Split)
		}
		seenInput[example.InputDigest] = example.Split
	}

	if m.Counts.Examples != len(m.Examples) {
		return fmt.Errorf("manifest counts %d examples, carries %d", m.Counts.Examples, len(m.Examples))
	}

	digest, err := m.contentDigest()
	if err != nil {
		return err
	}
	if digest != m.Digest {
		return fmt.Errorf("manifest digest %s, content hashes to %s", m.Digest, digest)
	}
	return nil
}

func (e Example) validate(declaredSplits map[string]struct{}) error {
	if _, ok := declaredSplits[e.Split]; !ok {
		return fmt.Errorf("example %s sits in undeclared split %q", e.ID, e.Split)
	}
	if e.InputDigest == "" {
		return fmt.Errorf("example %s has no input digest", e.ID)
	}
	if e.Primary.Model == "" || e.Primary.OutputDigest == "" {
		return fmt.Errorf("example %s has no primary arm identity", e.ID)
	}
	if len(e.Shadows) == 0 {
		return fmt.Errorf("example %s has no shadow arm", e.ID)
	}
	for _, shadow := range e.Shadows {
		if shadow.Model == "" || shadow.OutputDigest == "" {
			return fmt.Errorf("example %s has a shadow arm without identity", e.ID)
		}
	}
	if e.Lineage.ReplayID == "" {
		return fmt.Errorf("example %s has no lineage", e.ID)
	}
	if e.ID != exampleID(e.InputDigest, e.Primary.Model, e.Lineage.ReplayID) {
		return fmt.Errorf("example %s does not match its own identity", e.ID)
	}
	return nil
}

func (p Policy) validate() error {
	if p.Seed == "" {
		return fmt.Errorf("policy needs a seed, otherwise the split cannot be reproduced")
	}
	if len(p.Splits) == 0 {
		return fmt.Errorf("policy needs at least one split")
	}
	seen := make(map[string]struct{}, len(p.Splits))
	for _, split := range p.Splits {
		if split.Name == "" {
			return fmt.Errorf("policy has an unnamed split")
		}
		if split.Weight <= 0 {
			return fmt.Errorf("split %q has weight %d, want a positive weight", split.Name, split.Weight)
		}
		if _, ok := seen[split.Name]; ok {
			return fmt.Errorf("policy declares split %q twice", split.Name)
		}
		seen[split.Name] = struct{}{}
	}
	return nil
}

// splitFor places an example by hashing the seed with its identity. The split
// therefore follows the input rather than the order records arrived in, so
// adding observations never moves an example that was already placed.
func (p Policy) splitFor(exampleID string) string {
	last := p.Splits[len(p.Splits)-1].Name
	var total uint64
	for _, split := range p.Splits {
		if split.Weight <= 0 {
			continue
		}
		total += uint64(split.Weight)
	}
	if total == 0 {
		return last
	}

	sum := sha256.Sum256([]byte(p.Seed + "\x00" + exampleID))
	point := binary.BigEndian.Uint64(sum[:8]) % total
	for _, split := range p.Splits {
		if split.Weight <= 0 {
			continue
		}
		weight := uint64(split.Weight)
		if point < weight {
			return split.Name
		}
		point -= weight
	}
	return last
}

func buildExample(rec store.Record) (Example, string) {
	// A shadow arm that failed is excluded, so a primary that failed has to be
	// too. The router keeps the error body of a 4xx or 5xx in the record, and
	// without this check that body would be published as the model's answer.
	if !primarySucceeded(rec) {
		return Example{}, ExcludePrimaryUnsuccessful
	}
	shadows := completedShadowArms(rec)
	if len(shadows) == 0 {
		return Example{}, ExcludeNoShadowArm
	}
	if rec.SelectedModel == "" {
		return Example{}, ExcludePrimaryModelMissing
	}
	if rec.RequestBody == "" {
		return Example{}, ExcludeRequestMissing
	}
	if rec.RequestBodyTruncated {
		return Example{}, ExcludeRequestTruncated
	}
	primary, ok := primaryArm(rec)
	if !ok {
		return Example{}, ExcludeNoPrimaryArm
	}

	inputDigest := digestOf(rec.RequestBody)
	return Example{
		ID:          exampleID(inputDigest, primary.Model, rec.ID),
		InputDigest: inputDigest,
		Primary:     primary,
		Shadows:     shadows,
		Lineage: Lineage{
			ReplayID:  rec.ID,
			RequestID: rec.RequestID,
			Recipe:    rec.Recipe,
			Decision:  rec.Decision,
		},
	}, ""
}

// primarySucceeded reports whether the primary request is finished and was
// answered. An unfinished record is not evidence yet, and a non 2xx answer is
// a transport failure rather than something the model said.
func primarySucceeded(rec store.Record) bool {
	if rec.LifecycleState != store.LifecycleCompleted {
		return false
	}
	return rec.ResponseStatus >= 200 && rec.ResponseStatus < 300
}

// primaryArm reads the digest the router recorded for the selected model. It
// deliberately does not hash rec.ResponseBody, which holds the encoded protocol
// response rather than the assistant text a shadow arm is hashed from.
//
// The model comes from the record rather than the outcome. The example id is
// built from it, and the id decides the split, so a name that could arrive from
// either of two fields would let the same observation land in two places.
func primaryArm(rec store.Record) (Arm, bool) {
	for _, outcome := range rec.Outcomes {
		if outcome.Source != primaryResponseSource || outcome.Verdict != shadowVerdictCompleted {
			continue
		}
		digest := outcome.Metadata["response_sha256"]
		if digest == "" {
			continue
		}
		return Arm{Model: rec.SelectedModel, OutputDigest: digest}, true
	}
	return Arm{}, false
}

// completedShadowArms reads the arms the shadow dispatch plugin recorded. The
// response hash is the one the plugin already wrote, so the dataset compares
// what was actually sent rather than a rehash of a stored excerpt.
func completedShadowArms(rec store.Record) []Arm {
	arms := make([]Arm, 0, len(rec.Outcomes))
	for _, outcome := range rec.Outcomes {
		if outcome.Source != shadowDispatchSource || outcome.Verdict != shadowVerdictCompleted {
			continue
		}
		digest := outcome.Metadata["response_sha256"]
		model := firstNonEmpty(outcome.Metadata["shadow_model"], outcome.TargetRef)
		if digest == "" || model == "" {
			continue
		}
		arms = append(arms, Arm{
			Model:        model,
			Backend:      outcome.Metadata["shadow_backend"],
			OutputDigest: digest,
			SampleRate:   outcome.Metadata["sample_rate"],
		})
	}
	sort.Slice(arms, func(i, j int) bool {
		if arms[i].Model != arms[j].Model {
			return arms[i].Model < arms[j].Model
		}
		return arms[i].OutputDigest < arms[j].OutputDigest
	})
	return arms
}

// dropDuplicateInputs keeps one example per input. The kept one is the first in
// record order, which sortedRecords already fixed, so the choice does not move
// between runs.
func dropDuplicateInputs(examples []Example, excluded map[string]int) []Example {
	seen := make(map[string]struct{}, len(examples))
	kept := examples[:0]
	for _, example := range examples {
		if _, ok := seen[example.InputDigest]; ok {
			excluded[ExcludeDuplicateInput]++
			continue
		}
		seen[example.InputDigest] = struct{}{}
		kept = append(kept, example)
	}
	return kept
}

// sortedRecords fixes the order Build reads records in. A store lists newest
// first and a caller may concatenate pages, so without this the surviving row
// of a duplicated input would depend on how the records were fetched.
func sortedRecords(records []store.Record) []store.Record {
	ordered := append([]store.Record(nil), records...)
	sort.SliceStable(ordered, func(i, j int) bool {
		if !ordered[i].Timestamp.Equal(ordered[j].Timestamp) {
			return ordered[i].Timestamp.Before(ordered[j].Timestamp)
		}
		return ordered[i].ID < ordered[j].ID
	})
	return ordered
}

// contentDigest hashes the policy and the examples. The digest field itself and
// the counts are left out, so a manifest can be verified by rebuilding its
// content and comparing, without having to reproduce the exclusion tally.
func (m Manifest) contentDigest() (string, error) {
	payload := struct {
		Version  string    `json:"version"`
		Policy   Policy    `json:"policy"`
		Examples []Example `json:"examples"`
	}{Version: m.Version, Policy: m.Policy, Examples: m.Examples}
	encoded, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("hash manifest content: %w", err)
	}
	sum := sha256.Sum256(encoded)
	return hex.EncodeToString(sum[:]), nil
}

func exampleID(inputDigest, primaryModel, replayID string) string {
	return digestOf(strings.Join([]string{inputDigest, primaryModel, replayID}, "\x00"))
}

func digestOf(value string) string {
	sum := sha256.Sum256([]byte(value))
	return hex.EncodeToString(sum[:])
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}
