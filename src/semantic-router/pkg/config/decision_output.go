package config

import (
	"encoding/json"
	"fmt"
	"slices"
	"strings"
)

// The kinds and actions of a decision's output contract.
const (
	OutputKindText  = "text"
	OutputKindImage = "image"
	OutputKindAudio = "audio"
	OutputKindVideo = "video"

	OutputActionGenerate = "generate"
	OutputActionEdit     = "edit"
)

// OutputKindNames lists the kinds an output contract may name, in schema order.
func OutputKindNames() []string {
	return []string{OutputKindText, OutputKindImage, OutputKindAudio, OutputKindVideo}
}

// OutputActionNames lists the actions an output contract may name.
func OutputActionNames() []string {
	return []string{OutputActionGenerate, OutputActionEdit}
}

// DecisionOutput is a decision's output contract: the media its answers carry
// and whether they generate media or edit the media the request provides. A
// decision without one answers in text.
type DecisionOutput struct {
	Kind   OutputKinds `yaml:"kind,omitempty" json:"kind,omitempty"`
	Action string      `yaml:"action,omitempty" json:"action,omitempty" jsonschema:"enum=generate,enum=edit"`
}

// OutputKinds is one kind, written as a scalar, or several, written as a list.
type OutputKinds []string

func (kinds *OutputKinds) UnmarshalYAML(unmarshal func(interface{}) error) error {
	var one string
	if unmarshal(&one) == nil {
		*kinds = OutputKinds{one}
		return nil
	}
	list := []string{}
	if err := unmarshal(&list); err != nil {
		return fmt.Errorf("kind must be one of %s, or a list of them", strings.Join(OutputKindNames(), ", "))
	}
	*kinds = append(OutputKinds{}, list...)
	return nil
}

func (kinds OutputKinds) MarshalYAML() (interface{}, error) {
	if len(kinds) == 1 {
		return kinds[0], nil
	}
	return []string(kinds), nil
}

func (kinds *OutputKinds) UnmarshalJSON(data []byte) error {
	var one string
	if json.Unmarshal(data, &one) == nil {
		*kinds = OutputKinds{one}
		return nil
	}
	var list []string
	if err := json.Unmarshal(data, &list); err != nil {
		return fmt.Errorf("kind must be one of %s, or a list of them", strings.Join(OutputKindNames(), ", "))
	}
	*kinds = list
	return nil
}

func (kinds OutputKinds) MarshalJSON() ([]byte, error) {
	if len(kinds) == 1 {
		return json.Marshal(kinds[0])
	}
	return json.Marshal([]string(kinds))
}

// Kinds are the contract's kinds; a decision without a contract answers text.
func (o *DecisionOutput) Kinds() []string {
	if o == nil || len(o.Kind) == 0 {
		return []string{OutputKindText}
	}
	return slices.Clone(o.Kind)
}

// EffectiveAction is the contract's action; generate unless it edits.
func (o *DecisionOutput) EffectiveAction() string {
	if o == nil || o.Action == "" {
		return OutputActionGenerate
	}
	return o.Action
}

// Includes reports whether the contract's answers carry kind.
func (o *DecisionOutput) Includes(kind string) bool {
	return slices.Contains(o.Kinds(), kind)
}

// GeneratesImages reports whether the contract's answers carry a generated
// image, which the hosted image_generation operation produces.
func (o *DecisionOutput) GeneratesImages() bool {
	return o.Includes(OutputKindImage) && o.EffectiveAction() == OutputActionGenerate
}

// GeneratesVideos reports whether the contract's answers are a generated
// video, which the hosted video_generation operation produces.
func (o *DecisionOutput) GeneratesVideos() bool {
	return o.Includes(OutputKindVideo) && o.EffectiveAction() == OutputActionGenerate
}

// ImageOnly reports whether the contract's answers are a generated image and
// nothing else.
func (o *DecisionOutput) ImageOnly() bool {
	kinds := o.Kinds()
	return len(kinds) == 1 && kinds[0] == OutputKindImage && o.EffectiveAction() == OutputActionGenerate
}

// String writes the contract's kinds as the configuration names them: one
// kind, or a list such as [text, image].
func (o *DecisionOutput) String() string {
	kinds := o.Kinds()
	if len(kinds) == 1 {
		return kinds[0]
	}
	return "[" + strings.Join(kinds, ", ") + "]"
}

// requiredCapabilities are the capabilities every model a contract references
// must declare.
func (o *DecisionOutput) requiredCapabilities() []string {
	edit := o.EffectiveAction() == OutputActionEdit
	var required []string
	for _, kind := range o.Kinds() {
		switch kind {
		case OutputKindImage:
			required = append(required, "image_generation")
			if edit {
				required = append(required, "image_input")
			}
		case OutputKindAudio:
			required = append(required, "audio_output")
		case OutputKindVideo:
			if edit {
				required = append(required, "video_output")
				required = append(required, "video_input")
			} else {
				required = append(required, "video_generation")
			}
		}
	}
	return required
}

// missingCodec names the codec a contract needs that this release does not
// have, or "" when the Router executes it: text, image generation, and text
// with a generated image from one model.
func (o *DecisionOutput) missingCodec() string {
	edit := o.EffectiveAction() == OutputActionEdit
	for _, kind := range o.Kinds() {
		switch {
		case kind == OutputKindImage && edit:
			return "the image edit codec"
		case kind == OutputKindAudio:
			return "the audio output codec"
		case kind == OutputKindVideo && edit:
			return "the video codec"
		}
	}
	return ""
}

// Validate refuses a contract that names an output the schema does not have.
func (o *DecisionOutput) Validate() error {
	if o.Kind != nil && len(o.Kind) == 0 {
		return fmt.Errorf("kind lists no output")
	}
	seen := map[string]bool{}
	for _, kind := range o.Kind {
		if !slices.Contains(OutputKindNames(), kind) {
			return fmt.Errorf("kind %q is not one of %s", kind, strings.Join(OutputKindNames(), ", "))
		}
		if seen[kind] {
			return fmt.Errorf("kind lists %s twice", kind)
		}
		seen[kind] = true
	}
	switch o.Action {
	case "", OutputActionGenerate:
	case OutputActionEdit:
		if !o.Includes(OutputKindImage) && !o.Includes(OutputKindVideo) {
			return fmt.Errorf("action edit edits an image or a video the request provides, and kind %s has neither", o)
		}
	default:
		return fmt.Errorf("action %q is not one of %s", o.Action, strings.Join(OutputActionNames(), ", "))
	}
	return nil
}
