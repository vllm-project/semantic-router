package config

import (
	"fmt"
	"slices"
	"strings"

	"gopkg.in/yaml.v3"
)

// multimodalForms are the forms of a decision's output before decisions
// declared it: a model's modality keyword (ar, diffusion, omni), a decision
// routed on image requests that took its output from such a model, and the
// image_gen plugin. The decision forms come first: they read the keywords
// the model form rewrites.
var multimodalForms = []ConfigForm{
	{
		Code:     CodeRemovedField,
		Paths:    RoutingScopes("decisions[]"),
		MatchAt:  func(at FormMatch) bool { kind, rewritable := legacyImageRoute(at); return kind != nil && rewritable },
		Describe: describeLegacyImageRoute,
		Migrate: func(m *Migration, at FormMatch) error {
			kind, _ := legacyImageRoute(at)
			insertAfter(at.Node, "rules", "output", flowMapping("kind", kind, "action", OutputActionGenerate))
			reachOutputAPI(m, at, kind.Kind == yaml.SequenceNode)
			return nil
		},
	},
	{
		Code:     CodeRemovedField,
		Paths:    RoutingScopes("decisions[]"),
		MatchAt:  func(at FormMatch) bool { kind, rewritable := legacyImageRoute(at); return kind != nil && !rewritable },
		Describe: describeLegacyImageRoute,
		Migrate:  dropLegacyImageRoute,
	},
	{
		Code:     CodeRemovedField,
		Paths:    []string{"providers.models[].modality"},
		Describe: describeModalityKeyword,
		Migrate:  migrateModalityKeyword,
	},
	{
		Code:    CodeRemovedPluginType,
		Paths:   RoutingScopes("decisions[].plugins[]"),
		Match:   hasType("image_gen"),
		What:    "plugin image_gen was removed",
		Becomes: RemovedPluginTypes["image_gen"],
	},
}

// legacyImageRoute is the output kind a decision without an output contract
// took from the modality keywords of its models: text with an image when it
// routes text_image requests, an image when it routes image requests or
// references a diffusion model, and nil for any decision that references no
// diffusion or omni model. It also says whether every model the decision
// references writes that output once the keywords are rewritten.
func legacyImageRoute(at FormMatch) (*yaml.Node, bool) {
	decision := at.Node
	recipe := decisionRecipe(at)
	if recipe == nil || mappingValue(decision, "output") != nil {
		return nil, false
	}
	models := providerModelsByName(at.Root)
	legacy, diffusion := false, false
	for _, ref := range sequenceAt(decision, "modelRefs") {
		switch strings.ToLower(scalarAt(models[scalarAt(ref, "model")], "modality")) {
		case "diffusion":
			legacy, diffusion = true, true
		case "omni":
			legacy = true
		}
	}
	if !legacy {
		return nil, false
	}
	var read []string
	positiveModalityLabels(mappingValue(decision, "rules"), modalityRuleLabels(recipe), false, &read)
	textAndImage := slices.Contains(read, "text_image")
	if !textAndImage && !diffusion && !slices.Contains(read, OutputKindImage) {
		return nil, false
	}
	rewritable := true
	for _, ref := range sequenceAt(decision, "modelRefs") {
		text, images := modelWrites(models[scalarAt(ref, "model")])
		rewritable = rewritable && images && (text || !textAndImage)
	}
	if textAndImage {
		return stringSequence(OutputKindText, OutputKindImage), rewritable
	}
	return Scalar(OutputKindImage), rewritable
}

func describeLegacyImageRoute(at FormMatch) (string, string) {
	kind, rewritable := legacyImageRoute(at)
	textAndImage := kind != nil && kind.Kind == yaml.SequenceNode
	what := "a decision's output no longer follows the modality of its models"
	switch {
	case textAndImage && rewritable:
		return what, "output: {kind: [text, image], action: generate}"
	case textAndImage:
		return what, "nothing: migrate drops the decision, since a model it references cannot answer in text and generate images; " +
			"reference models on api_format: responses that do both, with output: {kind: [text, image], action: generate}"
	case rewritable:
		return what, "output: {kind: image, action: generate}"
	default:
		return what, "nothing: migrate drops the decision, since a model it references generates no images; " +
			"reference image models only, with output: {kind: image, action: generate}"
	}
}

// reachOutputAPI points each model the image route at references to an API
// that carries its output, when it answered on one that cannot: the images
// API for a model that only generates images, the Responses API for one that
// also answers in text. The note asks to check that the backend serves it.
func reachOutputAPI(m *Migration, at FormMatch, withText bool) {
	models := providerModelsByName(at.Root)
	for _, ref := range sequenceAt(at.Node, "modelRefs") {
		name := scalarAt(ref, "model")
		model := models[name]
		written := strings.ToLower(scalarAt(model, "api_format"))
		text, _ := modelWrites(model)
		format, api := APIFormatImages, "POST /v1/images/generations"
		if withText || text {
			format, api = APIFormatResponses, "POST /v1/responses"
		}
		if model == nil || written == APIFormatResponses || (written == APIFormatImages && format == APIFormatImages) {
			continue
		}
		if written == "" {
			written = APIFormatOpenAI
		}
		setNode(model, "api_format", Scalar(format))
		m.Note("providers.models["+name+"].api_format", fmt.Sprintf("it was %s, which cannot carry a generated image, and decision %s takes its output from the model: "+
			"api_format is %s now; check that the model's backend serves %s", written, scalarAt(at.Node, "name"), format, api))
	}
}

// dropLegacyImageRoute drops a decision that took text with an image, or an
// image, from models that do not write it: no decision of this release
// executes it, so the requests it matched go to the other decisions.
func dropLegacyImageRoute(m *Migration, at FormMatch) error {
	kind, _ := legacyImageRoute(at)
	textAndImage := kind.Kind == yaml.SequenceNode
	models := providerModelsByName(at.Root)
	var unable []string
	for _, ref := range sequenceAt(at.Node, "modelRefs") {
		if text, images := modelWrites(models[scalarAt(ref, "model")]); !images || (textAndImage && !text) {
			unable = append(unable, scalarAt(ref, "model"))
		}
	}
	removeItem(at.Parent, at.Node)
	if textAndImage {
		m.Note(at.Path, fmt.Sprintf("the decision is dropped: it answered with text and an image, and %s cannot do both, so no decision of this release executes it; "+
			"the requests it matched go to the other decisions. To keep it, reference models on api_format: responses that answer in text and generate images, "+
			"with output: {kind: [text, image], action: generate}", strings.Join(unable, ", ")))
		return nil
	}
	m.Note(at.Path, fmt.Sprintf("the decision is dropped: it routed image requests, and %s generates no images; the requests it matched go to the other decisions. "+
		"To keep it, reference image models only, with output: {kind: image, action: generate}", strings.Join(unable, ", ")))
	return nil
}

// modelWrites says whether a providers.models[] entry answers in text and
// generates images once its modality keyword is rewritten.
func modelWrites(model *yaml.Node) (text, images bool) {
	diffusion := strings.EqualFold(scalarAt(model, "modality"), "diffusion")
	images = diffusion || slices.Contains(stringItems(mappingValue(model, "capabilities")), "image_generation")
	if output := mappingValue(mappingValue(model, "modalities"), "output"); output != nil {
		return slices.Contains(stringItems(output), OutputKindText), images
	}
	return !diffusion, images
}

func describeModalityKeyword(at FormMatch) (string, string) {
	switch keyword := strings.ToLower(at.Node.Value); keyword {
	case "diffusion":
		return "modality: diffusion is removed", "capabilities: [image_generation] and modalities: {input: [text], output: [image]}"
	case "ar":
		return "modality: ar is removed", "nothing: a model answers in text unless its modalities say otherwise"
	case "omni":
		return "modality: omni is removed", "modalities naming what the model reads and writes"
	default:
		return fmt.Sprintf("modality: %s is removed", at.Node.Value), "capabilities and modalities"
	}
}

// migrateModalityKeyword states a model's modality keyword as the facts the
// Router reads: a diffusion model generates images from text; an ar model is
// the default; an omni model reads images and answers in text, unless it
// declares image generation, which the note asks to check.
func migrateModalityKeyword(m *Migration, at FormMatch) error {
	model := at.Parent
	keyword := strings.ToLower(at.Node.Value)
	m.Delete(model, "modality")
	modalities := mappingValue(model, "modalities")
	switch keyword {
	case "diffusion":
		capabilities := resolveAlias(mappingValue(model, "capabilities"))
		switch {
		case capabilities == nil || capabilities.Kind != yaml.SequenceNode:
			setNode(model, "capabilities", stringSequence("image_generation"))
		case !slices.Contains(stringItems(capabilities), "image_generation"):
			capabilities.Content = append(capabilities.Content, Scalar("image_generation"))
		}
		if modalities == nil {
			setNode(model, "modalities", flowMapping("input", stringSequence(OutputKindText), "output", stringSequence(OutputKindImage)))
			return nil
		}
		output := ensureSequence(modalities, "output")
		if !slices.Contains(stringItems(output), OutputKindImage) {
			output.Content = append(output.Content, Scalar(OutputKindImage))
		}
	case "omni":
		if modalities != nil {
			return nil
		}
		output := []string{OutputKindText}
		if slices.Contains(stringItems(mappingValue(model, "capabilities")), "image_generation") {
			output = append(output, OutputKindImage)
		}
		setNode(model, "modalities", flowMapping("input", stringSequence(OutputKindText, OutputKindImage), "output", stringSequence(output...)))
		m.Note(strings.TrimSuffix(at.Path, ".modality"), "modality omni is written as modalities: the model reads text and images and answers in "+strings.Join(output, " and ")+"; correct them to the media it reads and writes, and declare image_generation in capabilities if it generates images")
	}
	return nil
}

// decisionRecipe is the recipe that holds the decision at.
func decisionRecipe(at FormMatch) *yaml.Node {
	for _, recipe := range sequenceAt(at.Root, "recipes") {
		recipe = resolveAlias(recipe)
		if resolveAlias(mappingValue(recipe, "decisions")) == at.Parent {
			return recipe
		}
	}
	return nil
}

// modalityRuleLabels are the labels each modality rule of a recipe matches;
// a rule still named for its trained option matches that option.
func modalityRuleLabels(recipe *yaml.Node) map[string][]string {
	labels := map[string][]string{}
	for _, rule := range sequenceAt(recipe, "signals", SignalTypeModality) {
		name := scalarAt(rule, "name")
		labels[name] = stringItems(mappingValue(rule, "labels"))
		if legacy, ok := legacyModalityLabels[name]; ok && len(labels[name]) == 0 {
			labels[name] = []string{legacy}
		}
	}
	return labels
}

// positiveModalityLabels collects the modality labels the conditions of a
// rule tree read outside a NOT: a condition's label, else its rule's labels.
func positiveModalityLabels(node *yaml.Node, rules map[string][]string, negated bool, read *[]string) {
	node = resolveAlias(node)
	if node == nil {
		return
	}
	if strings.EqualFold(scalarAt(node, "operator"), "NOT") {
		negated = !negated
	}
	for _, child := range sequenceAt(node, "conditions") {
		positiveModalityLabels(child, rules, negated, read)
	}
	if negated || scalarAt(node, "type") != SignalTypeModality {
		return
	}
	if label := scalarAt(node, "label"); label != "" {
		*read = append(*read, label)
		return
	}
	*read = append(*read, rules[scalarAt(node, "name")]...)
}

// providerModelsByName are a document's providers.models[] entries by name.
func providerModelsByName(root *yaml.Node) map[string]*yaml.Node {
	models := map[string]*yaml.Node{}
	for _, model := range sequenceAt(root, "providers", "models") {
		model = resolveAlias(model)
		models[scalarAt(model, "name")] = model
	}
	return models
}
