package handlers

import (
	"net/http"
	"slices"
	"strings"

	"gopkg.in/yaml.v3"
	sigsyaml "sigs.k8s.io/yaml"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const keyReferenceDestinationRefused = "Forbidden: an environment reference (${NAME} or $NAME) or a saved key's file or variable (auth.api_key_file, auth.api_key_env) can stay only where the saved config uses it, on the same destination; enter the value itself to send it elsewhere"

// rejectKeyReferenceRedirect refuses a save, by a caller who may not read
// configuration secrets, that sends a reference somewhere the stored config
// does not. The Router expands every variable of its environment across the
// whole document, and reads a saved key from the file or key Secret variable an
// auth block names, while the editor shows the reference, not the value. So a
// writer could otherwise send a value they never saw, a provider key such as
// ${OPENAI_API_KEY} or any variable like $HOME, to a host of theirs, in a
// backend, a plugin prompt, a path, a header or a store password. A value the
// caller types is theirs to send anywhere.
func rejectKeyReferenceRedirect(w http.ResponseWriter, r *http.Request, existing, updated []byte) bool {
	if callerCanViewConfigSecrets(r) || keyReferencesKept(existing, updated) {
		return false
	}
	http.Error(w, keyReferenceDestinationRefused, http.StatusForbidden)
	return true
}

// keyReferencesKept reports whether every reference updated names is kept
// (keyReferenceCheck.kept): every environment variable the Router expands in
// a string value, and the file or key Secret variable an auth block names. A
// document that cannot be read keeps none.
func keyReferencesKept(existing, updated []byte) bool {
	stored, err := readKeyReferenceDocument(existing)
	if err != nil {
		return false
	}
	saved, err := readKeyReferenceDocument(updated)
	if err != nil {
		return false
	}
	check := keyReferenceCheck{stored: stored, saved: saved, unchanged: map[*yaml.Node]bool{}, reached: map[[3]*yaml.Node]bool{}}
	return check.walk(saved.root, stored.root, nil, "")
}

// readKeyReferenceDocument reads a document as the Router reads it: older
// forms in their canonical form, aliases and merge keys resolved.
func readKeyReferenceDocument(data []byte) (*keyReferenceScan, error) {
	if canonical, err := routerconfig.CanonicalConfigBytes(data); err == nil {
		data = canonical
	}
	resolved, err := sigsyaml.YAMLToJSON(data)
	if err != nil {
		return nil, err
	}
	doc, err := parseYAMLDocument(resolved)
	if err != nil {
		return nil, err
	}
	return &keyReferenceScan{root: doc.Content[0], reaches: map[[3]*yaml.Node][]*yaml.Node{}}, nil
}

// keyReferenceCheck walks the saved document beside the stored one, each node
// with its stored counterpart: the same field, or the sequence item with the
// same name or id (matchingConfigSequenceItem), else at the same index.
type keyReferenceCheck struct {
	stored, saved *keyReferenceScan
	unchanged     map[*yaml.Node]bool
	reached       map[[3]*yaml.Node]bool
	connections   *bool
}

// keyReferencePair is a saved mapping that encloses a value, its stored
// counterpart (nil when it has none), and its path as the editor's redaction
// reads it.
type keyReferencePair struct {
	saved, stored *yaml.Node
	path          string
}

func (c *keyReferenceCheck) walk(saved, stored *yaml.Node, chain []keyReferencePair, path string) bool {
	if stored != nil && stored.Kind != saved.Kind {
		stored = nil
	}
	switch saved.Kind {
	case yaml.MappingNode:
		return c.walkMapping(saved, stored, append(chain[:len(chain):len(chain)], keyReferencePair{saved, stored, path}), path)
	case yaml.ScalarNode:
		return c.walkScalar(saved, stored, chain)
	}
	for i, child := range saved.Content {
		if !c.walk(child, matchingConfigSequenceItem(stored, child, i), chain, path) {
			return false
		}
	}
	return true
}

// walkScalar checks each environment reference a value names: the stored
// value at the same place names it too, and it is kept.
func (c *keyReferenceCheck) walkScalar(saved, stored *yaml.Node, chain []keyReferencePair) bool {
	for _, name := range routerconfig.EnvironmentReferenceNames(saved.Value) {
		if stored == nil || !slices.Contains(routerconfig.EnvironmentReferenceNames(stored.Value), name) || !c.kept(chain) {
			return false
		}
	}
	return true
}

func (c *keyReferenceCheck) walkMapping(saved, stored *yaml.Node, chain []keyReferencePair, path string) bool {
	for i := 0; i+1 < len(saved.Content); i += 2 {
		key := saved.Content[i].Value
		if !c.walkField(key, saved.Content[i+1], mappingValueNode(stored, key), chain, path+"."+key) {
			return false
		}
	}
	return true
}

// walkField checks a mapping's field: a key file or variable an auth block
// names must be the stored one and kept. A listener's client keys (api_keys,
// api_key_files) are not followed, as they check what clients send, not a
// value the Router sends on.
func (c *keyReferenceCheck) walkField(key string, value, previous *yaml.Node, chain []keyReferencePair, field string) bool {
	switch {
	case key == "api_key_file" || key == "api_key_env":
		return value.Value == "" || (previous != nil && previous.Value == value.Value && c.kept(chain))
	case field == ".listeners.api_keys" || field == ".listeners.api_key_files":
		return true
	}
	return c.walk(value, previous, chain, field)
}

// kept reports whether a reference held inside chain keeps its
// destinations. The same reference is named at the same place of the stored
// config, and every enclosing mapping has a stored counterpart that sends
// the same way (enclosingFieldsChanged). Where its value rides on the
// requests sent to models (requestScope), every backend they can reach was
// reachable (sameReach), and outside providers.models the global
// connections and its recipe's model_bindings are unchanged
// (sameConnections).
func (c *keyReferenceCheck) kept(chain []keyReferencePair) bool {
	for _, pair := range chain {
		same, ok := c.unchanged[pair.saved]
		if !ok {
			same = pair.stored != nil && !enclosingFieldsChanged(pair)
			c.unchanged[pair.saved] = same
		}
		if !same {
			return false
		}
	}
	scope, rides := requestScope(chain)
	return !rides || (c.sameReach(scope) && (scope[0].saved != nil || c.sameConnections(scope[1])))
}

// modelDestinationFields are the fields of a model that choose where its
// backends' requests go, beside its backend_refs: a named deployment, the API
// and schema a request is sent as, the catalog card, and a provider profile
// of the older layout.
var modelDestinationFields = []string{"deployment", "api_format", "schema", "catalog", "provider_profile"}

// enclosingFieldsChanged reports whether an enclosing mapping changed where
// it sends what it holds. A model and providers compare their destination
// fields and modelDestinationFields, so a model's pricing, limits and
// descriptions stay editable beside a key, and every other mapping compares
// all its plain fields (directFieldsChanged).
func enclosingFieldsChanged(pair keyReferencePair) bool {
	if pair.path != ".providers" && pair.path != ".providers.models" {
		return directFieldsChanged(pair.stored, pair.saved)
	}
	return configSecretTargetChanged(pair.stored, pair.saved) || slices.ContainsFunc(modelDestinationFields, func(key string) bool {
		return !sameOptionalNode(mappingValueNode(pair.stored, key), mappingValueNode(pair.saved, key))
	})
}

// directFieldsChanged reports whether two mappings differ in a destination
// field, or in any other field whose value is a scalar naming no reference:
// a selector such as backend, store_backend, type or deployment, and a plain
// setting alike.
func directFieldsChanged(stored, saved *yaml.Node) bool {
	if configSecretTargetChanged(stored, saved) {
		return true
	}
	for _, mapping := range []*yaml.Node{stored, saved} {
		for i := 0; i+1 < len(mapping.Content); i += 2 {
			key := mapping.Content[i].Value
			before, after := mappingValueNode(stored, key), mappingValueNode(saved, key)
			if !isConfigSecretTargetField(key) && (plainScalar(before) || plainScalar(after)) && !sameOptionalNode(before, after) {
				return true
			}
		}
	}
	return false
}

func plainScalar(node *yaml.Node) bool {
	return node != nil && node.Kind == yaml.ScalarNode && len(routerconfig.EnvironmentReferenceNames(node.Value)) == 0
}

func sameOptionalNode(a, b *yaml.Node) bool {
	if a == nil || b == nil {
		return a == b
	}
	return sameConfigNode(a, b)
}

// sameReach reports whether every backend the requests of scope (a model,
// recipe and decision, from requestScope) can reach is one they reached.
func (c *keyReferenceCheck) sameReach(scope [3]keyReferencePair) bool {
	key := [3]*yaml.Node{scope[0].saved, scope[1].saved, scope[2].saved}
	if same, ok := c.reached[key]; ok {
		return same
	}
	stored := c.stored.reach(scope[0].stored, scope[1].stored, scope[2].stored)
	same := !slices.ContainsFunc(c.saved.reach(key[0], key[1], key[2]), func(destination *yaml.Node) bool {
		return !slices.ContainsFunc(stored, func(previous *yaml.Node) bool {
			return !configSecretTargetsChanged(previous, destination, "")
		})
	})
	c.reached[key] = same
	return same
}

// requestScope is the model, recipe and decision whose requests carry a
// value inside chain. A value rides on the requests sent to models unless it
// sits in a backend, which the enclosing mappings already compare, or in a
// block that names its own destination outside a model, a recipe,
// global.plugins and global.integrations.
func requestScope(chain []keyReferencePair) (scope [3]keyReferencePair, rides bool) {
	ownDestination := false
	for _, pair := range chain {
		switch pair.path {
		case ".providers.models.backend_refs":
			return scope, false
		case ".providers.models":
			scope[0], rides = pair, true
		case ".recipes":
			scope[1], rides = pair, true
		case ".recipes.decisions":
			scope[2] = pair
		case ".global.plugins", ".global.integrations":
			rides = true
		}
		ownDestination = ownDestination || namesDestination(pair.saved)
	}
	return scope, rides || !ownDestination
}

// sameConnections reports whether a value that rides outside a connection
// block, in a recipe, a plugin or global.integrations, still meets the same
// global connections (sameGlobalConnections) and its recipe's
// model_bindings, which choose the deployment a judge, a signal or a store
// asks. A value inside a store, a service or a deployment goes to that
// block, which its enclosing mappings compare.
func (c *keyReferenceCheck) sameConnections(recipe keyReferencePair) bool {
	if recipe.saved != nil && !sameOptionalNode(mappingValueNode(recipe.stored, "model_bindings"), mappingValueNode(recipe.saved, "model_bindings")) {
		return false
	}
	if c.connections == nil {
		same := sameGlobalConnections(c.stored.root, c.saved.root)
		c.connections = &same
	}
	return *c.connections
}

// sameGlobalConnections reports whether the effective global blocks that say
// where the Router connects and which deployment answers are unchanged: the
// stores, the services, the integrations, and the model catalog's
// deployments, embeddings, bindings and system models. A RAG source that
// reuses the cache connection, a vector store search, a judge or a binding
// sends what it reads through them.
func sameGlobalConnections(stored, saved *yaml.Node) bool {
	before, err := globalConnections(stored)
	if err != nil {
		return false
	}
	after, err := globalConnections(saved)
	return err == nil && slices.EqualFunc(before, after, sameOptionalNode)
}

func globalConnections(root *yaml.Node) ([]*yaml.Node, error) {
	comparison, err := configSecretComparisonDocument(root)
	if err != nil {
		return nil, err
	}
	global := mappingValueNode(comparison, "global")
	catalog := mappingValueNode(global, "model_catalog")
	return []*yaml.Node{
		mappingValueNode(global, "stores"), mappingValueNode(global, "services"), mappingValueNode(global, "integrations"),
		mappingValueNode(catalog, "deployments"), mappingValueNode(catalog, "embeddings"),
		mappingValueNode(catalog, "bindings"), mappingValueNode(catalog, "system"),
	}, nil
}

type keyReferenceScan struct {
	root    *yaml.Node
	reaches map[[3]*yaml.Node][]*yaml.Node
}

// reach is every backend a request can be sent to from a model, or from a
// decision or recipe (reachableModels), or from anywhere else. A config
// without a model list fails closed: every block that names a destination
// counts.
func (s *keyReferenceScan) reach(model, recipe, decision *yaml.Node) []*yaml.Node {
	key := [3]*yaml.Node{model, recipe, decision}
	if destinations, ok := s.reaches[key]; ok {
		return destinations
	}
	var destinations []*yaml.Node
	models := mappingValueNode(mappingValueNode(s.root, "providers"), "models")
	switch {
	case model != nil:
		destinations = s.modelDestinations(model)
	case models == nil || models.Kind != yaml.SequenceNode:
		destinations = destinationBlocks(s.root)
	default:
		for _, reached := range s.reachableModels(models, recipe, decision) {
			destinations = append(destinations, s.modelDestinations(reached)...)
		}
	}
	s.reaches[key] = destinations
	return destinations
}

// namesDestination reports whether mapping holds a field that says where it
// sends what it holds (an endpoint, address, URL, host, port, provider or
// protocol), not only a name, id, type or key source.
func namesDestination(mapping *yaml.Node) bool {
	for i := 0; i+1 < len(mapping.Content); i += 2 {
		switch key := mapping.Content[i].Value; key {
		case "name", "id", "type", "api_key_env", "api_key_file":
		default:
			if isConfigSecretTargetField(key) {
				return true
			}
		}
	}
	return false
}

// reachableModels are the models a request of decision, or of recipe when
// decision is nil, can be sent to: each model named by any value outside the
// model list, the other recipes and the recipe's other decisions (modelNamed),
// which covers modelRefs, fallbacks, Looper and escalation candidates, a
// route action, a shadow model and the default model. Outside a recipe, such
// as in global.plugins, which also shape a request that names a model, and
// when learning may pick beyond the decision (candidate_set tier or global),
// every model counts.
func (s *keyReferenceScan) reachableModels(models, recipe, decision *yaml.Node) []*yaml.Node {
	if recipe == nil || learningLeavesDecision(s.root) {
		return models.Content
	}
	skip := map[*yaml.Node]bool{models: true}
	for _, other := range sequenceItems(mappingValueNode(s.root, "recipes")) {
		skip[other] = other != recipe
	}
	for _, other := range sequenceItems(mappingValueNode(recipe, "decisions")) {
		skip[other] = decision != nil && other != decision
	}
	named := map[string]bool{}
	addScalars(named, s.root, skip)
	return slices.DeleteFunc(slices.Clone(models.Content), func(model *yaml.Node) bool {
		return !modelNamed(model, named)
	})
}

// modelNamed reports whether named holds a name that selects model: its
// name, an external model ID or one of its LoRA adapters.
func modelNamed(model *yaml.Node, named map[string]bool) bool {
	names := map[string]bool{}
	addScalars(names, mappingValueNode(model, "name"), nil)
	addScalars(names, mappingValueNode(model, "external_model_ids"), nil)
	for _, adapter := range sequenceItems(mappingValueNode(model, "loras")) {
		addScalars(names, mappingValueNode(adapter, "name"), nil)
	}
	for name := range names {
		if named[name] {
			return true
		}
	}
	return false
}

// modelDestinations are where a model's requests go: its backends and the
// deployment it names, or the model itself when it names neither or its
// deployment is not in the document.
func (s *keyReferenceScan) modelDestinations(model *yaml.Node) []*yaml.Node {
	destinations := slices.Clone(sequenceItems(mappingValueNode(model, "backend_refs")))
	if name := mappingValueNode(model, "deployment"); name != nil && name.Value != "" {
		deployments := mappingValueNode(mappingValueNode(mappingValueNode(s.root, "global"), "model_catalog"), "deployments")
		deployment := mappingValueNode(deployments, name.Value)
		if deployment == nil {
			deployment = model
		}
		destinations = append(destinations, deployment)
	}
	if len(destinations) == 0 {
		return []*yaml.Node{model}
	}
	return destinations
}

// learningLeavesDecision reports whether learning may send a decision's
// request to a model of another decision or to any model.
func learningLeavesDecision(node *yaml.Node) bool {
	if set := mappingValueNode(node, "candidate_set"); set != nil && set.Value != "" && set.Value != routerconfig.RouterLearningCandidateSetDecision {
		return true
	}
	return slices.ContainsFunc(node.Content, learningLeavesDecision)
}

func destinationBlocks(node *yaml.Node) []*yaml.Node {
	var blocks []*yaml.Node
	if node.Kind == yaml.MappingNode && namesDestination(node) {
		blocks = append(blocks, node)
	}
	for _, child := range node.Content {
		blocks = append(blocks, destinationBlocks(child)...)
	}
	return blocks
}

// addScalars adds every scalar under node, keys included, except under skip.
func addScalars(into map[string]bool, node *yaml.Node, skip map[*yaml.Node]bool) {
	if node == nil || skip[node] {
		return
	}
	if value := strings.TrimSpace(node.Value); node.Kind == yaml.ScalarNode && value != "" {
		into[value] = true
	}
	for _, child := range node.Content {
		addScalars(into, child, skip)
	}
}
