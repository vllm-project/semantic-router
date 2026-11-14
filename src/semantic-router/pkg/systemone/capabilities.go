package systemone

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"path"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/principal"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/systemonecodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/upstream"
)

// Where a pool member's capabilities come from.
const (
	// CapabilitySourceCard is the card of the deployment the member names,
	// as its runtime or its remote API lists it.
	CapabilitySourceCard = "card"
	// CapabilitySourceListing is the entry its backend's listing gives.
	CapabilitySourceListing = "listing"
	// CapabilitySourceDeclared is what its deployment declares in
	// capabilities, or the built-in catalog lists for the model it serves.
	CapabilitySourceDeclared = "declared"
)

// CapabilityFullInput is the need of a question that requires its whole
// input read: answers that prove input_coverage complete.
const CapabilityFullInput = "full_input"

// standardTypes are what a card that lists no question types answers.
var standardTypes = []string{config.SystemOneQuestionChoice, config.SystemOneQuestionNoul, config.SystemOneQuestionScore}

// Requirements are what a native request needs of the model that answers it:
// its question types, the modalities of the media it carries beyond text, and
// whether a question requires its whole input read.
type Requirements struct {
	QuestionTypes []string
	Modalities    []string
	FullInput     bool
}

// Capabilities are what a pool member answers: its question types, the input
// modalities it reads, whether its answers prove full input coverage, and
// where they came from.
type Capabilities struct {
	Source        string
	QuestionTypes []string
	Modalities    []string
	FullInput     bool
}

// beyondStandard reports whether r needs more than every System One model
// answers: a question type besides choice, noul and score, media, or full
// input.
func (r Requirements) beyondStandard() bool {
	if r.FullInput || len(r.Modalities) > 0 {
		return true
	}
	for _, questionType := range r.QuestionTypes {
		if !slices.Contains(standardTypes, questionType) {
			return true
		}
	}
	return false
}

// Missing are the needs of r that c lacks: question types, then modalities,
// then full input.
func (c Capabilities) Missing(r Requirements) []string {
	var missing []string
	for _, questionType := range r.QuestionTypes {
		if !slices.Contains(c.QuestionTypes, questionType) {
			missing = append(missing, questionType)
		}
	}
	for _, modality := range r.Modalities {
		if !slices.Contains(c.Modalities, modality) {
			missing = append(missing, modality)
		}
	}
	if r.FullInput && !c.FullInput {
		missing = append(missing, CapabilityFullInput)
	}
	return missing
}

// cardCapabilities are what a card says its model answers: the System One
// types when it lists none, and text when it lists no modalities.
func cardCapabilities(card modelservice.ModelCard, source string) Capabilities {
	questionTypes := slices.Clone(card.QuestionTypes)
	if len(questionTypes) == 0 {
		questionTypes = slices.Clone(standardTypes)
	}
	if len(card.Surfaces) > 0 && !card.Serves("decisions") {
		questionTypes = nil
	}
	return Capabilities{
		Source: source, QuestionTypes: questionTypes, Modalities: card.ReadModalities(),
		FullInput: card.ReportsInputCoverage(),
	}
}

// MemberCapabilities tells what a pool member, a System One provider alias,
// answers. ok is false when nothing tells; such a member is asked as before.
type MemberCapabilities func(ctx context.Context, model string) (Capabilities, bool)

type membersKey struct{}

// withMembers carries a generation's member capabilities to the request's
// routing.
func withMembers(ctx context.Context, members MemberCapabilities) context.Context {
	if members == nil {
		return ctx
	}
	return context.WithValue(ctx, membersKey{}, members)
}

func membersFrom(ctx context.Context) MemberCapabilities {
	members, _ := ctx.Value(membersKey{}).(MemberCapabilities)
	return members
}

// Cards are what a generation's native model clients know of the models they
// reach.
type Cards interface {
	// Card is a deployment's current card, without waiting or network I/O.
	Card(deployment string) (modelservice.ModelCard, bool)
	// Listings are the listings of the generation's systemone backends.
	Listings() *BackendListings
}

// poolMembers finds what each System One provider alias of a generation
// answers: the card of the deployment it names (as the deployment lists it),
// else what that deployment declares; for an alias with backend_refs, the
// entry its backend lists for the model, read once in the background.
func poolMembers(cfg *config.RouterConfig, listener string, cards Cards, remote *upstream.Set) MemberCapabilities {
	if cfg == nil || cards == nil {
		return nil
	}
	return func(ctx context.Context, alias string) (Capabilities, bool) {
		model, ok := cfg.ModelConfig[alias]
		if !ok || model.APIFormat != config.APIFormatSystemOne {
			return Capabilities{}, false
		}
		if model.Deployment != "" {
			return deploymentCapabilities(cfg, cards, model.Deployment)
		}
		if remote == nil {
			return Capabilities{}, false
		}
		return cards.Listings().capabilities(ctx, alias, func(ctx context.Context) (Capabilities, error) {
			return readBackendListing(ctx, cfg, listener, remote, alias)
		})
	}
}

// deploymentCapabilities are a deployment's card, else what its configuration
// tells: a systemone deployment's declared capabilities (or its schema's
// standard types), or the built-in catalog's entry for the model a
// model_runtime deployment serves.
func deploymentCapabilities(cfg *config.RouterConfig, cards Cards, name string) (Capabilities, bool) {
	declaredModalities := cfg.DeploymentModalities(name)
	if card, ok := cards.Card(name); ok {
		capabilities := cardCapabilities(card, CapabilitySourceCard)
		if len(card.Modalities) == 0 && len(declaredModalities) > 0 {
			capabilities.Modalities = declaredModalities
		}
		return capabilities, true
	}
	questionTypes := cfg.DeploymentQuestionTypes(name)
	if questionTypes == nil {
		return Capabilities{}, false
	}
	deployment := cfg.ModelDeployments[name]
	card := modelservice.RemoteCard(name, questionTypes, nil)
	card.Modalities = declaredModalities
	if deployment.IsSystemOne() {
		card.Schema = deployment.EffectiveSchema()
	}
	return cardCapabilities(card, CapabilitySourceDeclared), true
}

// listingTimeout bounds one read of a backend's listing, and listingWait how
// long a request waits for the first one; a read that takes longer still
// answers later requests.
const (
	listingTimeout = 5 * time.Second
	listingWait    = time.Second
)

// listingLimit bounds a listing body.
const listingLimit = 4 << 20

// BackendListings are what a generation's systemone backends list: each
// alias's entry, read once, the first time a request needs it. When the read
// fails, or has not answered within listingWait, nothing tells.
type BackendListings struct {
	entries sync.Map // alias -> *backendListing
}

type backendListing struct {
	once   sync.Once
	done   chan struct{}
	listed atomic.Pointer[Capabilities]
}

func (b *BackendListings) capabilities(ctx context.Context, alias string, read func(context.Context) (Capabilities, error)) (Capabilities, bool) {
	if b == nil {
		return Capabilities{}, false
	}
	value, _ := b.entries.LoadOrStore(alias, &backendListing{done: make(chan struct{})})
	entry := value.(*backendListing)
	entry.once.Do(func() {
		go func() {
			defer close(entry.done)
			readCtx, cancel := context.WithTimeout(context.Background(), listingTimeout)
			defer cancel()
			if capabilities, err := read(readCtx); err == nil {
				entry.listed.Store(&capabilities)
			}
		}()
	})
	wait := time.NewTimer(listingWait)
	defer wait.Stop()
	select {
	case <-entry.done:
	case <-wait.C:
	case <-ctx.Done():
	}
	if listed := entry.listed.Load(); listed != nil {
		return *listed, true
	}
	return Capabilities{}, false
}

// readBackendListing reads what a systemone backend lists for the model an
// alias asks it for: GET beside its endpoint, as the backend's schema lists,
// and the entry's card, else its features. A backend that takes a client's
// delegated credential lists for no one.
func readBackendListing(ctx context.Context, cfg *config.RouterConfig, listener string, remote *upstream.Set, alias string) (Capabilities, error) {
	_, endpointName, found, err := cfg.ResolvePrimaryBackendForModel(alias)
	if err != nil || !found {
		return Capabilities{}, errors.New("inference backend binding unavailable")
	}
	profile, err := cfg.GetProviderProfileForEndpoint(endpointName)
	if err != nil {
		return Capabilities{}, err
	}
	if profile.Delegated() {
		return Capabilities{}, principal.ErrRouterCall
	}
	create, err := profile.ResolveCreatePath("vllm-sr/systemone@1")
	if err != nil {
		return Capabilities{}, err
	}
	schema := systemonecodec.SchemaID(cfg.UpstreamSchema(alias))
	codec, _, err := systemonecodec.Builtin().Resolve(schema, nil)
	if err != nil {
		return Capabilities{}, err
	}
	header := http.Header{}
	header.Set(headers.VSRSystemOneBackend, "1")
	endpoint, _ := cfg.GetEndpointByName(endpointName)
	if err = setBackendKey(header, profile, endpoint.APIKey); err != nil {
		return Capabilities{}, err
	}
	response, err := remote.Do(ctx, &upstream.Request{
		Method: http.MethodGet, Path: listingPath(create, codec.ListingPath()), Header: header,
		RouteKey: alias, ExactRoute: true, Listener: listener,
	})
	if err != nil {
		return Capabilities{}, err
	}
	defer response.Body.Close()
	data, err := io.ReadAll(io.LimitReader(response.Body, listingLimit+1))
	if err != nil || len(data) > listingLimit {
		return Capabilities{}, errors.New("invalid listing size")
	}
	listing, err := codec.DecodeListing(response.StatusCode, data)
	if err != nil {
		return Capabilities{}, err
	}
	external := cfg.ResolveExternalModelID(alias, endpointName)
	for _, entry := range listing.Models {
		if entry.Name != external {
			continue
		}
		if raw, ok := systemonecodec.ListedCardJSON(entry); ok {
			var card api.ModelCard
			if json.Unmarshal(raw, &card) == nil {
				decoded := modelservice.ModelCardFromAPI(card)
				decoded.Schema = string(schema)
				return cardCapabilities(decoded, CapabilitySourceListing), nil
			}
		}
		for _, member := range entry.Extensions {
			if member.Key != "features" {
				continue
			}
			card := modelservice.RemoteCard(external, nil, nil)
			card.Schema = string(schema)
			for _, item := range member.Value.Items() {
				if feature, ok := item.Text(); ok {
					card.QuestionTypes = append(card.QuestionTypes, feature)
				}
			}
			return cardCapabilities(card, CapabilitySourceListing), nil
		}
		return Capabilities{}, fmt.Errorf("the listing gives no card or features for %q", external)
	}
	return Capabilities{}, fmt.Errorf("the listing has no %q", external)
}

// listingPath is where a backend whose systemone endpoint is create lists its
// models: listing appended to it ("/models"), or replacing its last segment
// ("models": POST /v1/systemone lists at GET /v1/models).
func listingPath(create, listing string) string {
	if strings.HasPrefix(listing, "/") {
		return strings.TrimSuffix(create, "/") + listing
	}
	return path.Join(path.Dir(strings.TrimSuffix(create, "/")), listing)
}

// setBackendKey puts a backend's key in the header its provider profile names.
func setBackendKey(header http.Header, profile *config.ProviderProfile, key string) error {
	if key == "" {
		return nil
	}
	name, prefix, err := profile.ResolveAuthHeader()
	if err != nil {
		return err
	}
	if prefix != "" {
		key = prefix + " " + key
	}
	header.Set(name, key)
	return nil
}

// RequestError is a request auto cannot route as it is written, such as one
// with a question type auto does not answer; its message says why.
type RequestError struct{ Err error }

func (e *RequestError) Error() string { return e.Err.Error() }

func (e *RequestError) Unwrap() error { return e.Err }

// IncapableError means no candidate model of a decision can answer a
// request: what the candidates lack of what it needs, and each candidate with
// what it lacks and where its capabilities came from.
type IncapableError struct {
	Decision string
	Missing  []string
	Members  []IncapableMember
}

// IncapableMember is one candidate that cannot answer a request.
type IncapableMember struct {
	Model   string
	Source  string
	Missing []string
}

func (e *IncapableError) Error() string {
	members := make([]string, 0, len(e.Members))
	for _, member := range e.Members {
		members = append(members, fmt.Sprintf("%s lacks %s (%s)", member.Model, strings.Join(member.Missing, ", "), member.Source))
	}
	subject := "no candidate model"
	if e.Decision != "" {
		subject = fmt.Sprintf("no candidate model of decision %q", e.Decision)
	}
	return fmt.Sprintf("%s answers this request, which needs %s: %s", subject, strings.Join(e.Missing, ", "), strings.Join(members, "; "))
}
