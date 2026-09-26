package extproc

import (
	"errors"
	"fmt"
	"net/url"
	"strconv"
	"strings"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/shadowdataset"
)

const (
	routerReplayDatasetPath = routerReplayAPIBasePath + "/dataset"
	// A manifest describes the whole selection it was built from, and its digest
	// is that selection's identity. Exporting a page of a larger match would
	// publish a digest for a dataset nobody can rebuild, so an oversized
	// selection is refused rather than silently cut.
	routerReplayDatasetMaxRecords = 5000
)

var errRouterReplayDatasetTooLarge = fmt.Errorf(
	"replay selection exceeds %d records; narrow the filters and export again",
	routerReplayDatasetMaxRecords,
)

// handleRouterReplayDatasetAPI serves
// GET /api/v1/observability/replays/dataset?seed={seed}&split={name:weight}.
// It turns the selected observations into a shadow comparison manifest. The
// manifest holds identity and digests only, never prompt or response text, so
// it carries no payload a replay reader could not already list.
func (r *OpenAIRouter) handleRouterReplayDatasetAPI(
	method string,
	rawQuery string,
) *ext_proc.ProcessingResponse {
	_, manifest, _, failure := r.buildRouterReplayDataset(method, rawQuery)
	if failure != nil {
		return failure
	}
	return r.createRouterReplayJSONResponse(200, manifest)
}

// buildRouterReplayDataset reads the selection and policy from the query and
// builds the manifest over it. Every route that serves a view of the dataset
// starts here, so they all describe the same examples. A non-nil response is
// the error to return.
func (r *OpenAIRouter) buildRouterReplayDataset(
	method string,
	rawQuery string,
) (url.Values, shadowdataset.Manifest, []routerreplay.RoutingRecord, *ext_proc.ProcessingResponse) {
	if method != "GET" {
		return nil, shadowdataset.Manifest{}, nil, r.createErrorResponse(405, "method not allowed")
	}

	values, err := url.ParseQuery(rawQuery)
	if err != nil {
		return nil, shadowdataset.Manifest{}, nil, r.createErrorResponse(400, "invalid query parameters")
	}
	filters, err := parseRouterReplayFilters(values)
	if err != nil {
		return nil, shadowdataset.Manifest{}, nil, r.createErrorResponse(400, err.Error())
	}
	policy, err := parseShadowDatasetPolicy(values)
	if err != nil {
		return nil, shadowdataset.Manifest{}, nil, r.createErrorResponse(400, err.Error())
	}

	records, err := r.queryRouterReplayDatasetRecords(filters)
	if err != nil {
		if errors.Is(err, errRouterReplayDatasetTooLarge) {
			return nil, shadowdataset.Manifest{}, nil, r.createErrorResponse(400, err.Error())
		}
		return nil, shadowdataset.Manifest{}, nil, r.createErrorResponse(500, "router replay storage query failed")
	}

	// Build validates the policy, so a malformed seed or split plan is reported
	// as the caller error it is rather than checked twice.
	manifest, err := shadowdataset.Build(records, policy)
	if err != nil {
		return nil, shadowdataset.Manifest{}, nil, r.createErrorResponse(400, err.Error())
	}
	return values, manifest, records, nil
}

// parseShadowDatasetPolicy reads the export policy from the query. Splits are
// repeatable and weights are whole numbers, so a plan has no rounding to argue
// about and reads the same in a URL as it does in the manifest.
func parseShadowDatasetPolicy(values url.Values) (shadowdataset.Policy, error) {
	policy := shadowdataset.Policy{Seed: strings.TrimSpace(values.Get("seed"))}
	for _, entry := range values["split"] {
		name, weight, found := strings.Cut(entry, ":")
		if !found {
			return policy, fmt.Errorf("split %q must be written as name:weight", entry)
		}
		parsed, err := strconv.Atoi(strings.TrimSpace(weight))
		if err != nil {
			return policy, fmt.Errorf("split %q must carry an integer weight", entry)
		}
		policy.Splits = append(policy.Splits, shadowdataset.Split{
			Name:   strings.TrimSpace(name),
			Weight: parsed,
		})
	}
	return policy, nil
}

// queryRouterReplayDatasetRecords reads every record the filters select, with
// captured bodies, because an example needs the digests the router recorded on
// the observation rather than the summary a list page returns.
func (r *OpenAIRouter) queryRouterReplayDatasetRecords(
	filters routerReplayFilters,
) ([]routerreplay.RoutingRecord, error) {
	var records []routerreplay.RoutingRecord
	total := 0
	for _, reader := range r.routerReplayReaders() {
		page, supported, err := reader.QueryPage(
			filters.storageFilters(),
			routerReplayDatasetMaxRecords+1,
			0,
			true,
		)
		if err != nil {
			return nil, err
		}
		if !supported {
			all, listErr := reader.ListRecords()
			if listErr != nil {
				return nil, listErr
			}
			page.Records = filterRouterReplayRecords(all, filters)
			page.Total = len(page.Records)
		}
		records = append(records, page.Records...)
		total += page.Total
	}
	if total > routerReplayDatasetMaxRecords {
		return nil, errRouterReplayDatasetTooLarge
	}
	return records, nil
}
