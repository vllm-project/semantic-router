package apicontract

// Operation is the optional schema half of a route policy. A route that
// declares one is rendered with concrete request and response bodies; a route
// without one is still rendered with its policy and marked undocumented.
type Operation struct {
	// ID overrides the operation id derived from the method and path.
	ID          string
	Summary     string
	Description string
	Request     *RequestBody
	Responses   map[int]Response
}

// JSONRequest describes a required JSON body decoded into T.
func JSONRequest[T any](description string) *RequestBody {
	schema := SchemaFor[T]()
	return &RequestBody{
		Description: description,
		Required:    true,
		Content:     map[string]Media{"application/json": {Schema: &schema}},
	}
}

// JSONResponse describes a JSON body encoded from T.
func JSONResponse[T any](description string) Response {
	schema := SchemaFor[T]()
	return Response{
		Description: description,
		Content:     map[string]Media{"application/json": {Schema: &schema}},
	}
}

// JSONOneOfResponse describes a JSON body that matches exactly one of schemas.
func JSONOneOfResponse(description string, schemas ...Schema) Response {
	return Response{
		Description: description,
		Content:     map[string]Media{"application/json": {Schema: &Schema{OneOf: schemas}}},
	}
}

// EitherResponse describes a status whose body format depends on which path
// failed; clients choose by Content-Type. Each alternative must use a
// different media type.
func EitherResponse(description string, alternatives ...Response) Response {
	merged := Response{Description: description, Content: map[string]Media{}}
	for _, alternative := range alternatives {
		for mediaType, media := range alternative.Content {
			if _, exists := merged.Content[mediaType]; exists {
				panic("response alternatives repeat media type " + mediaType)
			}
			merged.Content[mediaType] = media
		}
	}
	return merged
}

// TextResponse describes a plain-text body, which is what http.Error writes.
func TextResponse(description string) Response {
	return Response{
		Description: description,
		Content:     map[string]Media{"text/plain": {Schema: &Schema{Type: "string"}}},
	}
}
