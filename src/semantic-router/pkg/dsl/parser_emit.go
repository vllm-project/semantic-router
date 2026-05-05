package dsl

// ---------- EMIT block conversion ----------

// rawToEmitDecl converts a parsed EMIT block into the resolved AST form.
// Field-level coercion of the retention payload is centralized here so the
// validator can enforce sentinel rules on a typed structure.
func rawToEmitDecl(r *rawEmitDecl) *EmitDecl {
	if r == nil {
		return nil
	}
	decl := &EmitDecl{
		Kind:      r.Kind,
		RawFields: r.Fields,
		Pos:       posFromLexer(r.Pos),
	}
	if r.Kind == "retention" {
		decl.Retention = rawToRetention(r.Fields)
	}
	return decl
}

func rawToRetention(entries []*FieldEntry) *RetentionDirective {
	r := &RetentionDirective{}
	for _, e := range entries {
		if e == nil || e.Value == nil {
			continue
		}
		switch e.Key {
		case "drop":
			if e.Value.Bool != nil {
				b := *e.Value.Bool == "true"
				r.Drop = &b
			}
		case "ttl_turns":
			if e.Value.Int != nil {
				n := *e.Value.Int
				r.TTLTurns = &n
			}
		case "keep_current_model":
			if e.Value.Bool != nil {
				b := *e.Value.Bool == "true"
				r.KeepCurrentModel = &b
			}
		case "prefer_prefix_retention":
			if e.Value.Bool != nil {
				b := *e.Value.Bool == "true"
				r.PreferPrefixRetention = &b
			}
		}
	}
	return r
}
