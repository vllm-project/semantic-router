package modelservice

// Card is the card name's questions compile against: the question types and
// presets its API lists for the model in GET <endpoint>/models, else the
// declared ones, else the System One types; and the modalities it lists, else
// the declared ones, else text. The first call reads the listing in the
// background unless a preparation has read it; until it answers, the declared
// or System One types apply.
func (r *RemoteSystemOne) Card(name string) (ModelCard, bool) {
	d := r.deployment(name)
	if d == nil {
		return ModelCard{}, false
	}
	if card := d.listed.Load(); card != nil {
		return *card, true
	}
	if d.started.CompareAndSwap(false, true) {
		go d.readListing(r.ctx, listingTimeout)
	}
	return d.declaredCard(), true
}

// ListedCard is the card name's API lists for its model, when the listing has
// answered; like Card, the first call reads the listing in the background.
func (r *RemoteSystemOne) ListedCard(name string) (ModelCard, bool) {
	d := r.deployment(name)
	if d == nil {
		return ModelCard{}, false
	}
	if card := d.listed.Load(); card != nil {
		return *card, true
	}
	if d.started.CompareAndSwap(false, true) {
		go d.readListing(r.ctx, listingTimeout)
	}
	return ModelCard{}, false
}
