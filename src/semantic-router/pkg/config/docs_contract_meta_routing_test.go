package config

func metaRoutingTutorialOverviewDoc() docNeedles {
	return docNeedles{
		path: repoRel("website", "docs", "tutorials", "meta-routing", "overview.md"),
		needles: []string{
			"`routing.meta`",
			"`observe`",
			"`shadow`",
			"`active`",
			"[Design](./design)",
			"[Usage](./usage)",
		},
	}
}

func metaRoutingTutorialSidebarEntries() []string {
	return []string{
		"'tutorials/meta-routing/overview'",
		"'tutorials/meta-routing/design'",
		"'tutorials/meta-routing/problems'",
		"'tutorials/meta-routing/usage'",
	}
}
