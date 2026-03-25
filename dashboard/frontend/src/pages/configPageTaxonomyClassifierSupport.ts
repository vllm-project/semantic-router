export interface TaxonomySignalBinding {
  kind: 'tier' | 'category'
  value: string
}

export interface TaxonomySignalReference {
  name: string
  bind: TaxonomySignalBinding
}

export interface TaxonomyClassifierTier {
  name: string
  description?: string
}

export interface TaxonomyClassifierCategory {
  name: string
  tier: string
  description?: string
  exemplars: string[]
}

export interface TaxonomyClassifierRecord {
  name: string
  type: string
  builtin: boolean
  managed: boolean
  editable: boolean
  threshold: number
  security_threshold?: number
  description?: string
  source: {
    path: string
    taxonomy_file?: string
  }
  tiers: TaxonomyClassifierTier[]
  categories: TaxonomyClassifierCategory[]
  tier_groups?: Record<string, string[]>
  signal_references: TaxonomySignalReference[]
  bind_options: {
    tiers: string[]
    categories: string[]
  }
  load_error?: string
}

export interface TaxonomyClassifierListResponse {
  items: TaxonomyClassifierRecord[]
}

export interface TaxonomyClassifierDraft {
  name: string
  threshold: number
  security_threshold: number
  description: string
  tiers: TaxonomyClassifierTier[]
  categories: TaxonomyClassifierCategory[]
  tier_groups: Array<{
    name: string
    categories: string
  }>
}

export interface TaxonomyTierDraft {
  name: string
  description: string
}

export interface TaxonomyCategoryDraft {
  name: string
  tier: string
  description: string
  exemplars: string
}

export function emptyTaxonomyClassifierDraft(): TaxonomyClassifierDraft {
  return {
    name: '',
    threshold: 0.3,
    security_threshold: 0.25,
    description: '',
    tiers: [],
    categories: [
      {
        name: '',
        tier: '',
        description: '',
        exemplars: [''],
      },
    ],
    tier_groups: [],
  }
}

export function emptyTaxonomyTierDraft(): TaxonomyTierDraft {
  return {
    name: '',
    description: '',
  }
}

export function emptyTaxonomyCategoryDraft(defaultTier = ''): TaxonomyCategoryDraft {
  return {
    name: '',
    tier: defaultTier,
    description: '',
    exemplars: '',
  }
}

export function classifierDraftFromRecord(record: TaxonomyClassifierRecord): TaxonomyClassifierDraft {
  return {
    name: record.name,
    threshold: record.threshold,
    security_threshold: record.security_threshold ?? record.threshold,
    description: record.description ?? '',
    tiers: record.tiers.map((tier) => ({
      name: tier.name,
      description: tier.description ?? '',
    })),
    categories: record.categories.map((category) => ({
      name: category.name,
      tier: category.tier,
      description: category.description ?? '',
      exemplars: category.exemplars.length > 0 ? [...category.exemplars] : [''],
    })),
    tier_groups: Object.entries(record.tier_groups ?? {}).map(([name, categories]) => ({
      name,
      categories: categories.join(', '),
    })),
  }
}

export function tierDraftFromTier(tier: TaxonomyClassifierTier): TaxonomyTierDraft {
  return {
    name: tier.name,
    description: tier.description ?? '',
  }
}

export function categoryDraftFromCategory(category: TaxonomyClassifierCategory): TaxonomyCategoryDraft {
  return {
    name: category.name,
    tier: category.tier,
    description: category.description ?? '',
    exemplars: category.exemplars.join('\n'),
  }
}

export function payloadFromDraft(draft: TaxonomyClassifierDraft) {
  const tierGroups = draft.tier_groups.reduce<Record<string, string[]>>((acc, group) => {
    const name = group.name.trim()
    if (!name) {
      return acc
    }

    const categories = group.categories
      .split(',')
      .map((item) => item.trim())
      .filter(Boolean)

    if (categories.length > 0) {
      acc[name] = categories
    }

    return acc
  }, {})

  return {
    name: draft.name.trim(),
    threshold: draft.threshold,
    security_threshold: draft.security_threshold,
    description: draft.description.trim(),
    tiers: draft.tiers.map((tier) => ({
      name: tier.name.trim(),
      description: tier.description?.trim() || '',
    })),
    categories: draft.categories.map((category) => ({
      name: category.name.trim(),
      tier: category.tier.trim(),
      description: category.description?.trim() || '',
      exemplars: category.exemplars
        .map((exemplar) => exemplar.trim())
        .filter(Boolean),
    })),
    tier_groups: tierGroups,
  }
}

export function formatSignalReference(reference: TaxonomySignalReference): string {
  return `${reference.name} -> ${reference.bind.kind}:${reference.bind.value}`
}

export function countSignalsForTier(record: TaxonomyClassifierRecord, tierName: string): number {
  return record.signal_references.filter(
    (reference) => reference.bind.kind === 'tier' && reference.bind.value === tierName
  ).length
}

export function countSignalsForCategory(record: TaxonomyClassifierRecord, categoryName: string): number {
  return record.signal_references.filter(
    (reference) => reference.bind.kind === 'category' && reference.bind.value === categoryName
  ).length
}

export function renameTierInDraft(
  draft: TaxonomyClassifierDraft,
  originalName: string,
  nextTier: TaxonomyTierDraft
): TaxonomyClassifierDraft {
  const nextName = nextTier.name.trim()
  return {
    ...draft,
    tiers: draft.tiers.map((tier) =>
      tier.name === originalName
        ? { name: nextName, description: nextTier.description.trim() }
        : tier
    ),
    categories: draft.categories.map((category) =>
      category.tier === originalName
        ? { ...category, tier: nextName }
        : category
    ),
  }
}

export function addTierToDraft(
  draft: TaxonomyClassifierDraft,
  nextTier: TaxonomyTierDraft
): TaxonomyClassifierDraft {
  return {
    ...draft,
    tiers: [
      ...draft.tiers,
      {
        name: nextTier.name.trim(),
        description: nextTier.description.trim(),
      },
    ],
  }
}

export function removeTierFromDraft(
  draft: TaxonomyClassifierDraft,
  tierName: string
): TaxonomyClassifierDraft {
  return {
    ...draft,
    tiers: draft.tiers.filter((tier) => tier.name !== tierName),
  }
}

function rewriteTierGroupsOnCategoryRename(
  tierGroups: TaxonomyClassifierDraft['tier_groups'],
  originalName: string,
  nextName: string
) {
  return tierGroups.map((group) => ({
    ...group,
    categories: group.categories
      .split(',')
      .map((item) => item.trim())
      .filter(Boolean)
      .map((item) => (item === originalName ? nextName : item))
      .join(', '),
  }))
}

function rewriteTierGroupsOnCategoryDelete(
  tierGroups: TaxonomyClassifierDraft['tier_groups'],
  categoryName: string
) {
  return tierGroups
    .map((group) => ({
      ...group,
      categories: group.categories
        .split(',')
        .map((item) => item.trim())
        .filter((item) => item && item !== categoryName)
        .join(', '),
    }))
    .filter((group) => group.name.trim())
}

export function renameCategoryInDraft(
  draft: TaxonomyClassifierDraft,
  originalName: string,
  nextCategory: TaxonomyCategoryDraft
): TaxonomyClassifierDraft {
  const nextName = nextCategory.name.trim()
  return {
    ...draft,
    categories: draft.categories.map((category) =>
      category.name === originalName
        ? {
            name: nextName,
            tier: nextCategory.tier.trim(),
            description: nextCategory.description.trim(),
            exemplars: nextCategory.exemplars
              .split('\n')
              .map((item) => item.trim())
              .filter(Boolean),
          }
        : category
    ),
    tier_groups: rewriteTierGroupsOnCategoryRename(draft.tier_groups, originalName, nextName),
  }
}

export function addCategoryToDraft(
  draft: TaxonomyClassifierDraft,
  nextCategory: TaxonomyCategoryDraft
): TaxonomyClassifierDraft {
  return {
    ...draft,
    categories: [
      ...draft.categories,
      {
        name: nextCategory.name.trim(),
        tier: nextCategory.tier.trim(),
        description: nextCategory.description.trim(),
        exemplars: nextCategory.exemplars
          .split('\n')
          .map((item) => item.trim())
          .filter(Boolean),
      },
    ],
  }
}

export function removeCategoryFromDraft(
  draft: TaxonomyClassifierDraft,
  categoryName: string
): TaxonomyClassifierDraft {
  return {
    ...draft,
    categories: draft.categories.filter((category) => category.name !== categoryName),
    tier_groups: rewriteTierGroupsOnCategoryDelete(draft.tier_groups, categoryName),
  }
}
