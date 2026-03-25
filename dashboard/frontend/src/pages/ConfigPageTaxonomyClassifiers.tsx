import { useCallback, useEffect, useMemo, useState } from 'react'
import type { FieldConfig } from '../components/EditModal'
import { DataTable, type Column } from '../components/DataTable'
import TableHeader from '../components/TableHeader'
import pageStyles from './ConfigPage.module.css'
import ConfigPageTaxonomyClassifierEditor from './ConfigPageTaxonomyClassifierEditor'
import ConfigPageTaxonomyClassifierDetail from './ConfigPageTaxonomyClassifierDetail'
import type { OpenEditModal } from './configPageRouterSectionSupport'
import styles from './ConfigPageTaxonomyClassifiers.module.css'
import {
  addCategoryToDraft,
  addTierToDraft,
  categoryDraftFromCategory,
  classifierDraftFromRecord,
  countSignalsForCategory,
  countSignalsForTier,
  emptyTaxonomyCategoryDraft,
  emptyTaxonomyClassifierDraft,
  emptyTaxonomyTierDraft,
  normalizeTaxonomyClassifierListResponse,
  payloadFromDraft,
  removeCategoryFromDraft,
  removeTierFromDraft,
  renameCategoryInDraft,
  renameTierInDraft,
  tierDraftFromTier,
  type TaxonomyCategoryDraft,
  type TaxonomyClassifierCategory,
  type TaxonomyClassifierDraft,
  type TaxonomyClassifierRecord,
  type TaxonomyClassifierTier,
  type TaxonomyTierDraft,
} from './configPageTaxonomyClassifierSupport'

type TaxonomyManagerView = 'classifiers' | 'tiers' | 'categories' | 'exemplars'

interface ConfigPageTaxonomyClassifiersProps {
  isReadonly: boolean
  openEditModal: OpenEditModal
  activeView?: TaxonomyManagerView
}

interface ClassifierRow extends TaxonomyClassifierRecord {
  modeLabel: string
  tier_count: number
  category_count: number
  signal_count: number
}

interface TierRow extends TaxonomyClassifierTier {
  category_count: number
  signal_count: number
}

interface CategoryRow extends TaxonomyClassifierCategory {
  exemplar_count: number
  signal_count: number
}

interface ExemplarRow {
  key: string
  category: string
  tier: string
  exemplar: string
  exemplarIndex: number
}

interface TaxonomyExemplarDraft {
  category: string
  exemplar: string
}

function buildClassifierModeLabel(classifier: TaxonomyClassifierRecord): string {
  if (classifier.builtin) {
    return 'Built-in'
  }
  if (classifier.editable) {
    return 'Custom managed'
  }
  return 'External'
}

function addExemplarToDraft(
  draft: TaxonomyClassifierDraft,
  categoryName: string,
  exemplar: string
): TaxonomyClassifierDraft {
  return {
    ...draft,
    categories: draft.categories.map((category) =>
      category.name === categoryName
        ? { ...category, exemplars: [...category.exemplars, exemplar] }
        : category
    ),
  }
}

function updateExemplarInDraft(
  draft: TaxonomyClassifierDraft,
  originalCategory: string,
  originalIndex: number,
  nextCategory: string,
  nextExemplar: string
): TaxonomyClassifierDraft {
  return {
    ...draft,
    categories: draft.categories.map((category) => {
      if (category.name === originalCategory && category.name === nextCategory) {
        return {
          ...category,
          exemplars: category.exemplars.map((exemplar, index) =>
            index === originalIndex ? nextExemplar : exemplar
          ),
        }
      }
      if (category.name === originalCategory) {
        return {
          ...category,
          exemplars: category.exemplars.filter((_, index) => index !== originalIndex),
        }
      }
      if (category.name === nextCategory) {
        return {
          ...category,
          exemplars: [...category.exemplars, nextExemplar],
        }
      }
      return category
    }),
  }
}

function removeExemplarFromDraft(
  draft: TaxonomyClassifierDraft,
  categoryName: string,
  exemplarIndex: number
): TaxonomyClassifierDraft {
  return {
    ...draft,
    categories: draft.categories.map((category) =>
      category.name === categoryName
        ? {
            ...category,
            exemplars: category.exemplars.filter((_, index) => index !== exemplarIndex),
          }
        : category
    ),
  }
}

export default function ConfigPageTaxonomyClassifiers({
  isReadonly,
  openEditModal,
  activeView = 'classifiers',
}: ConfigPageTaxonomyClassifiersProps) {
  const [classifiers, setClassifiers] = useState<TaxonomyClassifierRecord[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedClassifierName, setSelectedClassifierName] = useState('')
  const [classifierSearch, setClassifierSearch] = useState('')
  const [tierSearch, setTierSearch] = useState('')
  const [categorySearch, setCategorySearch] = useState('')
  const [exemplarSearch, setExemplarSearch] = useState('')

  const loadClassifiers = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch('/api/router/config/classifiers')
      if (!response.ok) {
        const message = await response.text()
        throw new Error(message || `HTTP ${response.status}: ${response.statusText}`)
      }
      const payload = normalizeTaxonomyClassifierListResponse(await response.json())
      const items = payload.items || []
      setClassifiers(items)
      setSelectedClassifierName((current) => {
        if (current && items.some((classifier) => classifier.name === current)) {
          return current
        }
        return items.find((classifier) => classifier.editable)?.name ?? items[0]?.name ?? ''
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load taxonomy classifiers')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void loadClassifiers()
  }, [loadClassifiers])

  const classifierRows = useMemo<ClassifierRow[]>(() => {
    return classifiers
      .filter((classifier) => {
        const query = classifierSearch.trim().toLowerCase()
        if (!query) {
          return true
        }
        return (
          classifier.name.toLowerCase().includes(query) ||
          (classifier.description ?? '').toLowerCase().includes(query) ||
          classifier.source.path.toLowerCase().includes(query)
        )
      })
      .map((classifier) => ({
        ...classifier,
        modeLabel: buildClassifierModeLabel(classifier),
        tier_count: classifier.tiers.length,
        category_count: classifier.categories.length,
        signal_count: classifier.signal_references.length,
      }))
  }, [classifierSearch, classifiers])

  const selectedClassifier = useMemo(
    () => classifiers.find((classifier) => classifier.name === selectedClassifierName) ?? classifierRows[0] ?? null,
    [classifierRows, classifiers, selectedClassifierName]
  )

  const tierRows = useMemo<TierRow[]>(() => {
    if (!selectedClassifier) {
      return []
    }
    const query = tierSearch.trim().toLowerCase()
    return selectedClassifier.tiers
      .filter((tier) => {
        if (!query) {
          return true
        }
        return (
          tier.name.toLowerCase().includes(query) ||
          (tier.description ?? '').toLowerCase().includes(query)
        )
      })
      .map((tier) => ({
        ...tier,
        category_count: selectedClassifier.categories.filter((category) => category.tier === tier.name).length,
        signal_count: countSignalsForTier(selectedClassifier, tier.name),
      }))
  }, [selectedClassifier, tierSearch])

  const categoryRows = useMemo<CategoryRow[]>(() => {
    if (!selectedClassifier) {
      return []
    }
    const query = categorySearch.trim().toLowerCase()
    return selectedClassifier.categories
      .filter((category) => {
        if (!query) {
          return true
        }
        return (
          category.name.toLowerCase().includes(query) ||
          category.tier.toLowerCase().includes(query) ||
          (category.description ?? '').toLowerCase().includes(query)
        )
      })
      .map((category) => ({
        ...category,
        exemplar_count: category.exemplars.length,
        signal_count: countSignalsForCategory(selectedClassifier, category.name),
      }))
  }, [categorySearch, selectedClassifier])

  const exemplarRows = useMemo<ExemplarRow[]>(() => {
    if (!selectedClassifier) {
      return []
    }
    const query = exemplarSearch.trim().toLowerCase()
    return selectedClassifier.categories.flatMap((category) =>
      category.exemplars
        .map((exemplar, exemplarIndex) => ({
          key: `${category.name}:${exemplarIndex}`,
          category: category.name,
          tier: category.tier,
          exemplar,
          exemplarIndex,
        }))
        .filter((row) => {
          if (!query) {
            return true
          }
          return (
            row.category.toLowerCase().includes(query) ||
            row.tier.toLowerCase().includes(query) ||
            row.exemplar.toLowerCase().includes(query)
          )
        })
    )
  }, [exemplarSearch, selectedClassifier])

  const counts = useMemo(
    () => ({
      total: classifiers.length,
      builtin: classifiers.filter((classifier) => classifier.builtin).length,
      custom: classifiers.filter((classifier) => !classifier.builtin).length,
    }),
    [classifiers]
  )

  const classifierEditorField = useCallback(
    (disableName: boolean): FieldConfig[] => [
      {
        name: 'draft',
        label: 'Classifier',
        type: 'custom',
        customRender: (value, onChange) => (
          <ConfigPageTaxonomyClassifierEditor
            value={value}
            onChange={(nextValue) => onChange(nextValue)}
            disableName={disableName}
          />
        ),
      },
    ],
    []
  )

  const persistClassifier = useCallback(
    async (
      endpoint: string,
      method: 'POST' | 'PUT',
      draft: TaxonomyClassifierDraft,
      nextSelection?: string
    ) => {
      const response = await fetch(endpoint, {
        method,
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payloadFromDraft(draft)),
      })
      if (!response.ok) {
        const message = await response.text()
        throw new Error(message || `HTTP ${response.status}: ${response.statusText}`)
      }
      if (nextSelection) {
        setSelectedClassifierName(nextSelection)
      }
      await loadClassifiers()
    },
    [loadClassifiers]
  )

  const persistSelectedClassifier = useCallback(
    async (mutate: (draft: TaxonomyClassifierDraft) => TaxonomyClassifierDraft) => {
      if (!selectedClassifier || !selectedClassifier.editable) {
        throw new Error('Select a classifier to edit.')
      }
      const currentDraft = classifierDraftFromRecord(selectedClassifier)
      const nextDraft = mutate(currentDraft)
      await persistClassifier(
        `/api/router/config/classifiers/${selectedClassifier.name}`,
        'PUT',
        nextDraft,
        selectedClassifier.name
      )
    },
    [persistClassifier, selectedClassifier]
  )

  const openCreateModal = () => {
    openEditModal<{ draft: TaxonomyClassifierDraft }>(
      'Add Taxonomy Classifier',
      { draft: emptyTaxonomyClassifierDraft() },
      classifierEditorField(false),
      async (data) => {
        const nextName = data.draft.name.trim()
        await persistClassifier('/api/router/config/classifiers', 'POST', data.draft, nextName)
      },
      'add'
    )
  }

  const openEditClassifierModal = (classifier: TaxonomyClassifierRecord) => {
    openEditModal<{ draft: TaxonomyClassifierDraft }>(
      `Edit ${classifier.name}`,
      { draft: classifierDraftFromRecord(classifier) },
      classifierEditorField(true),
      async (data) => {
        await persistClassifier(`/api/router/config/classifiers/${classifier.name}`, 'PUT', data.draft, classifier.name)
      },
      'edit'
    )
  }

  const handleDeleteClassifier = async (classifier: TaxonomyClassifierRecord) => {
    if (!window.confirm(`Delete classifier "${classifier.name}"? This removes it from the active taxonomy catalog.`)) {
      return
    }

    setError(null)
    try {
      const response = await fetch(`/api/router/config/classifiers/${classifier.name}`, {
        method: 'DELETE',
      })
      if (!response.ok) {
        const message = await response.text()
        throw new Error(message || `HTTP ${response.status}: ${response.statusText}`)
      }
      if (selectedClassifierName === classifier.name) {
        setSelectedClassifierName('')
      }
      await loadClassifiers()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete taxonomy classifier')
    }
  }

  const openAddTierModal = () => {
    if (!selectedClassifier) {
      return
    }
    openEditModal<TaxonomyTierDraft>(
      `Add Tier · ${selectedClassifier.name}`,
      emptyTaxonomyTierDraft(),
      [
        { name: 'name', label: 'Tier Name', type: 'text', required: true, placeholder: 'privacy_policy' },
        { name: 'description', label: 'Description', type: 'textarea', placeholder: 'What this tier means.' },
      ],
      async (data) => {
        const nextName = data.name.trim()
        if (selectedClassifier.tiers.some((tier) => tier.name === nextName)) {
          throw new Error(`Tier "${nextName}" already exists.`)
        }
        await persistSelectedClassifier((draft) => addTierToDraft(draft, data))
      },
      'add'
    )
  }

  const openEditTierModal = (tier: TaxonomyClassifierTier) => {
    if (!selectedClassifier) {
      return
    }
    openEditModal<TaxonomyTierDraft>(
      `Edit Tier · ${tier.name}`,
      tierDraftFromTier(tier),
      [
        { name: 'name', label: 'Tier Name', type: 'text', required: true, placeholder: 'privacy_policy' },
        { name: 'description', label: 'Description', type: 'textarea', placeholder: 'What this tier means.' },
      ],
      async (data) => {
        const nextName = data.name.trim()
        if (nextName !== tier.name && selectedClassifier.tiers.some((existingTier) => existingTier.name === nextName)) {
          throw new Error(`Tier "${nextName}" already exists.`)
        }
        if (nextName !== tier.name && countSignalsForTier(selectedClassifier, tier.name) > 0) {
          throw new Error(`Tier "${tier.name}" is referenced by taxonomy signals. Update those signals before renaming it.`)
        }
        await persistSelectedClassifier((draft) => renameTierInDraft(draft, tier.name, data))
      },
      'edit'
    )
  }

  const handleDeleteTier = async (tier: TaxonomyClassifierTier) => {
    if (!selectedClassifier) {
      return
    }
    const boundCategories = selectedClassifier.categories.filter((category) => category.tier === tier.name)
    if (boundCategories.length > 0) {
      setError(`Tier "${tier.name}" still owns ${boundCategories.length} categor${boundCategories.length === 1 ? 'y' : 'ies'}. Reassign or delete those categories first.`)
      return
    }
    if (countSignalsForTier(selectedClassifier, tier.name) > 0) {
      setError(`Tier "${tier.name}" is referenced by taxonomy signals. Remove those signals first.`)
      return
    }
    if (!window.confirm(`Delete tier "${tier.name}" from ${selectedClassifier.name}?`)) {
      return
    }
    setError(null)
    try {
      await persistSelectedClassifier((draft) => removeTierFromDraft(draft, tier.name))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete tier')
    }
  }

  const openAddCategoryModal = () => {
    if (!selectedClassifier) {
      return
    }
    if (selectedClassifier.tiers.length === 0) {
      setError('Add at least one tier before creating categories.')
      return
    }
    openEditModal<TaxonomyCategoryDraft>(
      `Add Category · ${selectedClassifier.name}`,
      emptyTaxonomyCategoryDraft(selectedClassifier.tiers[0]?.name ?? ''),
      [
        { name: 'name', label: 'Category Name', type: 'text', required: true, placeholder: 'proprietary_code' },
        { name: 'tier', label: 'Tier', type: 'select', required: true, options: selectedClassifier.tiers.map((tier) => tier.name) },
        { name: 'description', label: 'Description', type: 'textarea', placeholder: 'What this category catches.' },
        { name: 'exemplars', label: 'Exemplars', type: 'textarea', required: true, placeholder: 'One exemplar per line' },
      ],
      async (data) => {
        const nextName = data.name.trim()
        if (selectedClassifier.categories.some((category) => category.name === nextName)) {
          throw new Error(`Category "${nextName}" already exists.`)
        }
        await persistSelectedClassifier((draft) => addCategoryToDraft(draft, data))
      },
      'add'
    )
  }

  const openEditCategoryModal = (category: TaxonomyClassifierCategory) => {
    if (!selectedClassifier) {
      return
    }
    openEditModal<TaxonomyCategoryDraft>(
      `Edit Category · ${category.name}`,
      categoryDraftFromCategory(category),
      [
        { name: 'name', label: 'Category Name', type: 'text', required: true, placeholder: 'proprietary_code' },
        { name: 'tier', label: 'Tier', type: 'select', required: true, options: selectedClassifier.tiers.map((tier) => tier.name) },
        { name: 'description', label: 'Description', type: 'textarea', placeholder: 'What this category catches.' },
        { name: 'exemplars', label: 'Exemplars', type: 'textarea', required: true, placeholder: 'One exemplar per line' },
      ],
      async (data) => {
        const nextName = data.name.trim()
        if (
          nextName !== category.name &&
          selectedClassifier.categories.some((existingCategory) => existingCategory.name === nextName)
        ) {
          throw new Error(`Category "${nextName}" already exists.`)
        }
        if (nextName !== category.name && countSignalsForCategory(selectedClassifier, category.name) > 0) {
          throw new Error(`Category "${category.name}" is referenced by taxonomy signals. Update those signals before renaming it.`)
        }
        await persistSelectedClassifier((draft) => renameCategoryInDraft(draft, category.name, data))
      },
      'edit'
    )
  }

  const handleDeleteCategory = async (category: TaxonomyClassifierCategory) => {
    if (!selectedClassifier) {
      return
    }
    if (countSignalsForCategory(selectedClassifier, category.name) > 0) {
      setError(`Category "${category.name}" is referenced by taxonomy signals. Remove those signals first.`)
      return
    }
    if (!window.confirm(`Delete category "${category.name}" from ${selectedClassifier.name}?`)) {
      return
    }
    setError(null)
    try {
      await persistSelectedClassifier((draft) => removeCategoryFromDraft(draft, category.name))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete category')
    }
  }

  const openAddExemplarModal = () => {
    if (!selectedClassifier) {
      return
    }
    if (selectedClassifier.categories.length === 0) {
      setError('Add at least one category before creating exemplars.')
      return
    }
    openEditModal<TaxonomyExemplarDraft>(
      `Add Exemplar · ${selectedClassifier.name}`,
      {
        category: selectedClassifier.categories[0]?.name ?? '',
        exemplar: '',
      },
      [
        { name: 'category', label: 'Category', type: 'select', required: true, options: selectedClassifier.categories.map((category) => category.name) },
        { name: 'exemplar', label: 'Exemplar', type: 'textarea', required: true, placeholder: 'Representative prompt for this category' },
      ],
      async (data) => {
        const nextCategory = data.category.trim()
        const nextExemplar = data.exemplar.trim()
        if (!nextCategory || !nextExemplar) {
          throw new Error('Category and exemplar are required.')
        }
        await persistSelectedClassifier((draft) => addExemplarToDraft(draft, nextCategory, nextExemplar))
      },
      'add'
    )
  }

  const openEditExemplarModal = (row: ExemplarRow) => {
    if (!selectedClassifier) {
      return
    }
    openEditModal<TaxonomyExemplarDraft>(
      `Edit Exemplar · ${row.category}`,
      {
        category: row.category,
        exemplar: row.exemplar,
      },
      [
        { name: 'category', label: 'Category', type: 'select', required: true, options: selectedClassifier.categories.map((category) => category.name) },
        { name: 'exemplar', label: 'Exemplar', type: 'textarea', required: true, placeholder: 'Representative prompt for this category' },
      ],
      async (data) => {
        const nextCategory = data.category.trim()
        const nextExemplar = data.exemplar.trim()
        if (!nextCategory || !nextExemplar) {
          throw new Error('Category and exemplar are required.')
        }
        await persistSelectedClassifier((draft) =>
          updateExemplarInDraft(draft, row.category, row.exemplarIndex, nextCategory, nextExemplar)
        )
      },
      'edit'
    )
  }

  const handleDeleteExemplar = async (row: ExemplarRow) => {
    if (!selectedClassifier) {
      return
    }
    if (!window.confirm(`Delete this exemplar from "${row.category}"?`)) {
      return
    }
    setError(null)
    try {
      await persistSelectedClassifier((draft) => removeExemplarFromDraft(draft, row.category, row.exemplarIndex))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete exemplar')
    }
  }

  const classifierColumns = useMemo<Column<ClassifierRow>[]>(() => [
    {
      key: 'name',
      header: 'Classifier',
      sortable: true,
      render: (row) => (
        <div className={styles.primaryCell}>
          <div className={styles.primaryCellTitleRow}>
            <span className={styles.primaryCellTitle}>{row.name}</span>
            {selectedClassifier?.name === row.name ? <span className={styles.selectedBadge}>Selected</span> : null}
          </div>
          <span className={styles.primaryCellMeta}>{row.description || 'No classifier description provided.'}</span>
        </div>
      ),
    },
    {
      key: 'modeLabel',
      header: 'Mode',
      sortable: true,
      render: (row) => <span className={styles.tableBadge}>{row.modeLabel}</span>,
    },
    {
      key: 'tier_count',
      header: 'Tiers',
      sortable: true,
      align: 'center',
    },
    {
      key: 'category_count',
      header: 'Categories',
      sortable: true,
      align: 'center',
    },
    {
      key: 'signal_count',
      header: 'Signals',
      sortable: true,
      align: 'center',
    },
    {
      key: 'source',
      header: 'Source',
      render: (row) => <code className={styles.inlineCode}>{row.source.path}</code>,
    },
    {
      key: 'actions',
      header: 'Actions',
      align: 'right',
      render: (row) => (
        <div className={styles.tableActionGroup}>
          <button type="button" className={styles.secondaryButton} onClick={() => setSelectedClassifierName(row.name)}>
            {selectedClassifier?.name === row.name ? 'Open' : 'View'}
          </button>
          {row.editable && !isReadonly ? (
            <>
              <button type="button" className={styles.secondaryButton} onClick={() => openEditClassifierModal(row)}>
                Edit
              </button>
              <button type="button" className={styles.removeButton} onClick={() => void handleDeleteClassifier(row)}>
                Delete
              </button>
            </>
          ) : null}
        </div>
      ),
    },
  ], [handleDeleteClassifier, isReadonly, openEditClassifierModal, selectedClassifier?.name])

  const tierColumns = useMemo<Column<TierRow>[]>(() => [
    {
      key: 'name',
      header: 'Tier',
      sortable: true,
      render: (row) => (
        <div className={styles.primaryCell}>
          <span className={styles.primaryCellTitle}>{row.name}</span>
          <span className={styles.primaryCellMeta}>{row.description || 'No tier description provided.'}</span>
        </div>
      ),
    },
    {
      key: 'category_count',
      header: 'Categories',
      sortable: true,
      align: 'center',
    },
    {
      key: 'signal_count',
      header: 'Signals',
      sortable: true,
      align: 'center',
    },
    {
      key: 'actions',
      header: 'Actions',
      align: 'right',
      render: (row) => (
        <div className={styles.tableActionGroup}>
          {!isReadonly && selectedClassifier?.editable ? (
            <>
              <button type="button" className={styles.secondaryButton} onClick={() => openEditTierModal(row)}>
                Edit
              </button>
              <button type="button" className={styles.removeButton} onClick={() => void handleDeleteTier(row)}>
                Delete
              </button>
            </>
          ) : (
            <span className={styles.readOnlyHint}>Read-only</span>
          )}
        </div>
      ),
    },
  ], [handleDeleteTier, isReadonly, openEditTierModal, selectedClassifier?.editable])

  const categoryColumns = useMemo<Column<CategoryRow>[]>(() => [
    {
      key: 'name',
      header: 'Category',
      sortable: true,
      render: (row) => (
        <div className={styles.primaryCell}>
          <span className={styles.primaryCellTitle}>{row.name}</span>
          <span className={styles.primaryCellMeta}>{row.description || 'No category description provided.'}</span>
        </div>
      ),
    },
    {
      key: 'tier',
      header: 'Tier',
      sortable: true,
      render: (row) => <span className={styles.tableBadge}>{row.tier}</span>,
    },
    {
      key: 'exemplar_count',
      header: 'Exemplars',
      sortable: true,
      align: 'center',
    },
    {
      key: 'signal_count',
      header: 'Signals',
      sortable: true,
      align: 'center',
    },
    {
      key: 'actions',
      header: 'Actions',
      align: 'right',
      render: (row) => (
        <div className={styles.tableActionGroup}>
          {!isReadonly && selectedClassifier?.editable ? (
            <>
              <button type="button" className={styles.secondaryButton} onClick={() => openEditCategoryModal(row)}>
                Edit
              </button>
              <button type="button" className={styles.removeButton} onClick={() => void handleDeleteCategory(row)}>
                Delete
              </button>
            </>
          ) : (
            <span className={styles.readOnlyHint}>Read-only</span>
          )}
        </div>
      ),
    },
  ], [handleDeleteCategory, isReadonly, openEditCategoryModal, selectedClassifier?.editable])

  const exemplarColumns = useMemo<Column<ExemplarRow>[]>(() => [
    {
      key: 'category',
      header: 'Category',
      sortable: true,
      render: (row) => <span className={styles.tableBadge}>{row.category}</span>,
    },
    {
      key: 'tier',
      header: 'Tier',
      sortable: true,
      render: (row) => <span className={styles.tableBadge}>{row.tier}</span>,
    },
    {
      key: 'exemplar',
      header: 'Exemplar',
      render: (row) => (
        <div className={styles.primaryCell}>
          <span className={styles.primaryCellMeta}>{row.exemplar}</span>
        </div>
      ),
    },
    {
      key: 'actions',
      header: 'Actions',
      align: 'right',
      render: (row) => (
        <div className={styles.tableActionGroup}>
          {!isReadonly && selectedClassifier?.editable ? (
            <>
              <button type="button" className={styles.secondaryButton} onClick={() => openEditExemplarModal(row)}>
                Edit
              </button>
              <button type="button" className={styles.removeButton} onClick={() => void handleDeleteExemplar(row)}>
                Delete
              </button>
            </>
          ) : (
            <span className={styles.readOnlyHint}>Read-only</span>
          )}
        </div>
      ),
    },
  ], [handleDeleteExemplar, isReadonly, openEditExemplarModal, selectedClassifier?.editable])

  return (
    <section id="taxonomy-classifiers" className={styles.section}>
      <div className={styles.summaryGrid}>
        <article className={styles.summaryCard}>
          <span className={styles.summaryLabel}>Total Classifiers</span>
          <strong className={styles.summaryValue}>{counts.total}</strong>
          <span className={styles.summaryHint}>All taxonomy packages currently discoverable by the router.</span>
        </article>
        <article className={styles.summaryCard}>
          <span className={styles.summaryLabel}>Built-In</span>
          <strong className={styles.summaryValue}>{counts.builtin}</strong>
          <span className={styles.summaryHint}>Router-shipped classifiers that can be updated or deleted explicitly.</span>
        </article>
        <article className={styles.summaryCard}>
          <span className={styles.summaryLabel}>Custom Managed</span>
          <strong className={styles.summaryValue}>{counts.custom}</strong>
          <span className={styles.summaryHint}>Editable classifier directories under `classifiers/custom/`.</span>
        </article>
      </div>

      {loading ? <div className={styles.notice}>Loading classifier catalog…</div> : null}
      {error ? <div className={styles.error}>{error}</div> : null}

      <div className={pageStyles.sectionTableBlock}>
        <TableHeader
          title={activeView === 'classifiers' ? 'Classifier Catalog' : 'Classifier Context'}
          count={classifierRows.length}
          searchPlaceholder="Search classifier name, description, or source"
          searchValue={classifierSearch}
          onSearchChange={setClassifierSearch}
          onSecondaryAction={() => {
            void loadClassifiers()
          }}
          secondaryActionText="Refresh"
          onAdd={!isReadonly ? openCreateModal : undefined}
          addButtonText="Add Classifier"
          variant="embedded"
        />
        <DataTable
          columns={classifierColumns}
          data={classifierRows}
          keyExtractor={(row) => row.name}
          emptyMessage="No taxonomy classifiers found."
          className={pageStyles.managerTable}
        />
      </div>

      <ConfigPageTaxonomyClassifierDetail selectedClassifier={selectedClassifier} />

      {activeView === 'tiers' ? (
        <div className={pageStyles.sectionTableBlock}>
          <TableHeader
            title={selectedClassifier ? `Tiers · ${selectedClassifier.name}` : 'Tiers'}
            count={tierRows.length}
            searchPlaceholder="Search tiers"
            searchValue={tierSearch}
            onSearchChange={setTierSearch}
            onAdd={!isReadonly && selectedClassifier?.editable ? openAddTierModal : undefined}
            addButtonText="Add Tier"
            variant="embedded"
          />
          <DataTable
            columns={tierColumns}
            data={tierRows}
            keyExtractor={(row) => row.name}
            emptyMessage={selectedClassifier ? 'No tiers defined for this classifier.' : 'Select a classifier first.'}
            className={pageStyles.managerTable}
          />
        </div>
      ) : null}

      {activeView === 'categories' ? (
        <div className={pageStyles.sectionTableBlock}>
          <TableHeader
            title={selectedClassifier ? `Categories · ${selectedClassifier.name}` : 'Categories'}
            count={categoryRows.length}
            searchPlaceholder="Search categories"
            searchValue={categorySearch}
            onSearchChange={setCategorySearch}
            onAdd={!isReadonly && selectedClassifier?.editable ? openAddCategoryModal : undefined}
            addButtonText="Add Category"
            variant="embedded"
          />
          <DataTable
            columns={categoryColumns}
            data={categoryRows}
            keyExtractor={(row) => row.name}
            emptyMessage={selectedClassifier ? 'No categories defined for this classifier.' : 'Select a classifier first.'}
            className={pageStyles.managerTable}
          />
        </div>
      ) : null}

      {activeView === 'exemplars' ? (
        <div className={pageStyles.sectionTableBlock}>
          <TableHeader
            title={selectedClassifier ? `Exemplars · ${selectedClassifier.name}` : 'Exemplars'}
            count={exemplarRows.length}
            searchPlaceholder="Search exemplar text, category, or tier"
            searchValue={exemplarSearch}
            onSearchChange={setExemplarSearch}
            onAdd={!isReadonly && selectedClassifier?.editable ? openAddExemplarModal : undefined}
            addButtonText="Add Exemplar"
            variant="embedded"
          />
          <DataTable
            columns={exemplarColumns}
            data={exemplarRows}
            keyExtractor={(row) => row.key}
            emptyMessage={selectedClassifier ? 'No exemplars defined for this classifier.' : 'Select a classifier first.'}
            className={pageStyles.managerTable}
          />
        </div>
      ) : null}
    </section>
  )
}
