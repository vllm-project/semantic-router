import styles from './ConfigPage.module.css'
import { normalizeModelScores, type Category } from './configPageSupport'
import { cloneConfig, type LegacyCategoriesSectionProps } from './configPageRouterSectionSupport'

export default function ConfigPageLegacyCategoriesSection({
  config,
  isReadonly,
  openEditModal,
  saveConfig,
}: LegacyCategoriesSectionProps) {
  const defaultModel = config?.default_model || ''
  const categories = config?.categories || []

  return (
    <div className={styles.section}>
      <div className={styles.sectionHeader}>
        <h3 className={styles.sectionTitle}>Categories Configuration</h3>
        <span className={styles.badge}>{categories.length} categories</span>
      </div>
      <div className={styles.sectionContent}>
        <div className={styles.coreSettingsInline}>
          <div className={styles.inlineConfigRow}>
            <span className={styles.inlineConfigLabel}>Default Model:</span>
            <span className={styles.inlineConfigValue}>{defaultModel || 'N/A'}</span>
          </div>
          <div className={styles.inlineConfigRow}>
            <span className={styles.inlineConfigLabel}>Default Reasoning Effort:</span>
            <span className={`${styles.badge} ${styles[`badge${config?.default_reasoning_effort || 'medium'}`]}`}>
              {config?.default_reasoning_effort || 'medium'}
            </span>
          </div>
        </div>

        {categories.length > 0 ? (
          <div className={styles.categoryGridTwoColumn}>
            {categories.map((category: Category, index: number) => {
              const normalizedScores = normalizeModelScores(category.model_scores)
              const bestModel = normalizedScores[0]
              const useReasoning = bestModel?.use_reasoning || false
              const reasoningEffort = bestModel?.reasoning_effort || 'medium'
              const reasoningDescription = bestModel?.reasoning_description || ''

              return (
                <div key={index} className={styles.categoryCard}>
                  <div className={styles.categoryHeader}>
                    <span className={styles.categoryName}>{category.name}</span>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      {useReasoning && <span className={`${styles.reasoningBadge} ${styles[`reasoning${reasoningEffort}`]}`}>{reasoningEffort}</span>}
                      {!isReadonly && (
                        <button
                          className={styles.editButton}
                          onClick={() => {
                            openEditModal(
                              `Edit Category: ${category.name}`,
                              { system_prompt: category.system_prompt || '' },
                              [{ name: 'system_prompt', label: 'System Prompt', type: 'textarea', placeholder: 'Enter system prompt for this category...', description: 'Instructions for the model when handling this category' }],
                              async (data) => {
                                const newConfig = cloneConfig(config)
                                if (newConfig.categories) {
                                  newConfig.categories[index] = { ...category, ...data }
                                }
                                await saveConfig(newConfig)
                              }
                            )
                          }}
                        />
                      )}
                    </div>
                  </div>

                  {category.system_prompt && (
                    <div className={styles.systemPromptSection}>
                      <div className={styles.systemPromptLabel}>System Prompt</div>
                      <div className={styles.systemPromptText}>{category.system_prompt}</div>
                    </div>
                  )}
                  {reasoningDescription && <p className={styles.categoryDescription}>{reasoningDescription}</p>}

                  <div className={styles.categoryModels}>
                    <div className={styles.categoryModelsHeader}>
                      <span>Model Scores</span>
                      {!isReadonly && (
                        <button
                          className={styles.addModelButton}
                          onClick={() => {
                            const availableModels = config?.model_config ? Object.keys(config.model_config) : []
                            openEditModal(
                              `Add Model to ${category.name}`,
                              { model: availableModels[0] || '', score: 0.5, use_reasoning: false },
                              [
                                { name: 'model', label: 'Model', type: 'select', options: availableModels, required: true, description: 'Select from configured models' },
                                { name: 'score', label: 'Score', type: 'number', required: true, placeholder: '0.5', description: 'Model score (0-1)' },
                                { name: 'use_reasoning', label: 'Use Reasoning', type: 'boolean', description: 'Enable reasoning for this model in this category' },
                              ],
                              async (data) => {
                                const newConfig = cloneConfig(config)
                                if (newConfig.categories) {
                                  const updatedCategory = { ...category }
                                  const scores = normalizeModelScores(updatedCategory.model_scores)
                                  scores.push(data)
                                  updatedCategory.model_scores = scores
                                  newConfig.categories[index] = updatedCategory
                                }
                                await saveConfig(newConfig)
                              },
                              'add'
                            )
                          }}
                        />
                      )}
                    </div>
                    {normalizedScores.length > 0 ? (
                      normalizedScores.map((modelScore, modelIdx) => (
                        <div key={modelIdx} className={styles.modelScoreRow}>
                          <span className={styles.modelScoreName}>
                            {modelScore.model}
                            {modelScore.use_reasoning && <span className={styles.reasoningIcon}></span>}
                          </span>
                          <div className={styles.scoreBar}>
                            <div className={styles.scoreBarFill} style={{ width: `${(modelScore.score ?? 0) * 100}%` }}></div>
                            <span className={styles.scoreText}>{((modelScore.score ?? 0) * 100).toFixed(0)}%</span>
                          </div>
                          <div className={styles.modelScoreActions}>
                            {!isReadonly && (
                              <>
                                <button
                                  className={styles.editButton}
                                  onClick={() => {
                                    const availableModels = config?.model_config ? Object.keys(config.model_config) : []
                                    openEditModal(
                                      `Edit Model: ${modelScore.model}`,
                                      { ...modelScore },
                                      [
                                        { name: 'model', label: 'Model', type: 'select', options: availableModels, required: true, description: 'Select from configured models' },
                                        { name: 'score', label: 'Score', type: 'number', required: true, placeholder: '0.5', description: 'Model score (0-1)' },
                                        { name: 'use_reasoning', label: 'Use Reasoning', type: 'boolean', description: 'Enable reasoning for this model in this category' },
                                      ],
                                      async (data) => {
                                        const newConfig = cloneConfig(config)
                                        if (newConfig.categories) {
                                          const updatedCategory = { ...category }
                                          const scores = normalizeModelScores(updatedCategory.model_scores)
                                          scores[modelIdx] = data
                                          updatedCategory.model_scores = scores
                                          newConfig.categories[index] = updatedCategory
                                        }
                                        await saveConfig(newConfig)
                                      }
                                    )
                                  }}
                                />
                                <button
                                  className={styles.deleteButton}
                                  onClick={() => {
                                    if (confirm(`Remove model "${modelScore.model}" from this category?`)) {
                                      const newConfig = cloneConfig(config)
                                      if (newConfig.categories) {
                                        const updatedCategory = { ...category }
                                        const scores = normalizeModelScores(updatedCategory.model_scores)
                                        scores.splice(modelIdx, 1)
                                        updatedCategory.model_scores = scores
                                        newConfig.categories[index] = updatedCategory
                                      }
                                      saveConfig(newConfig)
                                    }
                                  }}
                                />
                              </>
                            )}
                          </div>
                        </div>
                      ))
                    ) : (
                      <div className={styles.emptyModelScores}>No models configured for this category</div>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        ) : (
          <div className={styles.emptyState}>No categories configured</div>
        )}
      </div>
    </div>
  )
}
