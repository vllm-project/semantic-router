import React from 'react'
import { useDoc } from '@docusaurus/plugin-content-docs/client'
import useDocusaurusContext from '@docusaurus/useDocusaurusContext'
import Translate from '@docusaurus/Translate'
import styles from './styles.module.css'

interface TranslationMeta {
  source_commit?: string
  source_file?: string
  outdated?: boolean
}

export default function TranslationBanner(): JSX.Element | null {
  const { i18n } = useDocusaurusContext()
  const { metadata, frontMatter } = useDoc()
  const translation = frontMatter.translation as TranslationMeta | undefined
  const isMTPE = Boolean(frontMatter.is_mtpe)

  // Show the banner for translated docs.
  //
  // Primary signal: explicit translation frontmatter (recommended).
  // Fallback signal: the doc source is under @site/i18n/... (some docs may be missing frontmatter).
  const hasTranslationFrontMatter = Boolean(
    translation?.source_commit || translation?.source_file || translation?.outdated !== undefined,
  )

  const isNonDefaultLocale = i18n.currentLocale !== i18n.defaultLocale
  const source = metadata.source ?? ''
  const isI18nSource = source.startsWith('@site/i18n/') || source.includes('/i18n/')

  if (!isNonDefaultLocale || (!hasTranslationFrontMatter && !isI18nSource)) {
    return null
  }

  // Build edit URL for GitHub contribution
  const editUrl = metadata.editUrl

  // Priority 1: Show outdated warning (regardless of is_mtpe)
  if (translation?.outdated) {
    return (
      <div className={styles.banner + ' ' + styles.outdated}>
        <span className={styles.icon}>Review</span>
        <span>
          <Translate
            id="translationBanner.outdated"
            description="Warning shown when translation is outdated"
          >
            此翻译可能已过时。英文原文已更新，中文尚未同步。
          </Translate>
        </span>
        {editUrl && (
          <a href={editUrl} target="_blank" rel="noopener noreferrer" className={styles.editLink}>
            <Translate id="translationBanner.edit" description="Edit button text">
              帮助更新
            </Translate>
          </a>
        )}
      </div>
    )
  }

  // Priority 2: Show AI translation notice only if not MTPE reviewed
  if (isMTPE) {
    return null
  }

  return (
    <div className={styles.banner + ' ' + styles.translated}>
      <span className={styles.icon}>AI</span>
      <span>
        <Translate
          id="translationBanner.aiTranslated"
          description="Notice shown for AI translated content"
        >
          本页面由 AI 从英文翻译，可能存在错误或不准确之处。
        </Translate>
      </span>
      {editUrl && (
        <a href={editUrl} target="_blank" rel="noopener noreferrer" className={styles.editLink}>
          <Translate id="translationBanner.edit" description="Edit button text">
            帮助改进
          </Translate>
        </a>
      )}
    </div>
  )
}
