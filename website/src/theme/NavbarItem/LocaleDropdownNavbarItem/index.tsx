import React, { type ReactNode } from 'react'
import useDocusaurusContext from '@docusaurus/useDocusaurusContext'
import { useLocation } from '@docusaurus/router'
import { applyTrailingSlash } from '@docusaurus/utils-common'
import { translate } from '@docusaurus/Translate'
import { mergeSearchStrings, useHistorySelector } from '@docusaurus/theme-common'
import DropdownNavbarItem from '@theme/NavbarItem/DropdownNavbarItem'
import IconLanguage from '@theme/Icon/Language'
import type { LinkLikeNavbarItemProps } from '@theme/NavbarItem'
import type { Props } from '@theme/NavbarItem/LocaleDropdownNavbarItem'

import styles from './styles.module.css'

function normalizeLocalUrl(url: string): string {
  return `/${url.replace(/^\/+/, '')}`.replace(/\/+/g, '/')
}

function normalizeLocalePrefix(prefix: string | undefined): string | undefined {
  if (!prefix || prefix === '/') {
    return undefined
  }

  return normalizeLocalUrl(prefix).replace(/\/$/, '')
}

function stripPrefix(pathname: string, prefix: string): string | undefined {
  const normalizedPath = normalizeLocalUrl(pathname)
  const normalizedPrefix = normalizeLocalePrefix(prefix)

  if (!normalizedPrefix) {
    return undefined
  }

  const lowerPath = normalizedPath.toLowerCase()
  const lowerPrefix = normalizedPrefix.toLowerCase()

  if (lowerPath === lowerPrefix) {
    return ''
  }

  if (lowerPath.startsWith(`${lowerPrefix}/`)) {
    return normalizedPath.slice(normalizedPrefix.length + 1)
  }

  return undefined
}

function getPathnameSuffix(pathname: string, prefixes: Array<string | undefined>): string {
  for (const prefix of prefixes) {
    const suffix = prefix ? stripPrefix(pathname, prefix) : undefined
    if (typeof suffix !== 'undefined') {
      return suffix
    }
  }

  return normalizeLocalUrl(pathname).replace(/^\//, '')
}

function joinBaseUrl(baseUrl: string, pathnameSuffix: string): string {
  return normalizeLocalUrl(`${baseUrl.replace(/\/$/, '')}/${pathnameSuffix}`)
}

function useLocaleDropdownUtils() {
  const {
    siteConfig,
    i18n: { currentLocale, localeConfigs },
  } = useDocusaurusContext()
  const { baseUrl, trailingSlash } = siteConfig
  const { pathname } = useLocation()
  const search = useHistorySelector(history => history.location.search)
  const hash = useHistorySelector(history => history.location.hash)

  const getLocaleConfig = (locale: string) => {
    const localeConfig = localeConfigs[locale]
    if (!localeConfig) {
      throw new Error(`Docusaurus bug, no locale config found for locale=${locale}`)
    }
    return localeConfig
  }

  const currentLocaleConfig = getLocaleConfig(currentLocale)
  const canonicalPathname = applyTrailingSlash(pathname, { trailingSlash, baseUrl })
  const pathnameSuffix = getPathnameSuffix(canonicalPathname, [
    baseUrl,
    currentLocaleConfig.baseUrl,
    currentLocaleConfig.path,
  ])

  const getBaseURLForLocale = (locale: string) => {
    const localeConfig = getLocaleConfig(locale)
    const isSameDomain = localeConfig.url === siteConfig.url
    const localePath = joinBaseUrl(localeConfig.baseUrl, pathnameSuffix)
    const localeUrl = `${isSameDomain ? '' : localeConfig.url}${localePath}`

    return isSameDomain ? `pathname://${normalizeLocalUrl(localeUrl)}` : localeUrl
  }

  return {
    getURL: (locale: string, options: { queryString: string | undefined }) => {
      const finalSearch = mergeSearchStrings([search, options.queryString], 'append')
      return `${getBaseURLForLocale(locale)}${finalSearch}${hash}`
    },
    getLabel: (locale: string) => getLocaleConfig(locale).label,
    getLang: (locale: string) => getLocaleConfig(locale).htmlLang,
  }
}

export default function LocaleDropdownNavbarItem({
  mobile,
  dropdownItemsBefore,
  dropdownItemsAfter,
  queryString,
  ...props
}: Props): ReactNode {
  const utils = useLocaleDropdownUtils()

  const {
    i18n: { currentLocale, locales },
  } = useDocusaurusContext()

  const localeItems = locales.map((locale): LinkLikeNavbarItemProps => ({
    label: utils.getLabel(locale),
    lang: utils.getLang(locale),
    to: utils.getURL(locale, { queryString }),
    target: '_self',
    autoAddBaseUrl: false,
    className:
      locale === currentLocale
        ? mobile
          ? 'menu__link--active'
          : 'dropdown__link--active'
        : '',
  }))

  const items = [...dropdownItemsBefore, ...localeItems, ...dropdownItemsAfter]
  const dropdownLabel = mobile
    ? translate({
        message: 'Languages',
        id: 'theme.navbar.mobileLanguageDropdown.label',
        description: 'The label for the mobile language switcher dropdown',
      })
    : utils.getLabel(currentLocale)

  return (
    <DropdownNavbarItem
      {...props}
      mobile={mobile}
      label={(
        <>
          <IconLanguage className={styles.iconLanguage} />
          {dropdownLabel}
        </>
      )}
      items={items}
    />
  )
}
