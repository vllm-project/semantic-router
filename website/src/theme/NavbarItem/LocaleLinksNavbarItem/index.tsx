import React, { type ReactNode } from 'react'
import clsx from 'clsx'
import NavbarItem from '@theme/NavbarItem'
import { useLocaleItems } from '../LocaleDropdownNavbarItem'

interface Props {
  mobile?: boolean
  isDropdownItem?: boolean
  queryString?: string
  className?: string
}

/**
 * The locale choices as flat links, for use inside another dropdown. Docusaurus
 * rejects a `localeDropdown` nested in a dropdown, so the navbar's More menu
 * uses this instead while the standalone locale dropdown is out of room.
 */
export default function LocaleLinksNavbarItem({
  mobile,
  isDropdownItem,
  queryString,
  className,
}: Props): ReactNode {
  const items = useLocaleItems({ mobile, queryString })

  return (
    <>
      {items.map(item => (
        <NavbarItem
          key={item.lang}
          {...item}
          mobile={mobile}
          isDropdownItem={isDropdownItem}
          className={clsx(item.className, className)}
        />
      ))}
    </>
  )
}
