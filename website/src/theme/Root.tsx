import React from 'react'
import Root from '@theme-original/Root'
import ScrollToTop from '../components/ScrollToTop'

export default function RootWrapper(props: any): React.ReactElement {
  return (
    <>
      <Root {...props} />
      <ScrollToTop />
    </>
  )
}
