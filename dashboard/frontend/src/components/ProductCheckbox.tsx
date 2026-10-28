import { forwardRef, type InputHTMLAttributes } from 'react'

import styles from './ProductCheckbox.module.css'

type ProductCheckboxProps = Omit<InputHTMLAttributes<HTMLInputElement>, 'type'>

const ProductCheckbox = forwardRef<HTMLInputElement, ProductCheckboxProps>(
  ({ className, ...props }, ref) => (
    <input
      {...props}
      ref={ref}
      type="checkbox"
      className={[styles.checkbox, className].filter(Boolean).join(' ')}
    />
  ),
)

ProductCheckbox.displayName = 'ProductCheckbox'

export default ProductCheckbox
