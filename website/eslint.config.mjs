import stylistic from '@stylistic/eslint-plugin'
import react from 'eslint-plugin-react'

export default [
  {
    ignores: ['.docusaurus', 'build'],
  },
  stylistic.configs['recommended-flat'],
  {
    files: ['**/*.js', '**/*.jsx'],
    plugins: {
      react: react,
    },
    rules: {
      ...react.configs['jsx-runtime'].rules,
    },
    languageOptions: {
      parserOptions: {
        ecmaFeatures: { jsx: true },
      },
    },
  },
]
