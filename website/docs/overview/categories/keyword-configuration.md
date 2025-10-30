---
sidebar_label: Keyword Classifier Configuration
---

# Keyword Classifier Configuration

The Keyword Classifier allows you to define custom routing rules based on the presence or absence of specific keywords or regular expressions within the input text. This provides a flexible and powerful way to categorize and route requests without relying solely on machine learning models.

## Configuration Structure

Keyword classification rules are defined in the `config.yaml` file under the `classifier.keyword_rules` section. Each rule is an object with the following parameters:

```yaml
classifier:
  keyword_rules:
    - category: "category_name"
      operator: "AND" | "OR" | "NOR"
      keywords: ["keyword1", "keyword2"]
      case_sensitive: true | false
```

## Configuration Parameters

### `category` (Required)

- **Type**: String
- **Description**: The classification label to assign if this rule matches. This will be the `category` returned by the classifier.
- **Example**: `"urgent_request"`, `"sensitive_data"`

### `operator` (Required)

- **Type**: String
- **Description**: Defines how multiple keywords within this rule are combined to determine a match.
- **Valid Values**:
  - `AND`: All keywords in the `keywords` list must be present in the input text for the rule to match.
  - `OR`: At least one keyword from the `keywords` list must be present in the input text for the rule to match.
  - `NOR`: None of the keywords from the `keywords` list must be present in the input text for the rule to match.
- **Example**: `"OR"`, `"AND"`, `"NOR"`

### `keywords` (Required)

- **Type**: Array of Strings
- **Description**: A list of strings that the classifier will search for in the input text. These strings are treated as regular expressions.
- **Behavior**:
  - For robustness and to allow for special characters, all keywords are automatically escaped using `regexp.QuoteMeta` before being compiled into regular expressions. This means you can use special regex characters (like `.`, `*`, `+`) as literal characters in your keywords without needing to escape them yourself in the `config.yaml`.
  - Word boundaries (`\b`) are conditionally applied around keywords that contain word characters. This helps ensure whole-word matching where appropriate (e.g., "cat" matches "cat" but not "category"). Keywords consisting solely of non-word characters (like punctuation) will not have word boundaries applied.
- **Example**: `["urgent", "immediate", "asap"]`, `["SSN", "social security number"]`, `["user\\.name@domain\\.com", "C:\\Program Files\\\\"]`

### `case_sensitive` (Optional)

- **Type**: Boolean
- **Description**: Determines whether the keyword matching should be case-sensitive.
- **Default**: `false` (case-insensitive)
- **Example**: `true`

## Complete Configuration Example

```yaml
classifier:
  keyword_rules:
    - category: "urgent_request"
      operator: "OR"
      keywords: ["urgent", "immediate", "asap"]
      case_sensitive: false
    - category: "sensitive_data"
      operator: "AND"
      keywords: ["SSN", "social security number", "credit card"]
      case_sensitive: false
    - category: "exclude_spam"
      operator: "NOR"
      keywords: ["buy now", "free money"]
      case_sensitive: false
    - category: "regex_pattern_match"
      operator: "OR"
      keywords: ["user\\.name@domain\\.com", "C:\\Program Files\\\\"] # Keywords are treated as regex
      case_sensitive: false
```
