# Hybrid Translation System

## Overview

The hybrid translation system intelligently chooses between incremental and full translation based on the extent of changes in your markdown files.

## How It Works

### Change Detection

- Calculates percentage of lines changed between old and new versions
- Uses git history to get previous file versions
- Falls back to stored content from commit history

### Translation Strategy

- **Small changes (< 20%)**: Incremental translation using 3-file LLM approach
- **Large changes (≥ 20%)**: Full translation using existing system
- **New files**: Always full translation
- **Missing context**: Falls back to full translation

### 3-File Incremental Translation

When using incremental translation, the LLM receives:

1. **OLD_SOURCE**: Previous version of the source file
2. **NEW_SOURCE**: Current version of the source file
3. **CURRENT_TARGET**: Existing translation

The LLM analyzes the differences and updates only the necessary sections in the translation.

## Configuration

Create `.translation_config.json` in your repository root:

```json
{
  "incremental_threshold": 20,
  "enable_incremental": true
}
```

### Settings

- `incremental_threshold`: Percentage threshold (0-100) for switching to full translation
- `enable_incremental`: Enable/disable incremental translation feature

## Benefits

- **Efficiency**: Only translates changed sections
- **Consistency**: Preserves existing good translations
- **Speed**: Much faster for large files with small changes
- **Cost**: Reduces API calls and token usage
- **Quality**: Maintains translation coherence and style

## Fallback Behavior

The system gracefully falls back to full translation when:

- No previous version is available
- No existing translation exists
- Incremental translation fails
- Changes exceed the threshold
- Git history is unavailable

## File Access

The system works by:

1. Running in the target repository context
2. Using git commands to access file history
3. Reading existing translation files
4. Storing content in `.translation_commits.json` for future comparisons

## Example Workflow

1. Developer edits `docs/guide.en.md` (small change)
2. System detects 5% change (< 20% threshold)
3. Retrieves previous version via `git show HEAD~1:docs/guide.en.md`
4. Reads existing `docs/guide.ja.md`
5. Sends all 3 files to LLM for incremental update
6. LLM updates only the changed sections
7. Applies formatting and commits result

## Debugging

Enable debug output by checking the workflow logs for:

- Change percentage calculations
- Translation strategy decisions
- File access attempts
- LLM response handling
