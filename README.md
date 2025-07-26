# Bilingual-GitHub 

## Overview
The bilingual-github repository provides reusable GitHub Actions workflows for translating GitHub issues and comments.
This documentation explains how to use the workflows for:
- Translating GitHub Issues
- Translating GitHub Comments

## Translation Behavior
- Translation only occurs when an issue has the "**need translation**" label
- When the label is present:
  - The issue content will be translated
  - All comments on that issue will be translated
  - Any edits to the issue or its comments will trigger new translations
- If an issue doesn't have the **"need translation"** label:
  - No translation will occur for the issue or its comments
  - The translation workflows will skip processing

## Steps to Use the Reusable Workflows in a Target Repository

#### 1. Translate GitHub Issues
To use the workflow for translating issues, create the following file in the target repository:
```bash
.github/workflows/translate-issues.yml
```
#### Code for the Calling Workflow (Translate Issues)
```python
name: Translate GitHub Issues

on:
  issues:
    types:
      - labeled
      - edited

jobs:
  call-translate-issues:
    if: |
      github.event.label.name == 'need translation' ||
      (github.event.action == 'edited' && contains(github.event.issue.labels.*.name, 'need translation'))
    uses: rimoapp/bilingual-github/.github/workflows/translate-issues.yml@main
    secrets:
      OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    with:
      issue_number: ${{ github.event.issue.number }}
```
#### 2. Translate GitHub Comments
To use the workflow for translating comments, create the following file in the target repository:
```base
.github/workflows/translate-comments.yml
```
#### Code for the Calling Workflow (Translate Comments)
```python
name: Translate Issue Comments

on:
  issue_comment:
    types:
      - created
      - edited
  issues:
    types:
      - labeled

jobs:
  call-translate-comments:
    if: |
      (github.event_name == 'issues' && github.event.label.name == 'need translation') ||
      github.event_name == 'issue_comment'
    uses: rimoapp/bilingual-github/.github/workflows/translate-comments.yml@main
    secrets:
      OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    with:
      issue_number: ${{ github.event.issue.number }}
      comment_id: ${{ github.event.comment.id || '' }}
```
#### 3. Translate .md files
```base
.github/workflows/translate-markdown.yml
```
```python
name: Translate Markdown Files
on:
  push:
    paths:
      - '**/*.md'
      - '.github/workflows/translate-markdown.yml'
    branches:
      - main

jobs:
  check-for-workflow-file:
    runs-on: ubuntu-latest
    outputs:
      workflow_added: ${{ steps.check_workflow.outputs.was_added }}
    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 2
      
      - name: Check if workflow file was just added
        id: check_workflow
        run: |
          # Check if the workflow file exists in the previous commit
          git checkout HEAD~1
          if [ ! -f .github/workflows/translate-markdown.yml ]; then
            echo "was_added=true" >> $GITHUB_OUTPUT
          else
            echo "was_added=false" >> $GITHUB_OUTPUT
          fi
          git checkout HEAD

  process-existing-files:
    needs: check-for-workflow-file
    if: needs.check-for-workflow-file.outputs.workflow_added == 'true'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Find and touch all Markdown files
        run: |
          find . -name "*.md" -not -name "*.ja.md" -exec touch {} \;
          
      - name: Commit touched files
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add -u
          git commit -m "Trigger initial translation for all Markdown files [skip ci]" || echo "No files to commit"
          git push || echo "Nothing to push"

  translate-markdown:
    needs: [check-for-workflow-file, process-existing-files]
    if: always() && (needs.check-for-workflow-file.outputs.workflow_added == 'false' || needs.process-existing-files.result == 'success' || needs.process-existing-files.result == 'skipped')
    uses: rimoapp/bilingual-github/.github/workflows/translate-markdown.yml@main
    secrets:
      OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}   
```

## Steps to Configure the Target Repository

#### 1. Add Secrets
- Navigate to target repository's settings on GitHub.
- Under **Security > Secrets and variables > Actions**, add the following
  secret:
  - **OPENAI_API_KEY:** Your OpenAI API key for translations.

#### 2. Enable the Workflows
Ensure that GitHub Actions are enabled for the repository. The workflows will trigger on the following events:
- **Translate Issues:**
  - When an issue is created or edited.
- **Translate Comments:**
  - When a comment is added or edited.

#### 3. Create the Required Label
- Go to your repository's **Issues > Labels**
- Create a new label named **"need translation"**
- This label will be used to control which issues get translated
  
## Testing the Integration

#### 1. Translate Issues
- Open or edit an issue in the target repository.
- Add the "need translation" label to the issue
- The ```call-translate-issues``` job will trigger and translate the issue.

#### 2. Translate Comments
- Add or edit a comment in an issue in the target repository.
- The ```call-translate-comments``` job will trigger and translate the comment.

#### 3. Monitor the Workflow
- Go to the **Actions** tab in the target repository.
- Check the status of the ```Translate GitHub Issues``` and ```Translate Issue Comments``` workflows.
- If successful, the translated content will appear in the issue or 
  comment.
