# バイリンガル-GitHub

## 概要
bilingual-github リポジトリは、GitHubのIssueやコメントを翻訳するための再利用可能なGitHub Actionsワークフローを提供します。 このドキュメントでは、以下のワークフローの使用方法について説明します：
- GitHubのIssueの翻訳
- GitHubのコメントの翻訳

## 翻訳の動作
- Issue に **「need translation」** ラベルが付いている場合のみ翻訳が実行されます。
- ラベルが付いている場合：
  - Issue の内容が翻訳される
  - その Issue 内のすべてのコメントが翻訳される
  - Issue やコメントが編集されると、新しい翻訳が実行される
- **「need translation」** ラベルがない場合:
  - Issue やコメントは翻訳されない
  - 翻訳ワークフローは処理をスキップする

## 対象リポジトリで再利用可能なワークフローを使用する手順

#### 1. GitHubのIssueを翻訳

Issueを翻訳するためのワークフローを使用するには、対象リポジトリに以下のファイルを作成してください：
```bash
.github/workflows/translate_issues.yml
```
#### 呼び出しワークフローのコード (Issueの翻訳)
```python
name: Translate GitHub Issues

on:
  issues:
    types:
      - opened
      - edited

jobs:
  call-translate-issues:
    uses: rimoapp/bilingual-github/.github/workflows/translate-issues.yml@main
    secrets:
      OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    with:
      issue_number: ${{ github.event.issue.number }}
```
#### 2. GitHubのコメントを翻訳
コメントを翻訳するためのワークフローを使用するには、対象リポジトリに以下のファイルを作成してください：
```base
.github/workflows/translate_comments.yml
```
#### 呼び出しワークフローのコード (コメントの翻訳)
```python
name: Translate Issue Comments

on:
  issue_comment:
    types:
      - created
      - edited

jobs:
  call-translate-comments:
    uses: rimoapp/bilingual-github/.github/workflows/translate-comments.yml@main
    secrets:
      OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    with:
      issue_number: ${{ github.event.issue.number }}
      comment_id: ${{ github.event.comment.id }}
```
    
## 対象リポジトリを設定する手順

#### 1. シークレットを追加
- GitHubで対象リポジトリの設定ページに移動します。
- **セキュリティ > シークレットと変数 > Actions**の下に、以下のシークレットを追加します：
  - **OPENAI_API_KEY:** 翻訳用のOpenAI APIキー。

#### 2. ワークフローを有効化
対象リポジトリでGitHub Actionsが有効化されていることを確認してください。以下のイベントでワークフローがトリガーされます：
- **Issueの翻訳:**
  - Issueが作成または編集されたとき。
- **コメントの翻訳:**
  - コメントが追加または編集されたとき。

## 統合のテスト

#### 1. Issueの翻訳
- 対象リポジトリでIssueを作成または編集します。
- 「need translation」 というラベルを作成します。
- ```call-translate-issues``` ジョブがトリガーされ、Issueが翻訳されます。

#### 2. コメントの翻訳
- 対象リポジトリのIssueでコメントを追加または編集します。
- ```call-translate-comments``` ジョブがトリガーされ、コメントが翻訳されます。

#### 3. ワークフローの監視
- 対象リポジトリの**Actions**タブに移動します。
- ```Translate GitHub Issues```と```Translate Issue Comments```ワークフローのステータスを確認します。
- 成功した場合、翻訳された内容がIssueまたはコメントに表示されます。
