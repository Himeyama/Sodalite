<h1><img src="docs/icon.png" height="24" />&nbsp;Sodalite</h1>

日本語 | [English](README.en.md)

Stable Diffusion 画像生成デスクトップアプリ。一から独自実装したもの。

<img width="600" alt="image" src="https://github.com/user-attachments/assets/e1151f70-408a-4a58-9549-4c01de492100" />

- **フロントエンド**: WinUI3 (.NET 9 / Windows App SDK)
- **バックエンド**: Python 3.13+ / FastAPI / diffusers (uv管理)
- **通信方式**: フロントエンドがバックエンドをローカルサブプロセスとして起動し、HTTP経由で通信する
- セーフティチェッカーは無効化しており、🔞NSFW 画像の生成も可能。生成物の利用・公開は各自の責任・使用モデルのライセンス・居住地の法令に従うこと。

## ダウンロードとインストール

- GPU: VRAM 12 GiB 以上が推奨です。(VRAM 10 GiB でも一応動作します)
- `uv` コマンドの Windows 版インストールが必要です。  
  https://docs.astral.sh/uv/#installation

- [Releases](https://github.com/Himeyama/Sodalite/releases) から、最新版の EXE ファイルをダウンロードしてください。  
  個人開発アプリのためブラウザで一時ブロックされますが、「⋯」→「保存」→「削除」ボタンの「∨」→「保持する」から保存可能です。

  <img src="docs\download-save-app.png" width="200" />

  セキュリティー上、インストーラーは [Actions](https://github.com/Himeyama/Sodalite/actions) から自動作成してます。

## ディレクトリ構成

```
Sodalite/
├── backend/           # Pythonバックエンド (uv管理, FastAPI + diffusers)
│   ├── src/sodalite_backend/
│   │   ├── main.py            # FastAPIエントリポイント
│   │   ├── config.py          # 起動設定(ポート・モデルID)
│   │   ├── api/                # REST APIエンドポイント
│   │   ├── schemas/            # Pydanticリクエスト/レスポンスモデル
│   │   ├── inference/           # diffusersパイプライン管理・サンプラー
│   │   └── imaging/            # PNGメタデータ埋め込み・画像保存
│   └── tests/
├── frontend/Sodalite/    # WinUI3フロントエンド
│   ├── MainWindow.xaml(.cs)    # バックエンド起動・ナビゲーション
│   ├── Views/GenerationPage    # プロンプト入力・生成・画像表示
│   ├── ViewModels/              # GenerationViewModel
│   └── Services/                # BackendProcessManager, BackendApiClient
├── docs/               # セットアップ記録等のドキュメント
├── skills/             # 開発規約 (winui3-app, python-coding)
├── run.ps1             # アプリ起動スクリプト(ルート)
└── CLAUDE.md
```

## セットアップ

### 前提条件

- Windows 10 22H2以降 / Windows 11
- .NET 9 SDK
- Python 3.13+ と [uv](https://docs.astral.sh/uv/)
- NVIDIA GPU (CUDA対応、VRAM 8GB以上推奨。CPUのみでも動作するが低速)

### 初回セットアップ

```powershell
# バックエンドの依存関係インストール
cd backend
uv sync
```

初回起動時、バックエンドが Hugging Face から画像生成モデル(既定: `stabilityai/sd-turbo`)を自動ダウンロードする。

### 起動

```powershell
# ルートで実行: バックエンド同期 + フロントエンドビルド + 起動を一括で行う
./run.ps1
```

アプリを起動すると、WinUI3プロセスが自動的にPythonバックエンドを子プロセスとして起動する(空きポートを動的に検出し、ヘルスチェック完了まで待機)。アプリを閉じるとバックエンドの子プロセスも確実に終了する。

### バックエンド単体での動作確認

```powershell
cd backend
./run.ps1
# 別ターミナルで
curl http://localhost:8000/api/v1/health
```

## 開発

- Pythonバックエンドのコーディング規約: [`skills/python-coding/SKILL.md`](skills/python-coding/SKILL.md)
- WinUI3フロントエンドのコーディング規約: [`skills/winui3-app/SKILL.md`](skills/winui3-app/SKILL.md)

```powershell
# バックエンドのlint/format/test
cd backend
uv run ruff check --fix .
uv run ruff format .
uv run pytest

# フロントエンドのビルド
cd frontend/Sodalite
dotnet build -c Debug
```

## 配布 (インストーラー)

エンドユーザー向けに NSIS 製インストーラー (`Sodalite-Setup-<version>.exe`) を生成できる。

### 前提条件

- **ビルド側**: .NET 9 SDK, [NSIS](https://nsis.sourceforge.io/) (`makensis`)
- **エンドユーザー側**: [uv](https://docs.astral.sh/uv/) がインストール済みであること。
  Python 3.13 は uv が初回セットアップ時に自動取得するため、別途の Python インストールは不要。

### インストーラーのビルド

```powershell
# ルートで実行 (publish → backend ステージング → NSIS コンパイルを一括)
./installer/build-installer.ps1 -Version 1.0.0
# または
make installer
```

`installer/dist/Sodalite-Setup-<version>.exe` が生成される。

### インストール後の初回起動

インストーラーは `%LOCALAPPDATA%\Programs\Sodalite\` に `app\` (フロントエンド) と `backend\`
(Python ソース) を配置する(管理者権限不要のユーザーインストール)。`.venv` は同梱されず、
**初回起動時にフロントエンドが `uv sync` を実行**して仮想環境を作成・依存パッケージを
インストールする (torch 等で数GB、数分〜数十分)。

- セットアップの成否は `%LOCALAPPDATA%\Sodalite\.venv-ready` に記録される
  (`uv.lock` のハッシュを保存)。成功した場合のみマーカーが書かれ、失敗した場合は
  **次回起動時に自動で再セットアップ**が走る。アプリ更新で依存が変わった場合も再同期される。
- uv が見つからない場合は、アプリ起動時に uv のインストールを促すメッセージが表示される。

## API概要

バックエンドは独自設計のREST API (`/api/v1/*`) を提供する。

| Method | Path | 説明 |
|---|---|---|
| GET | `/api/v1/health` | 起動確認・ロード中モデル・デバイス情報 |
| GET | `/api/v1/samplers` | 利用可能なサンプラー一覧 |
| POST | `/api/v1/generations/text-to-image` | txt2img生成 |

詳細は `backend/src/sodalite_backend/api/` を参照。

## 免責事項

- 本アプリは画像生成モデルを実行するためのツールであり、生成された画像の内容・利用・公開について作者は一切の責任を負いません。生成物の利用はすべて利用者自身の責任で行ってください。
- 生成画像には、使用したモデルのライセンス(モデルカード記載の利用規約・禁止事項等)が適用されます。商用利用・再配布・NSFW 表現の可否等はモデルごとに異なるため、利用前に必ず使用モデルのライセンス条項をご確認ください。
- 生成画像がプロンプトや学習データ由来で既存の著作物・商標・肖像等の権利に類似・抵触する可能性があります。生成物を公開・配布・商用利用する際は、著作権・商標権・パブリシティ権等の第三者の権利を侵害しないよう利用者自身でご確認・ご判断ください。
- アニメ・ゲーム・映画等の既存キャラクターやその他 IP (知的財産) を模倣・想起させる画像の生成についても、著作権・商標権の侵害となる可能性があります。とくに当該 IP の権利者が許諾していない商用利用・二次配布・性的表現等は権利侵害・規約違反となるおそれがあるため、利用者自身の責任でご判断ください。
- 実在の人物を描写する画像(なりすまし、誹謗中傷、性的な文脈での無断使用等)の生成・利用は、利用者の居住地の法令やプラットフォームの規約に違反する場合があります。法令・規約の遵守は利用者の責任とします。
- 法令で規制される内容の生成・所持・頒布は固く禁止します。
