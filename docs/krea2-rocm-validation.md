# Krea 2 Turbo: ROCm最適化と実機検証

検証日: 2026-10-03 / Sodalite 0.21.0

## 結果

AMD Radeon AI PRO R9700で実際に画像を生成した。1024×1024、8ステップ、同じプロンプトを連続使用した場合、既存のscaled FP8重みを保持する方式は2枚目を **21.412秒** で生成した。同じファイルをすべてBF16へ展開してGPUへブロック転送する参照方式は **58.853秒** だった。約 **2.75倍** の高速化で、seed 42・43の出力はそれぞれ **全ピクセル完全一致** した。

この比較では、参照方式にも修正済みテキストエンコーダ、ROCm attention、パディング除去、プロンプトキャッシュを適用した。FP8重みの保持とGPU上での展開による差を比較している。元の処理全体との1024サイズの完走比較ではない。

## 環境と条件

- GPU: AMD Radeon AI PRO R9700、物理VRAM 31.859 GiB
- RAM: 128 GiB / Windows
- PyTorch: `2.9.1+rocm7.2.1` / HIP: `7.2.53211-158bd99533`
- Diffusers: `0.39.0` / Transformers: `4.57.6`
- FP8ファイル: `C:/Users/hikari/01_SDXL_MODEL/krea2Turbo_v10_2951489.safetensors`
- BF16ファイル: `C:/Users/hikari/01_SDXL_MODEL/krea2Turbo_v10_bf16.safetensors`
- Euler、CFG 0、negative promptなし、8ステップ、seed 42・43、LoRAなし
- prompt: `A red fox sitting in a meadow of wildflowers, morning sunlight, detailed photograph`
- `HF_HUB_OFFLINE=1`でキャッシュ済み補助モデルを使用。ネットワーク取得時間は含まない。
- `MIOPEN_FIND_MODE=FAST` / `MIOPEN_FIND_ENFORCE=NONE`。アプリのROCmランチャーと同条件。

ロード時間はパイプライン構築を測り、Python起動・import時間を含まない。1枚目にはテキストエンコード、重みの初回GPU転送、GPU初期化などが含まれる。2枚目はプロンプトキャッシュを使用する。画像ごとの時間にはVAEデコードを含む。GPU同期を入れて計測した。

## 実測

| 実行方式 | 解像度 | ロード | 1枚目 | 2枚目 | peak allocated | peak reserved |
|---|---:|---:|---:|---:|---:|---:|
| 同一FP8ファイルを全BF16展開しGPUブロック転送（参照） | 1024² | 21.391 s | 77.815 s | 58.853 s | 8.758 GiB | 18.473 GiB |
| FP8保持・必要な層をGPUでBF16展開（採用） | 1024² | 2.109 s | 54.173 s | 21.412 s | 21.321 GiB | 28.717 GiB |
| BF16ファイル・GPUブロック転送（採用） | 1024² | 2.008 s | 119.074 s | 59.785 s | 8.758 GiB | 18.473 GiB |
| 元のattention・モデル単位offload、テキストエンコーダのみ修正 | 512² | 0.775 s | 95.596 s | 97.526 s | 25.181 GiB | 25.535 GiB |
| BF16ファイル・GPUブロック転送（採用） | 512² | 1.580 s | 88.704 s | 47.412 s | 8.618 GiB | 8.682 GiB |

BF16の1枚目はメモリマップした大容量重みの初回転送を含み、同じ重みをCPUへ展開済みのFP8参照とはロード後の準備状態が異なる。ロード時間だけで初回処理全体の速さを判断しない。

同一BF16ファイルで完走比較できた512サイズでは、2枚目は97.526秒から47.412秒へ約2.06倍高速化した。最終実装は1024サイズでも共有メモリへの過剰な退避を避ける設定を優先する。512専用のGPU常駐実験ではさらに速かったが、このGPUでは1024で極端に遅くなるため採用していない。

FP8保持方式はメモリを多く使う代わりに、各ステップの大きなCPU→GPU転送を避ける。参照方式より低メモリとは主張しない。どちらも推論演算はGPUで実行する。ROCmのPyTorch APIではGPUのdevice表記が `cuda:0` になる。

1プロンプト・各方式2枚の実測であり、統計的な平均値や他のGPUの保証値ではない。ディスクキャッシュ、GPUの初回処理、他のアプリのVRAM使用により時間は変わる。

## 修正内容

### テキストエンコーダを正しい重みでロード

この環境のTransformersでは、conditional-generation形式のQwen3-VLチェックポイントを直接 `Qwen3VLModel` に読み込むと、重み名が一致せず、多数の層がランダム初期化されていた。保存時のクラス `Qwen3VLForConditionalGeneration` で読み込んでから `.model` を取り出すよう修正した。不足・形状不一致・ロードエラーがある場合は生成へ進まない。

不正なテキストエンコーダの出力は比較基準に使わない。上の旧方式のベンチマークにも、この修正を適用した。

### ROCmのfused attentionを使用

Kreaのquery 48 heads / key-value 12 headsに対し、このROCm環境ではhead数が異なるSDPAが低速なmath実装へ落ちていた。KVを同じGQA対応関係のまま展開し、fused attentionで処理する。ロータリー位置埋め込み、QK正規化、gate、出力射影を維持する。

全プロンプトで無効なパディング列のみ除去する。途中のパディングに続く有効なsuffixは保持する。全要素が有効なマスクを省略することでflash kernelを使用できる。部分マスクが必要な場合は保持する。マスク・ロータリー埋め込みを含む等価性と、math backendを禁止したROCm上でのfused実行をテストする。

fused kernelは演算順序が異なるため、旧math方式との比較でビット単位の一致は保証しない。最終BF16実装の512サイズ比較ではPSNR 28.0 / 37.5 dB、画素の平均絶対差4.352 / 1.578（0〜255）だった。一方、上のFP8保持対全BF16展開の比較は同じfused方式を使い、両seedとも完全一致した。PSNRのみで品質保証は行わず、生成画像も目視確認した。

### 保存形式に応じたGPUメモリ管理

- 既存のscaled FP8: 主Transformer blockのLinear重みをFP8のまま保存・転送する。演算時だけGPU上でFP32スケーリング後BF16へ変換する。元のローダーと同じ変換式を使用し、追加の量子化は行わない。LoRAの重みはBF16で維持する。
- BF16: BF16のまま処理する。空きVRAMが「登録重みの容量 + 12 GiB」に足りない場合、2 blockずつGPUへ転送する。この32 GiB環境ではBF16にこの方式を適用する。
- 十分な空きVRAMがある場合はTransformerをGPUへ保持する。VAEデコード前には必要なworkspaceを確保し、必要なblockだけ退避する。GPU OOM時は同じlatentsのデコードをGPUで再試行する。
- CPUは非稼働の重みの保管と乱数生成などを担当する。テキストエンコーダ・Transformer・VAEのモデル演算はROCmで実行する。
- 同一プロンプトのエンコード結果を1件キャッシュする。プロンプトや最大長が変わると再計算する。

BF16モデルをこのGPUへ全部常駐させる実験では、VRAM予約が物理容量を超えて共有メモリへ退避し、2枚目が約224〜288秒まで遅くなった。例外が出るまで待つ方式では防げなかったため、最終実装では上記の12 GiB余裕とブロック転送を採用した。

### 生成設定と同時実行

モデルの実クラスをhealth APIの `model_family` として返し、ファイル名を変更してもKrea設定を適用する。両UIでCFG・negative prompt・samplerを無効化する。実際のCFG 0・Euler・negativeなし・16の倍数へ切り上げた寸法をPNGへ記録する。複数枚生成のseedは各画像の実値を保存する。

モデル切り替えと生成を同一ロックで直列化し、異なるジョブが同じGPU pipelineを同時操作しないようにした。Kreaのキャンセルはステップ間で反映し、未完成画像をデコード・保存しない。LoRA適用で失敗した場合も後処理を行う。

## 再現方法

既存のROCm仮想環境を使い、`backend` ディレクトリから実行する。`--no-sync` は導入済みのROCm PyTorchを維持するために指定する。

```powershell
$env:HF_HUB_OFFLINE='1'
$env:MIOPEN_FIND_MODE='FAST'
$env:MIOPEN_FIND_ENFORCE='NONE'

# 採用方式: 既存FP8の保持
uv --cache-dir .uv-cache run --no-sync python benchmark_krea2.py `
  --model C:/Users/hikari/01_SDXL_MODEL/krea2Turbo_v10_2951489.safetensors `
  --output outputs/krea2-final-fp8-1024

# 同一重みの参照: FP8を全BF16へ展開
uv --cache-dir .uv-cache run --no-sync python benchmark_krea2.py `
  --model C:/Users/hikari/01_SDXL_MODEL/krea2Turbo_v10_2951489.safetensors `
  --expand-fp8 --output outputs/krea2-reference-fp8-1024

# BF16ファイルの採用方式
uv --cache-dir .uv-cache run --no-sync python benchmark_krea2.py `
  --model C:/Users/hikari/01_SDXL_MODEL/krea2Turbo_v10_bf16.safetensors `
  --output outputs/krea2-final-bf16-1024

# 旧attention/offloadとの比較。テキストエンコーダ修正は有効。
uv --cache-dir .uv-cache run --no-sync python benchmark_krea2.py `
  --model C:/Users/hikari/01_SDXL_MODEL/krea2Turbo_v10_bf16.safetensors `
  --baseline --size 512 --output outputs/krea2-baseline-512

# 自動テスト
uv --cache-dir .uv-cache run --no-sync pytest -q
```

FP8保持と全BF16展開のPNGを再比較する場合:

```powershell
uv --cache-dir .uv-cache run --no-sync python -c "from PIL import Image; import numpy as np; from pathlib import Path; root=Path('outputs'); assert all(np.array_equal(np.asarray(Image.open(root/'krea2-final-fp8-1024'/f'seed-{s}.png')), np.asarray(Image.open(root/'krea2-reference-fp8-1024'/f'seed-{s}.png'))) for s in (42,43)); print('Both images are pixel-identical')"
```

実機試験にはCUDA/ROCm GPUを必須とし、CPU生成を代用しない。PNGと `measurements.json` を指定先に保存する。画像が空・単色なら試験を失敗させる。計測時のGPUを確認するhookは、ブロック転送後の重みの置き場所ではなく、演算結果のtensor deviceを記録する。

## 検証記録

- `pytest`: **91 passed、skip 0**、14.17秒。ROCm GPUを必要とする4テストも実行した。attentionの数学的等価性・fused実行、パディング除去とTransformer出力、GPU常駐とブロック転送、FP8保持・BF16展開との一致、GPU LoRAとadapterのBF16維持、プロンプトキャッシュ、VAEの同一latentsでのOOM再試行、キャンセル、生成中のモデル切り替え待機、health API、PNG設定・seed保存を含む。
- 3件のwarning: 既存のStarlette/httpx非推奨、テストfixtureのRMSNorm dtypeに関するfused正規化の警告、ROCmのAOTriton attention使用の通知。失敗はない。
- `ruff check src tests benchmark_krea2.py`: 合格。
- 変更したPythonファイル15件の `ruff format --check`: 合格。未変更の既存ファイルにformatter差分があるため、全ファイル整形は行っていない。
- `dotnet build frontend/Sodalite/Sodalite.csproj --no-restore -c Debug`: **0 warning / 0 error**、バージョン0.21.0。
- Web UI: `node --check` 合格。実際の `applyModelDefaults` / `setGenerating` をNode VMで実行し、改名したKreaモデル、誤解を招く名前のSDモデル、生成完了後の入力無効状態、SDへ戻した場合の入力復帰を確認した。
- 最終FP8・BF16の1024画像を表示して目視確認し、画像サイズ・非単色も確認した。実機試験はAPI・GUI経由ではなく、アプリと同じloader / PipelineManagerを使うベンチマークCLIで行った。GUI全体の自動操作試験は実施していない。

計測ファイルは `backend/outputs/` 内に保存し、Git管理対象外とする。

- `krea2-final-fp8-1024/measurements.json`: 採用FP8方式。encoder 1回、Transformer 16回、VAE decoder 2回がすべて `cuda:0`。
- `krea2-final-fp8-1024/comparison.json`: seed 42・43の完全一致確認。
- `krea2-final-bf16-1024/measurements.json`: 採用BF16方式。encoder 1回、Transformer 16回、VAE decoder 2回の出力がすべて `cuda:0`。
- `krea2-reference-fp8-1024/measurements.json`: 同一重みの全BF16展開参照。この実行は旧計測hookを使用し、Transformer欄の `cpu` はoffload後の重み保存先を表す。CPU推論を表す値ではない。新しいhookと最終BF16実機試験でGPU出力を確認する。
- `krea2-baseline-512/measurements.json`: 旧attention/offload方式の完走記録。
- `krea2-final-bf16-512/measurements.json` / `comparison.json`: 最終BF16方式の完走記録と、旧方式との画素比較。

旧attentionでの1024試験は、最初のstep約124秒、続くstep約80秒で打ち切った。完走していないため、旧方式の1024の総所要時間や高速化倍率は報告しない。
