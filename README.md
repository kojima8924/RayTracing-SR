# RayTracing-SR

メタボールレイマーチングによるデノイジング/超解像学習用データセット生成器

## 概要

Compute Shaderベースのレイマーチングで、ソフトシャドウ付きメタボールシーンをレンダリングします。
複数のSPP（samples per pixel）でレンダリングすることで、ノイズ除去や超解像の学習用データセットを生成できます。

## ファイル構成

```
RayTracing-SR/
├── main.py              # 既存のシンプルなレンダラー（参照用）
├── generate_dataset.py  # データセット生成CLI
└── README.md
```

## 必要環境

- Python 3.10+
- OpenGL 4.5対応GPU
- 依存ライブラリ: `pip install glfw PyOpenGL numpy opencv-python`

## 使用方法

```bash
python generate_dataset.py --out_dir dataset --n 1000 --low_spp 1 2 4 8 --high_spp 256
```

### 引数

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--out_dir` | (必須) | 出力ディレクトリ |
| `--n` | 1000 | 生成するシーン数 |
| `--width` | 256 | 画像幅 |
| `--height` | 256 | 画像高さ |
| `--low_spp` | 1 2 4 8 | 低SPPリスト（スペース区切り） |
| `--high_spp` | 256 | 高SPP（Ground Truth用） |
| `--seed` | 42 | 乱数シード |
| `--depth_far` | 5.0 | depth正規化の最大距離 |
| `--shininess` | 48.0 | スペキュラハイライトの鋭さ |
| `--spec_jitter` | 0.15 | スペキュラジッター強度 |
| `--aperture` | 0.0 | DOF絞り半径（0で無効） |
| `--focus_dist` | 1.0 | DOFフォーカス距離 |
| `--validate` | (フラグ) | 補助バッファの一貫性検証を実行 |

## 出力形式

```
dataset/
├── 000000/
│   ├── color_low_spp1.npy    # (H, W, 4) float32 - 低SPPカラー (linear)
│   ├── color_low_spp2.npy
│   ├── color_low_spp4.npy
│   ├── color_low_spp8.npy
│   ├── color_high.npy        # (H, W, 4) float32 - Ground Truth (linear)
│   ├── normal.npy            # (H, W, 4) float32 - 法線 [0,1]にマッピング
│   ├── depth.npy             # (H, W, 4) float32 - 深度 [0,1]にマッピング
│   ├── albedo.npy            # (H, W, 4) float32 - アルベド (linear)
│   ├── metadata.json         # シーンのメタデータ（後述）
│   ├── preview_low_spp1.png  # プレビュー画像 (gamma補正済み)
│   └── preview_high.png
├── 000001/
│   └── ...
```

### metadata.json

各シーンのメタデータをJSON形式で保存:

```json
{
  "spheres": [
    {
      "position": [x, y, z],
      "radius": 0.1,
      "color": [r, g, b],
      "ior": 1.5,
      "ratios": [ambient, diffuse, specular, refract]
    }
  ],
  "light": {
    "center": [x, y, z],
    "radius": 0.15,
    "intensity": 1.0
  },
  "background": {
    "type": 0,
    "color1": [r, g, b],
    "color2": [r, g, b],
    "phase": 0.0,
    "noise_scale": 0.1
  },
  "frame_seed": 12345,
  "render_params": {
    "depth_far": 5.0,
    "shininess": 48.0,
    "spec_jitter_scale": 0.15,
    "aperture_radius": 0.0,
    "focus_distance": 1.0
  }
}
```

### データ形式詳細

- **color_*.npy**: RGBA float32, 値域[0,1], **linear色空間**
- **normal.npy**: 法線ベクトル（xyz）を[0,1]にマッピング（(n+1)/2）
- **depth.npy**: カメラからヒット点までの距離を[0,1]に正規化（`length(hitPos-camOrigin)/depthFar`でクランプ）
- **albedo.npy**: シャドウなしの素の色、**linear色空間**
- **preview_*.png**: 見やすさのためgamma補正（1/2.2）を適用して保存

## 補助バッファの計算方法

**重要**: 補助バッファ（normal/depth/albedo）は**ジッター無しのprimary ray**から計算されます。

- SPPやseedに関わらず、同一シーンなら補助バッファは完全に一致します
- primary rayはピクセル中心（jitter = 0）を通るレイで、最初のヒット情報のみを記録
- これにより、デノイジングネットワークの入力として安定した補助情報を提供できます

### 一貫性検証

`--validate`フラグで補助バッファの一貫性を自動検証できます:

```bash
python generate_dataset.py --out_dir test --n 1 --validate
```

出力例:
```
=== 補助バッファ一貫性検証結果 ===
normal max_abs_diff: 0.0000000000
depth  max_abs_diff: 0.0000000000
albedo max_abs_diff: 0.0000000000
✓ 補助バッファはSPPに依存しません
```

## シーン生成

各シーンはランダムに以下が生成されます:

- **球**: 2〜7個のメタボール（位置・半径・色・マテリアルがランダム）
- **光源**: Area Light（ディスク光源）、上半球からランダム配置
- **背景**: グラデーション/チェッカー/Value Noiseの3種類からランダム

### マテリアルプリセット

球のマテリアルは以下のプリセットから重み付きランダム選択:

| プリセット | ambient | diffuse | specular | refract | 選択確率 |
|-----------|---------|---------|----------|---------|----------|
| diffuse   | 0.10    | 0.85    | 0.05     | 0.00    | 40%      |
| glossy    | 0.10    | 0.50    | 0.40     | 0.00    | 30%      |
| metallic  | 0.05    | 0.20    | 0.75     | 0.00    | 15%      |
| glassy    | 0.05    | 0.15    | 0.15     | 0.65    | 15%      |

同一シーンでは異なるSPPでも同じフレームシードが使われるため、
SPP増加による収束の過程を学習できます。

## ノイズ源

低SPPでノイズが発生する主な要因:

1. **サブピクセルジッター**: アンチエイリアシング用のランダムオフセット
2. **Area Light**: ディスク光源サンプリングによるソフトシャドウのモンテカルロノイズ
3. **スペキュラジッター**: 反射方向の微小ジッターによるハイライトノイズ
4. **DOF（オプション）**: 被写界深度によるボケのサンプリングノイズ
5. **背景Value Noise**: 低周波ノイズによる背景のザラつき

これらのノイズ源により、SPP=1〜256の間で明確なノイズ減少が観察できます。

## 技術詳細

### Area Light

従来の点光源ではなく、有限サイズのディスク光源を使用:
- 光源上のランダムな点をサンプリング
- 自然なペナンブラ（半影）を生成
- SPPが低いとソフトシャドウがノイズになる

### Value Noise

背景タイプ2（noise）では、Value Noiseを使用:
- 低周波のプロシージャルノイズ
- バイリニア補間による滑らかなグラデーション
- SPPに依存しないが、視覚的な複雑さを追加
