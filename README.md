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
| `--shadow_softness` | 0.15 | ソフトシャドウの柔らかさ |
| `--depth_far` | 5.0 | depth正規化の最大距離 |
| `--shininess` | 64.0 | スペキュラハイライトの鋭さ |

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
│   ├── preview_low_spp1.png  # プレビュー画像 (gamma補正済み)
│   └── preview_high.png
├── 000001/
│   └── ...
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

## シーン生成

各シーンはランダムに以下が生成されます:

- **球**: 2〜6個のメタボール（位置・半径・色・マテリアルがランダム）
- **光源**: 上半球からランダム方向、強度0.8〜1.2
- **背景**: グラデーション/チェッカー/ノイズの3種類からランダム

同一シーンでは異なるSPPでも同じフレームシードが使われるため、
SPP増加による収束の過程を学習できます。

## ノイズ源

低SPPでノイズが発生する主な要因:

1. **サブピクセルジッター**: アンチエイリアシング用のランダムオフセット
2. **ソフトシャドウ**: 光源方向の微小ブレによるペナンブラ表現
3. **スペキュラハイライト**: 反射方向の微小ジッターによるハイライトノイズ
4. **背景ノイズ**: 背景タイプに応じた低周波/エッジノイズ

これらのノイズ源により、SPP=1〜256の間で明確なノイズ減少が観察できます。
