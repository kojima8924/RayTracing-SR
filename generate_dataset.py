"""
デノイジング/超解像学習用データセット生成器
メタボールレイマーチングによるシーンを複数SPPでレンダリングし、
color/normal/depth/albedoの補助バッファとともに保存する。
"""

import argparse
import os
import sys

import cv2
import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

# =============================================================================
# Compute Shader
# =============================================================================
COMPUTE_SHADER = """
#version 450
precision highp float;
layout(local_size_x = 16, local_size_y = 16) in;

// 出力バッファ（4つ）
layout(rgba16f, binding = 0) uniform image2D img_color;
layout(rgba16f, binding = 1) uniform image2D img_normal;
layout(rgba16f, binding = 2) uniform image2D img_depth;
layout(rgba16f, binding = 3) uniform image2D img_albedo;

struct Sphere {
    vec4 posRadius; // xyz: 位置, w: 半径
    vec3 color;     // rgb: 色
    float ior;      // 屈折率
    vec4 ratios;    // x: 環境, y: 拡散, z: 反射, w: 屈折
};
layout(std430, binding=1) buffer Spheres { Sphere spheres[]; };

uniform int sphereCount;
uniform vec4 light;        // xyz: 方向, w: 強さ
uniform uint frameSeed;    // フレームごとのシード
uniform int spp;           // サンプル数
uniform float shadowSoftness; // ソフトシャドウの柔らかさ

// 背景パラメータ
uniform int bgType;        // 0:gradient, 1:checker, 2:noise
uniform vec3 bgColor1;
uniform vec3 bgColor2;
uniform float bgPhase;

const int MAX_SPHERES = 16;
const int MAX_DEPTH = 3;
const int MAX_STACK = 16;
const float HIT_THRESHOLD = 1.0e-4;
const float MAX_DISTANCE = 100.0;
const int MAX_MARCH_STEPS = 100;

// =============================================================================
// Wang Hash RNG
// =============================================================================
uint wang_hash(uint seed) {
    seed = (seed ^ 61u) ^ (seed >> 16u);
    seed *= 9u;
    seed = seed ^ (seed >> 4u);
    seed *= 0x27d4eb2du;
    seed = seed ^ (seed >> 15u);
    return seed;
}

float rand(inout uint state) {
    state = wang_hash(state);
    return float(state) / 4294967296.0;
}

uint initRNG(ivec2 pixel, int sampleIndex) {
    return wang_hash(uint(pixel.x) + wang_hash(uint(pixel.y) + wang_hash(frameSeed + uint(sampleIndex))));
}

// =============================================================================
// レイタスク構造体
// =============================================================================
struct RayTask {
    vec3 origin;
    vec3 direction;
    float weight;
    int depth;
    bool inside;
};

// =============================================================================
// 距離場の計算
// =============================================================================
void fieldAt(vec3 p, out float kMax, out float weightSum, out float weights[MAX_SPHERES]) {
    float k_ref = 8.0;
    kMax = 0.0;
    weightSum = 0.0;
    for (int j = 0; j < sphereCount; j++) {
        Sphere sphere = spheres[j];
        float r = max(sphere.posRadius.w, 1.0e-3);
        float kj = k_ref / r;
        float d = length(p - sphere.posRadius.xyz) - r;
        float w = exp(-kj * d);
        weights[j] = w;
        weightSum += w;
        kMax = max(kMax, kj);
    }
}

float distanceAt(vec3 p) {
    float k_ref = 8.0;
    float kMax = 0.0;
    float weightSum = 0.0;
    for (int j = 0; j < sphereCount; j++) {
        Sphere sphere = spheres[j];
        float r = max(sphere.posRadius.w, 1.0e-3);
        float kj = k_ref / r;
        float d = length(p - sphere.posRadius.xyz) - r;
        weightSum += exp(-kj * d);
        kMax = max(kMax, kj);
    }
    weightSum = max(weightSum, 1.0e-6);
    return -log(weightSum) / max(kMax, 1.0e-6);
}

// =============================================================================
// 法線計算（distanceAt差分）
// =============================================================================
vec3 calcNormal(vec3 p) {
    const float eps = 1.0e-4;
    return normalize(vec3(
        distanceAt(p + vec3(eps, 0.0, 0.0)) - distanceAt(p - vec3(eps, 0.0, 0.0)),
        distanceAt(p + vec3(0.0, eps, 0.0)) - distanceAt(p - vec3(0.0, eps, 0.0)),
        distanceAt(p + vec3(0.0, 0.0, eps)) - distanceAt(p - vec3(0.0, 0.0, eps))
    ));
}

// =============================================================================
// ソフトシャドウ
// =============================================================================
float calcSoftShadow(vec3 p, vec3 lightDir, inout uint rng) {
    // 光源方向のジッター
    vec3 tangent = normalize(cross(lightDir, vec3(0.0, 1.0, 0.001)));
    vec3 bitangent = cross(lightDir, tangent);
    vec3 jitter = tangent * (rand(rng) - 0.5) * shadowSoftness
                + bitangent * (rand(rng) - 0.5) * shadowSoftness;
    vec3 jitteredDir = normalize(lightDir + jitter);

    vec3 pos = p + jitteredDir * 0.01;
    float distance = 0.0;
    for (int i = 0; i < 50; i++) {
        float d = distanceAt(pos);
        if (d < HIT_THRESHOLD) return 0.2;
        if (distance > MAX_DISTANCE) break;
        distance += d;
        pos += jitteredDir * d;
    }
    return 1.0;
}

// =============================================================================
// 地面との交差判定
// =============================================================================
bool hitGround(vec3 origin, vec3 ray, out float t, out vec3 groundColor) {
    if (ray.y >= 0.0) return false;
    t = -origin.y / ray.y;
    vec3 groundPos = origin + ray * t;
    if (groundPos.x < -1.0 || groundPos.x > 1.0 || groundPos.z < -2.0 || groundPos.z > 0.0) {
        return false;
    }
    int checkX = int(floor(groundPos.x * 8.0));
    int checkZ = int(floor(groundPos.z * 8.0));
    int check = (checkX + checkZ) & 1;
    groundColor = (check == 0) ? vec3(0.3, 0.3, 0.3) : vec3(0.9, 0.9, 0.9);
    return true;
}

// =============================================================================
// Procedural背景
// =============================================================================
vec3 proceduralBg(vec3 ray, inout uint rng) {
    if (bgType == 0) {
        // グラデーション
        float t = ray.y * 0.5 + 0.5 + sin(bgPhase) * 0.1;
        return mix(bgColor1, bgColor2, clamp(t, 0.0, 1.0));
    } else if (bgType == 1) {
        // 球面チェッカー
        float u = atan(ray.z, ray.x) / 3.14159265 * 4.0 + bgPhase;
        float v = asin(ray.y) / 1.5707963 * 4.0;
        int cu = int(floor(u));
        int cv = int(floor(v));
        if (((cu + cv) & 1) == 0) {
            return bgColor1;
        } else {
            return bgColor2;
        }
    } else {
        // ノイズ風
        float noise = rand(rng) * 0.1;
        float t = ray.y * 0.5 + 0.5;
        return mix(bgColor1, bgColor2, clamp(t + noise, 0.0, 1.0));
    }
}

// =============================================================================
// メイン
// =============================================================================
void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(img_color);

    if (pixel.x >= size.x || pixel.y >= size.y) return;

    // カメラ設定
    float screen_depth = 1.2071067811865475;
    vec3 camOrigin = vec3(0.0, 0.2, 0.0);
    vec3 lookAt = vec3(0.0, 0.0, -1.0);
    vec3 upHint = vec3(0.0, 1.0, 0.0);
    vec3 forward = normalize(lookAt - camOrigin);
    vec3 right = normalize(cross(forward, upHint));
    vec3 up = normalize(cross(right, forward));

    // 補助バッファ用（プライマリヒット情報）
    vec3 primaryNormal = vec3(0.0);
    float primaryDepth = 1.0;
    vec3 primaryAlbedo = vec3(0.0);
    bool primaryHitRecorded = false;

    // SPPループで色を積算
    vec3 accumulatedColor = vec3(0.0);

    for (int s = 0; s < spp; s++) {
        uint rng = initRNG(pixel, s);

        // サブピクセルジッター
        vec2 jitter = vec2(rand(rng), rand(rng)) - 0.5;
        vec2 uv = (vec2(pixel) + 0.5 + jitter) / vec2(size) - 0.5;
        vec3 initialRay = normalize(uv.x * right - uv.y * up + screen_depth * forward);

        // スタック初期化
        RayTask stack[MAX_STACK];
        int stackPtr = 0;
        vec3 sampleColor = vec3(0.0);

        stack[stackPtr].origin = camOrigin;
        stack[stackPtr].direction = initialRay;
        stack[stackPtr].weight = 1.0;
        stack[stackPtr].depth = 0;
        stack[stackPtr].inside = false;
        stackPtr++;

        bool isFirstHit = true;

        // メインループ
        while (stackPtr > 0) {
            stackPtr--;
            RayTask task = stack[stackPtr];

            if (task.depth > MAX_DEPTH) continue;
            if (task.weight < 0.01) continue;

            vec3 pos = task.origin;
            vec3 ray = task.direction;
            float distance = 0.0;
            bool hit = false;

            // レイマーチング
            for (int i = 0; i < MAX_MARCH_STEPS; i++) {
                if (distance > MAX_DISTANCE) break;
                if (sphereCount == 0) break;

                float kMax;
                float weightSum;
                float weights[MAX_SPHERES];
                fieldAt(pos, kMax, weightSum, weights);
                weightSum = max(weightSum, 1.0e-6);
                float minDist = -log(weightSum) / max(kMax, 1.0e-6);

                if (minDist < HIT_THRESHOLD) {
                    // マテリアル情報を集約
                    vec4 ratios = vec4(0.0);
                    vec3 color = vec3(0.0);
                    float ior = 0.0;
                    for (int j = 0; j < sphereCount; j++) {
                        Sphere sphere = spheres[j];
                        ratios += sphere.ratios * weights[j];
                        color += sphere.color * weights[j];
                        ior += sphere.ior * weights[j];
                    }
                    ratios /= weightSum;
                    color /= weightSum;
                    ior /= weightSum;

                    vec3 normal = calcNormal(pos);
                    if (task.inside) normal = -normal;

                    // プライマリヒット情報を記録（最初のサンプルの最初のヒットのみ）
                    if (s == 0 && isFirstHit && task.depth == 0 && !primaryHitRecorded) {
                        primaryNormal = normal * 0.5 + 0.5; // [-1,1] -> [0,1]
                        primaryDepth = clamp(distance / 5.0, 0.0, 1.0); // 正規化
                        primaryAlbedo = color;
                        primaryHitRecorded = true;
                    }
                    isFirstHit = false;

                    vec3 lightDir = -light.xyz;
                    float shadow = calcSoftShadow(pos, lightDir, rng);

                    // 環境光
                    sampleColor += task.weight * ratios.x * color * light.w;

                    // 拡散光
                    float diff = max(dot(normal, lightDir), 0.0);
                    sampleColor += task.weight * ratios.y * color * diff * light.w * shadow;

                    // 反射
                    if (ratios.z > 0.01 && stackPtr < MAX_STACK) {
                        vec3 reflectDir = reflect(ray, normal);
                        stack[stackPtr].origin = pos + normal * 0.01;
                        stack[stackPtr].direction = reflectDir;
                        stack[stackPtr].weight = task.weight * ratios.z;
                        stack[stackPtr].depth = task.depth + 1;
                        stack[stackPtr].inside = false;
                        stackPtr++;
                    }

                    // 屈折
                    if (ratios.w > 0.01 && stackPtr < MAX_STACK) {
                        float eta = task.inside ? ior : (1.0 / ior);
                        vec3 refractDir = refract(ray, normal, eta);
                        if (length(refractDir) > 0.0) {
                            stack[stackPtr].origin = pos - normal * 0.01;
                            stack[stackPtr].direction = refractDir;
                            stack[stackPtr].weight = task.weight * ratios.w;
                            stack[stackPtr].depth = task.depth + 1;
                            stack[stackPtr].inside = !task.inside;
                            stackPtr++;
                        }
                    }

                    hit = true;
                    break;
                }

                distance += minDist;
                pos += ray * minDist;
            }

            // ヒットしなかった場合
            if (!hit) {
                float t;
                vec3 groundColor;
                if (hitGround(task.origin, ray, t, groundColor)) {
                    vec3 groundPos = task.origin + ray * t;
                    vec3 groundNormal = vec3(0.0, 1.0, 0.0);
                    vec3 lightDir = -light.xyz;
                    float shadow = calcSoftShadow(groundPos, lightDir, rng);
                    float diff = max(dot(groundNormal, lightDir), 0.0);
                    sampleColor += task.weight * groundColor * (0.3 + 0.7 * diff * shadow);

                    // プライマリヒット（地面）
                    if (s == 0 && task.depth == 0 && !primaryHitRecorded) {
                        primaryNormal = groundNormal * 0.5 + 0.5;
                        primaryDepth = clamp(t / 5.0, 0.0, 1.0);
                        primaryAlbedo = groundColor;
                        primaryHitRecorded = true;
                    }
                } else {
                    sampleColor += task.weight * proceduralBg(ray, rng);

                    // プライマリ背景
                    if (s == 0 && task.depth == 0 && !primaryHitRecorded) {
                        primaryNormal = vec3(0.5, 0.5, 1.0); // 背景向き
                        primaryDepth = 1.0;
                        primaryAlbedo = proceduralBg(ray, rng);
                        primaryHitRecorded = true;
                    }
                }
            }
        }

        accumulatedColor += sampleColor;
    }

    // 平均化
    vec3 finalColor = accumulatedColor / float(spp);
    finalColor = clamp(finalColor, 0.0, 1.0);

    // 出力
    imageStore(img_color, pixel, vec4(finalColor, 1.0));
    imageStore(img_normal, pixel, vec4(primaryNormal, 1.0));
    imageStore(img_depth, pixel, vec4(vec3(primaryDepth), 1.0));
    imageStore(img_albedo, pixel, vec4(primaryAlbedo, 1.0));
}
"""


def create_texture_rgba16f(width: int, height: int) -> int:
    """GL_RGBA16Fテクスチャを作成"""
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA16F, width, height)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    return tex


def read_texture_float32(tex: int, width: int, height: int) -> np.ndarray:
    """テクスチャからfloat32データを読み出し"""
    glBindTexture(GL_TEXTURE_2D, tex)
    data = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT)
    return np.frombuffer(data, dtype=np.float32).reshape(height, width, 4)


def generate_random_spheres(rng: np.random.Generator, n: int = None) -> np.ndarray:
    """ランダムな球データを生成"""
    if n is None:
        n = rng.integers(2, 7)

    spheres = []
    for _ in range(n):
        # 位置: x[-0.3, 0.3], y[-0.1, 0.2], z[-1.2, -0.6]
        x = rng.uniform(-0.3, 0.3)
        y = rng.uniform(-0.1, 0.2)
        z = rng.uniform(-1.2, -0.6)
        # 半径: [0.03, 0.15]
        r = rng.uniform(0.03, 0.15)
        # 色: [0.2, 1.0]
        color = rng.uniform(0.2, 1.0, size=3)
        # 屈折率
        ior = rng.uniform(1.3, 1.7)
        # ratios: ランダム後正規化
        ratios = rng.uniform(0.0, 1.0, size=4)
        ratios[0] = rng.uniform(0.05, 0.2)  # 環境光は小さめ
        ratios = ratios / ratios.sum()

        spheres.append([
            (x, y, z, r),
            (*color, ior),
            tuple(ratios)
        ])

    return np.array(spheres, dtype=np.float32)


def generate_random_light(rng: np.random.Generator) -> tuple:
    """ランダムな光源方向を生成（上半球）"""
    # 球面座標でランダム
    theta = rng.uniform(0, 2 * np.pi)
    phi = rng.uniform(0.1, np.pi / 2)  # 上半球
    x = np.sin(phi) * np.cos(theta)
    y = -np.cos(phi)  # 下向き成分（光が上から）
    z = np.sin(phi) * np.sin(theta)
    # 正規化
    direction = np.array([x, y, z])
    direction = direction / np.linalg.norm(direction)
    # 強度
    intensity = rng.uniform(0.8, 1.2)
    return direction, intensity


def generate_random_background(rng: np.random.Generator) -> dict:
    """ランダムな背景パラメータを生成"""
    bg_type = rng.integers(0, 3)
    color1 = rng.uniform(0.3, 1.0, size=3)
    color2 = rng.uniform(0.3, 1.0, size=3)
    phase = rng.uniform(0, 2 * np.pi)
    return {
        'type': bg_type,
        'color1': color1,
        'color2': color2,
        'phase': phase
    }


def render_scene(
    program: int,
    textures: dict,
    spheres: np.ndarray,
    ssbo: int,
    light_dir: np.ndarray,
    light_intensity: float,
    bg_params: dict,
    width: int,
    height: int,
    spp: int,
    frame_seed: int,
    shadow_softness: float = 0.1
):
    """シーンをレンダリング"""
    # SSBOにデータをアップロード
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)
    glBufferData(GL_SHADER_STORAGE_BUFFER, spheres.nbytes, spheres, GL_STATIC_DRAW)

    glUseProgram(program)

    # Uniform設定
    glUniform1i(glGetUniformLocation(program, "sphereCount"), spheres.shape[0])
    glUniform4f(
        glGetUniformLocation(program, "light"),
        light_dir[0], light_dir[1], light_dir[2], light_intensity
    )
    glUniform1ui(glGetUniformLocation(program, "frameSeed"), frame_seed)
    glUniform1i(glGetUniformLocation(program, "spp"), spp)
    glUniform1f(glGetUniformLocation(program, "shadowSoftness"), shadow_softness)

    # 背景パラメータ
    glUniform1i(glGetUniformLocation(program, "bgType"), bg_params['type'])
    glUniform3f(
        glGetUniformLocation(program, "bgColor1"),
        *bg_params['color1']
    )
    glUniform3f(
        glGetUniformLocation(program, "bgColor2"),
        *bg_params['color2']
    )
    glUniform1f(glGetUniformLocation(program, "bgPhase"), bg_params['phase'])

    # テクスチャをバインド
    glBindImageTexture(0, textures['color'], 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA16F)
    glBindImageTexture(1, textures['normal'], 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA16F)
    glBindImageTexture(2, textures['depth'], 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA16F)
    glBindImageTexture(3, textures['albedo'], 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA16F)

    # 実行
    glDispatchCompute((width + 15) // 16, (height + 15) // 16, 1)
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)


def save_preview_png(data: np.ndarray, path: str):
    """float32データをPNGプレビューとして保存"""
    # [0,1] -> [0,255]
    img = (np.clip(data[:, :, :3], 0, 1) * 255).astype(np.uint8)
    # RGB -> BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def main():
    parser = argparse.ArgumentParser(description='デノイジング/超解像学習用データセット生成')
    parser.add_argument('--out_dir', type=str, required=True, help='出力ディレクトリ')
    parser.add_argument('--n', type=int, default=1000, help='生成するシーン数')
    parser.add_argument('--width', type=int, default=256, help='画像幅')
    parser.add_argument('--height', type=int, default=256, help='画像高さ')
    parser.add_argument('--low_spp', type=int, nargs='+', default=[1, 2, 4, 8], help='低SPPリスト')
    parser.add_argument('--high_spp', type=int, default=256, help='高SPP（Ground Truth）')
    parser.add_argument('--seed', type=int, default=42, help='乱数シード')
    parser.add_argument('--shadow_softness', type=float, default=0.1, help='ソフトシャドウの柔らかさ')
    args = parser.parse_args()

    # 出力ディレクトリ作成
    os.makedirs(args.out_dir, exist_ok=True)

    # 乱数生成器
    rng = np.random.default_rng(args.seed)

    # GLFW初期化
    if not glfw.init():
        print("GLFWの初期化に失敗しました")
        sys.exit(1)

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 5)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)

    window = glfw.create_window(1, 1, "", None, None)
    if not window:
        glfw.terminate()
        print("OpenGLコンテキストの作成に失敗しました")
        sys.exit(1)

    glfw.make_context_current(window)

    # シェーダーコンパイル
    try:
        program = compileProgram(
            compileShader(COMPUTE_SHADER, GL_COMPUTE_SHADER),
        )
    except RuntimeError as e:
        print(f"シェーダーコンパイルエラー:\n{e}")
        glfw.terminate()
        sys.exit(1)

    # テクスチャ作成
    textures = {
        'color': create_texture_rgba16f(args.width, args.height),
        'normal': create_texture_rgba16f(args.width, args.height),
        'depth': create_texture_rgba16f(args.width, args.height),
        'albedo': create_texture_rgba16f(args.width, args.height),
    }

    # SSBO作成
    ssbo = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo)

    print(f"データセット生成開始: {args.n}シーン, {args.width}x{args.height}")
    print(f"低SPP: {args.low_spp}, 高SPP: {args.high_spp}")

    for scene_idx in range(args.n):
        scene_dir = os.path.join(args.out_dir, f"{scene_idx:06d}")
        os.makedirs(scene_dir, exist_ok=True)

        # シーンパラメータ生成
        spheres = generate_random_spheres(rng)
        light_dir, light_intensity = generate_random_light(rng)
        bg_params = generate_random_background(rng)

        # フレームシードはシーンごとに固定（同一シーンで異なるSPPを比較可能に）
        frame_seed = rng.integers(0, 2**31)

        # 高SPPレンダリング（Ground Truth）
        render_scene(
            program, textures, spheres, ssbo,
            light_dir, light_intensity, bg_params,
            args.width, args.height,
            args.high_spp, frame_seed, args.shadow_softness
        )

        # 読み出し・保存
        color_high = read_texture_float32(textures['color'], args.width, args.height)
        normal = read_texture_float32(textures['normal'], args.width, args.height)
        depth = read_texture_float32(textures['depth'], args.width, args.height)
        albedo = read_texture_float32(textures['albedo'], args.width, args.height)

        np.save(os.path.join(scene_dir, "color_high.npy"), color_high)
        np.save(os.path.join(scene_dir, "normal.npy"), normal)
        np.save(os.path.join(scene_dir, "depth.npy"), depth)
        np.save(os.path.join(scene_dir, "albedo.npy"), albedo)
        save_preview_png(color_high, os.path.join(scene_dir, "preview_high.png"))

        # 低SPPレンダリング
        for low_spp in args.low_spp:
            render_scene(
                program, textures, spheres, ssbo,
                light_dir, light_intensity, bg_params,
                args.width, args.height,
                low_spp, frame_seed, args.shadow_softness
            )

            color_low = read_texture_float32(textures['color'], args.width, args.height)
            np.save(os.path.join(scene_dir, f"color_low_spp{low_spp}.npy"), color_low)
            save_preview_png(color_low, os.path.join(scene_dir, f"preview_low_spp{low_spp}.png"))

        if (scene_idx + 1) % 10 == 0 or scene_idx == 0:
            print(f"進捗: {scene_idx + 1}/{args.n} シーン完了")

    print(f"データセット生成完了: {args.out_dir}")

    # クリーンアップ
    glDeleteTextures(4, list(textures.values()))
    glDeleteBuffers(1, [ssbo])
    glDeleteProgram(program)
    glfw.terminate()


if __name__ == "__main__":
    main()
