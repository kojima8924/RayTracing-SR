"""
デノイジング/超解像学習用データセット生成器
メタボールレイマーチングによるシーンを複数SPPでレンダリングし、
color/normal/depth/albedoの補助バッファとともに保存する。

補助バッファ（normal/depth/albedo）はジッター無しのprimary rayから計算され、
SPPやseedに関わらず同一シーンなら一致する。

ノイズ源:
1. Area Light: 面光源を円盤サンプルしてソフトシャドウを生成
2. Specular Jitter: スペキュラ反射方向を微小にブレさせる
3. DOF (被写界深度): レンズアパーチャでボケを表現
4. 背景ノイズ: 低周波value noiseベース
"""

import argparse
import json
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

// 出力バッファ（4つ）- WRITE_ONLYでbind
layout(rgba16f, binding = 0) uniform writeonly image2D img_color;
layout(rgba16f, binding = 1) uniform writeonly image2D img_normal;
layout(rgba16f, binding = 2) uniform writeonly image2D img_depth;
layout(rgba16f, binding = 3) uniform writeonly image2D img_albedo;

struct Sphere {
    vec4 posRadius; // xyz: 位置, w: 半径
    vec3 color;     // rgb: 色
    float ior;      // 屈折率
    vec4 ratios;    // x: 環境, y: 拡散, z: 反射, w: 屈折
};
layout(std430, binding=1) buffer Spheres { Sphere spheres[]; };

uniform int sphereCount;
uniform uint frameSeed;       // フレームごとのシード
uniform int spp;              // サンプル数
uniform float depthFar;       // depth正規化用の最大距離

// Area Light パラメータ
uniform vec3 lightCenter;     // 光源中心位置
uniform float lightRadius;    // 光源半径（面光源サイズ）
uniform float lightIntensity; // 光源強度

// Specular パラメータ
uniform float shininess;      // スペキュラの鋭さ
uniform float specJitterScale; // スペキュラジッターの強さ

// DOF パラメータ
uniform float apertureRadius; // レンズ開口半径（0で無効）
uniform float focusDistance;  // 焦点距離

// 背景パラメータ
uniform int bgType;           // 0:gradient, 1:checker, 2:noise
uniform vec3 bgColor1;
uniform vec3 bgColor2;
uniform float bgPhase;
uniform float bgNoiseScale;   // 背景ノイズ強度

const int MAX_SPHERES = 16;
const int MAX_DEPTH = 3;
const int MAX_STACK = 16;
const float HIT_THRESHOLD = 1.0e-4;
const float MAX_DISTANCE = 100.0;
const int MAX_MARCH_STEPS = 100;
const float PI = 3.14159265359;

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

// 円盤上の一様サンプル
vec2 sampleDisk(inout uint rng) {
    float r = sqrt(rand(rng));
    float theta = 2.0 * PI * rand(rng);
    return vec2(r * cos(theta), r * sin(theta));
}

// =============================================================================
// Value Noise（低周波ノイズ）
// =============================================================================
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float valueNoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f); // smoothstep
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
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
    int count = min(sphereCount, MAX_SPHERES);
    for (int j = 0; j < count; j++) {
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
    int count = min(sphereCount, MAX_SPHERES);
    for (int j = 0; j < count; j++) {
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
// Area Light ソフトシャドウ
// =============================================================================
float calcAreaLightShadow(vec3 hitPos, inout uint rng) {
    // 円盤上の光源サンプル
    vec2 diskSample = sampleDisk(rng) * lightRadius;

    // 光源座標系を構築
    vec3 lightDir = normalize(lightCenter - hitPos);
    vec3 tangent = normalize(cross(lightDir, vec3(0.0, 1.0, 0.001)));
    vec3 bitangent = cross(lightDir, tangent);

    // サンプル位置
    vec3 lightPos = lightCenter + tangent * diskSample.x + bitangent * diskSample.y;
    vec3 toLight = normalize(lightPos - hitPos);

    // シャドウレイ
    vec3 pos = hitPos + toLight * 0.01;
    float maxDist = length(lightPos - hitPos);
    float distance = 0.0;

    for (int i = 0; i < 50; i++) {
        float d = distanceAt(pos);
        if (d < HIT_THRESHOLD) return 0.15; // 影の中
        if (distance > maxDist) break;
        distance += d;
        pos += toLight * d;
    }
    return 1.0;
}

// シェーディング用の光源方向を取得（ジッター付き）
vec3 getAreaLightDir(vec3 hitPos, inout uint rng) {
    vec2 diskSample = sampleDisk(rng) * lightRadius;
    vec3 lightDir = normalize(lightCenter - hitPos);
    vec3 tangent = normalize(cross(lightDir, vec3(0.0, 1.0, 0.001)));
    vec3 bitangent = cross(lightDir, tangent);
    vec3 lightPos = lightCenter + tangent * diskSample.x + bitangent * diskSample.y;
    return normalize(lightPos - hitPos);
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
// Procedural背景（決定論的版 - primary用）
// =============================================================================
vec3 proceduralBgDeterministic(vec3 ray) {
    if (bgType == 0) {
        float t = ray.y * 0.5 + 0.5 + sin(bgPhase) * 0.1;
        return mix(bgColor1, bgColor2, clamp(t, 0.0, 1.0));
    } else if (bgType == 1) {
        float u = atan(ray.z, ray.x) / PI * 4.0 + bgPhase;
        float v = asin(ray.y) / (PI * 0.5) * 4.0;
        int cu = int(floor(u));
        int cv = int(floor(v));
        return (((cu + cv) & 1) == 0) ? bgColor1 : bgColor2;
    } else {
        // Value noiseベースの背景（決定論的成分のみ）
        vec2 uv = vec2(atan(ray.z, ray.x) / PI, ray.y);
        float noise = valueNoise(uv * 8.0 + bgPhase);
        float t = ray.y * 0.5 + 0.5 + (noise - 0.5) * bgNoiseScale;
        return mix(bgColor1, bgColor2, clamp(t, 0.0, 1.0));
    }
}

// =============================================================================
// Procedural背景（ノイズ付き版 - color用）
// =============================================================================
vec3 proceduralBgNoisy(vec3 ray, ivec2 pixel, inout uint rng) {
    if (bgType == 0) {
        // グラデーション + 微小ノイズ
        float sampleNoise = (rand(rng) - 0.5) * bgNoiseScale * 0.3;
        float t = ray.y * 0.5 + 0.5 + sin(bgPhase) * 0.1 + sampleNoise;
        return mix(bgColor1, bgColor2, clamp(t, 0.0, 1.0));
    } else if (bgType == 1) {
        // 球面チェッカー + エッジアンチエイリアス風ノイズ
        float u = atan(ray.z, ray.x) / PI * 4.0 + bgPhase;
        float v = asin(ray.y) / (PI * 0.5) * 4.0;
        float fu = fract(u);
        float fv = fract(v);
        float edgeDist = min(min(fu, 1.0-fu), min(fv, 1.0-fv));
        float noise = 0.0;
        if (edgeDist < 0.15) {
            noise = (rand(rng) - 0.5) * bgNoiseScale * (1.0 - edgeDist / 0.15);
        }
        int cu = int(floor(u + noise * 0.5));
        int cv = int(floor(v));
        return (((cu + cv) & 1) == 0) ? bgColor1 : bgColor2;
    } else {
        // Value noise + サンプルノイズ
        vec2 uv = vec2(atan(ray.z, ray.x) / PI, ray.y);
        float baseNoise = valueNoise(uv * 8.0 + bgPhase);
        float sampleNoise = rand(rng) * bgNoiseScale * 0.2;
        float t = ray.y * 0.5 + 0.5 + (baseNoise - 0.5) * bgNoiseScale + sampleNoise;
        return mix(bgColor1, bgColor2, clamp(t, 0.0, 1.0));
    }
}

// =============================================================================
// Primary Ray Hit判定（ジッターなし、補助バッファ用）
// =============================================================================
struct PrimaryHitResult {
    vec3 normal;
    float depth;
    vec3 albedo;
};

PrimaryHitResult tracePrimaryRay(vec3 camOrigin, vec3 ray) {
    PrimaryHitResult result;
    result.normal = vec3(0.5, 0.5, 1.0);
    result.depth = 1.0;
    result.albedo = vec3(0.0);

    vec3 pos = camOrigin;
    float distance = 0.0;
    int count = min(sphereCount, MAX_SPHERES);

    for (int i = 0; i < MAX_MARCH_STEPS; i++) {
        if (distance > MAX_DISTANCE) break;
        if (count == 0) break;

        float kMax;
        float weightSum;
        float weights[MAX_SPHERES];
        fieldAt(pos, kMax, weightSum, weights);
        weightSum = max(weightSum, 1.0e-6);
        float minDist = -log(weightSum) / max(kMax, 1.0e-6);

        if (minDist < HIT_THRESHOLD) {
            vec3 hitPos = pos;
            vec3 normal = calcNormal(hitPos);
            vec3 albedo = vec3(0.0);
            for (int j = 0; j < count; j++) {
                albedo += spheres[j].color * weights[j];
            }
            albedo /= weightSum;
            float depthDistance = length(hitPos - camOrigin);
            result.normal = normal * 0.5 + 0.5;
            result.depth = clamp(depthDistance / depthFar, 0.0, 1.0);
            result.albedo = albedo;
            return result;
        }

        distance += minDist;
        pos += ray * minDist;
    }

    float t;
    vec3 groundColor;
    if (hitGround(camOrigin, ray, t, groundColor)) {
        vec3 hitPos = camOrigin + ray * t;
        float depthDistance = length(hitPos - camOrigin);
        result.normal = vec3(0.5, 1.0, 0.5);
        result.depth = clamp(depthDistance / depthFar, 0.0, 1.0);
        result.albedo = groundColor;
        return result;
    }

    result.albedo = proceduralBgDeterministic(ray);
    return result;
}

// =============================================================================
// Color トレース（SPPループ内、ジッター付き）
// =============================================================================
vec3 traceColorSample(vec3 camOrigin, vec3 initialRay, ivec2 pixel, inout uint rng) {
    RayTask stack[MAX_STACK];
    int stackPtr = 0;
    vec3 sampleColor = vec3(0.0);

    stack[stackPtr].origin = camOrigin;
    stack[stackPtr].direction = initialRay;
    stack[stackPtr].weight = 1.0;
    stack[stackPtr].depth = 0;
    stack[stackPtr].inside = false;
    stackPtr++;

    int count = min(sphereCount, MAX_SPHERES);

    while (stackPtr > 0) {
        stackPtr--;
        RayTask task = stack[stackPtr];

        if (task.depth > MAX_DEPTH) continue;
        if (task.weight < 0.01) continue;

        vec3 pos = task.origin;
        vec3 ray = task.direction;
        float distance = 0.0;
        bool hit = false;

        for (int i = 0; i < MAX_MARCH_STEPS; i++) {
            if (distance > MAX_DISTANCE) break;
            if (count == 0) break;

            float kMax;
            float weightSum;
            float weights[MAX_SPHERES];
            fieldAt(pos, kMax, weightSum, weights);
            weightSum = max(weightSum, 1.0e-6);
            float minDist = -log(weightSum) / max(kMax, 1.0e-6);

            if (minDist < HIT_THRESHOLD) {
                vec4 ratios = vec4(0.0);
                vec3 color = vec3(0.0);
                float ior = 0.0;
                for (int j = 0; j < count; j++) {
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

                // Area Light シャドウ
                float shadow = calcAreaLightShadow(pos, rng);
                vec3 lightDir = getAreaLightDir(pos, rng);

                // 環境光
                sampleColor += task.weight * ratios.x * color * lightIntensity;

                // 拡散光
                float diff = max(dot(normal, lightDir), 0.0);
                sampleColor += task.weight * ratios.y * color * diff * lightIntensity * shadow;

                // スペキュラハイライト（ジッター強化）
                vec3 viewDir = -ray;
                vec3 reflectLightDir = reflect(-lightDir, normal);
                vec3 specTangent = normalize(cross(reflectLightDir, vec3(0.0, 1.0, 0.001)));
                vec3 specBitangent = cross(reflectLightDir, specTangent);
                vec3 specJitter = specTangent * (rand(rng) - 0.5) * specJitterScale
                                + specBitangent * (rand(rng) - 0.5) * specJitterScale;
                vec3 jitteredReflect = normalize(reflectLightDir + specJitter);
                float spec = pow(max(dot(jitteredReflect, viewDir), 0.0), shininess);
                vec3 specColor = mix(vec3(1.0), color, 0.3);
                sampleColor += task.weight * ratios.y * specColor * spec * lightIntensity * shadow * 0.6;

                // 反射
                if (ratios.z > 0.01 && stackPtr < MAX_STACK) {
                    vec3 reflectDir = reflect(ray, normal);
                    // 反射方向にも微小ジッター
                    vec3 refTangent = normalize(cross(reflectDir, vec3(0.0, 1.0, 0.001)));
                    vec3 refBitangent = cross(reflectDir, refTangent);
                    vec3 refJitter = refTangent * (rand(rng) - 0.5) * specJitterScale * 0.5
                                   + refBitangent * (rand(rng) - 0.5) * specJitterScale * 0.5;
                    reflectDir = normalize(reflectDir + refJitter);

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

        if (!hit) {
            float t;
            vec3 groundColor;
            if (hitGround(task.origin, ray, t, groundColor)) {
                vec3 groundPos = task.origin + ray * t;
                vec3 groundNormal = vec3(0.0, 1.0, 0.0);
                float shadow = calcAreaLightShadow(groundPos, rng);
                vec3 lightDir = getAreaLightDir(groundPos, rng);
                float diff = max(dot(groundNormal, lightDir), 0.0);
                sampleColor += task.weight * groundColor * (0.3 + 0.7 * diff * shadow);
            } else {
                sampleColor += task.weight * proceduralBgNoisy(ray, pixel, rng);
            }
        }
    }

    return sampleColor;
}

// =============================================================================
// メイン
// =============================================================================
void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = ivec2(imageSize(img_color));

    if (pixel.x >= size.x || pixel.y >= size.y) return;

    // カメラ設定
    float screen_depth = 1.2071067811865475;
    vec3 camOrigin = vec3(0.0, 0.2, 0.0);
    vec3 lookAt = vec3(0.0, 0.0, -1.0);
    vec3 upHint = vec3(0.0, 1.0, 0.0);
    vec3 forward = normalize(lookAt - camOrigin);
    vec3 right = normalize(cross(forward, upHint));
    vec3 up = normalize(cross(right, forward));

    // ========================================================================
    // Primary Pass: ジッターなしで補助バッファを計算（SPPに依存しない）
    // ========================================================================
    vec2 uv0 = (vec2(pixel) + 0.5) / vec2(size) - 0.5;
    vec3 primaryRay = normalize(uv0.x * right - uv0.y * up + screen_depth * forward);
    PrimaryHitResult primary = tracePrimaryRay(camOrigin, primaryRay);

    // ========================================================================
    // Color Pass: SPPループでジッター付きサンプリング
    // ========================================================================
    vec3 accumulatedColor = vec3(0.0);

    for (int s = 0; s < spp; s++) {
        uint rng = initRNG(pixel, s);

        // サブピクセルジッター
        vec2 jitter = vec2(rand(rng), rand(rng)) - 0.5;
        vec2 uv = (vec2(pixel) + 0.5 + jitter) / vec2(size) - 0.5;
        vec3 sampleRay = normalize(uv.x * right - uv.y * up + screen_depth * forward);

        // DOF: レンズアパーチャサンプリング
        vec3 sampleOrigin = camOrigin;
        if (apertureRadius > 0.0) {
            // 焦点面上の点を計算
            vec3 focalPoint = camOrigin + sampleRay * focusDistance;
            // レンズ上のランダム位置
            vec2 lensSample = sampleDisk(rng) * apertureRadius;
            sampleOrigin = camOrigin + right * lensSample.x + up * lensSample.y;
            // 新しいレイ方向
            sampleRay = normalize(focalPoint - sampleOrigin);
        }

        accumulatedColor += traceColorSample(sampleOrigin, sampleRay, pixel, rng);
    }

    vec3 finalColor = accumulatedColor / float(spp);
    finalColor = clamp(finalColor, 0.0, 1.0);

    imageStore(img_color, pixel, vec4(finalColor, 1.0));
    imageStore(img_normal, pixel, vec4(primary.normal, 1.0));
    imageStore(img_depth, pixel, vec4(vec3(primary.depth), 1.0));
    imageStore(img_albedo, pixel, vec4(primary.albedo, 1.0));
}
"""


# =============================================================================
# 材質プリセット
# =============================================================================
MATERIAL_PRESETS = {
    'diffuse': {'ratios': [0.1, 0.85, 0.05, 0.0], 'weight': 0.4},
    'glossy': {'ratios': [0.1, 0.5, 0.4, 0.0], 'weight': 0.3},
    'metallic': {'ratios': [0.05, 0.2, 0.75, 0.0], 'weight': 0.15},
    'glassy': {'ratios': [0.05, 0.15, 0.15, 0.65], 'weight': 0.15},
}


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
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    glBindTexture(GL_TEXTURE_2D, tex)
    data = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT)
    return np.frombuffer(data, dtype=np.float32).reshape(height, width, 4).copy()


def select_material_preset(rng: np.random.Generator) -> np.ndarray:
    """材質プリセットからランダムに選択"""
    names = list(MATERIAL_PRESETS.keys())
    weights = [MATERIAL_PRESETS[n]['weight'] for n in names]
    weights = np.array(weights) / sum(weights)
    name = rng.choice(names, p=weights)
    base_ratios = np.array(MATERIAL_PRESETS[name]['ratios'])
    # 少しだけランダムに乱す
    noise = rng.uniform(-0.05, 0.05, size=4)
    ratios = np.clip(base_ratios + noise, 0.01, 1.0)
    ratios = ratios / ratios.sum()
    return ratios


def generate_random_spheres(rng: np.random.Generator, n: int = None) -> np.ndarray:
    """ランダムな球データを生成"""
    if n is None:
        n = rng.integers(3, 8)
    n = min(n, 16)

    spheres = []
    for _ in range(n):
        x = rng.uniform(-0.3, 0.3)
        y = rng.uniform(-0.05, 0.2)
        z = rng.uniform(-1.2, -0.6)
        r = rng.uniform(0.04, 0.14)
        color = rng.uniform(0.3, 1.0, size=3)
        ior = rng.uniform(1.3, 1.6)
        ratios = select_material_preset(rng)

        spheres.append([
            (x, y, z, r),
            (*color, ior),
            tuple(ratios)
        ])

    return np.array(spheres, dtype=np.float32)


def generate_random_light(rng: np.random.Generator) -> dict:
    """ランダムな面光源パラメータを生成"""
    theta = rng.uniform(0, 2 * np.pi)
    phi = rng.uniform(0.2, np.pi / 3)
    distance = rng.uniform(1.5, 3.0)

    x = distance * np.sin(phi) * np.cos(theta)
    y = distance * np.cos(phi)
    z = distance * np.sin(phi) * np.sin(theta) - 1.0

    center = np.array([x, y, z])
    radius = rng.uniform(0.1, 0.4)
    intensity = rng.uniform(0.9, 1.3)

    return {
        'center': center,
        'radius': radius,
        'intensity': intensity
    }


def generate_random_background(rng: np.random.Generator) -> dict:
    """ランダムな背景パラメータを生成"""
    bg_type = rng.integers(0, 3)
    color1 = rng.uniform(0.3, 1.0, size=3)
    color2 = rng.uniform(0.3, 1.0, size=3)
    phase = rng.uniform(0, 2 * np.pi)
    noise_scale = rng.uniform(0.1, 0.4)
    return {
        'type': bg_type,
        'color1': color1,
        'color2': color2,
        'phase': phase,
        'noise_scale': noise_scale
    }


def render_scene(
    program: int,
    textures: dict,
    spheres: np.ndarray,
    ssbo: int,
    light_params: dict,
    bg_params: dict,
    width: int,
    height: int,
    spp: int,
    frame_seed: int,
    depth_far: float = 5.0,
    shininess: float = 48.0,
    spec_jitter_scale: float = 0.15,
    aperture_radius: float = 0.0,
    focus_distance: float = 1.0
):
    """シーンをレンダリング"""
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)
    glBufferData(GL_SHADER_STORAGE_BUFFER, spheres.nbytes, spheres, GL_STATIC_DRAW)

    glUseProgram(program)

    # Uniform設定
    glUniform1i(glGetUniformLocation(program, "sphereCount"), min(spheres.shape[0], 16))
    glUniform1ui(glGetUniformLocation(program, "frameSeed"), frame_seed)
    glUniform1i(glGetUniformLocation(program, "spp"), spp)
    glUniform1f(glGetUniformLocation(program, "depthFar"), depth_far)

    # Area Light
    glUniform3f(glGetUniformLocation(program, "lightCenter"), *light_params['center'])
    glUniform1f(glGetUniformLocation(program, "lightRadius"), light_params['radius'])
    glUniform1f(glGetUniformLocation(program, "lightIntensity"), light_params['intensity'])

    # Specular
    glUniform1f(glGetUniformLocation(program, "shininess"), shininess)
    glUniform1f(glGetUniformLocation(program, "specJitterScale"), spec_jitter_scale)

    # DOF
    glUniform1f(glGetUniformLocation(program, "apertureRadius"), aperture_radius)
    glUniform1f(glGetUniformLocation(program, "focusDistance"), focus_distance)

    # 背景
    glUniform1i(glGetUniformLocation(program, "bgType"), bg_params['type'])
    glUniform3f(glGetUniformLocation(program, "bgColor1"), *bg_params['color1'])
    glUniform3f(glGetUniformLocation(program, "bgColor2"), *bg_params['color2'])
    glUniform1f(glGetUniformLocation(program, "bgPhase"), bg_params['phase'])
    glUniform1f(glGetUniformLocation(program, "bgNoiseScale"), bg_params['noise_scale'])

    # テクスチャをバインド
    glBindImageTexture(0, textures['color'], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F)
    glBindImageTexture(1, textures['normal'], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F)
    glBindImageTexture(2, textures['depth'], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F)
    glBindImageTexture(3, textures['albedo'], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F)

    glDispatchCompute((width + 15) // 16, (height + 15) // 16, 1)
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)


def save_preview_png(data: np.ndarray, path: str, apply_gamma: bool = True):
    """float32データをPNGプレビューとして保存"""
    img = np.clip(data[:, :, :3], 0, 1)
    if apply_gamma:
        img = np.power(img, 1.0 / 2.2)
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def save_metadata(scene_dir: str, spheres: np.ndarray, light_params: dict,
                  bg_params: dict, frame_seed: int, render_params: dict):
    """シーンのメタデータをJSONで保存"""
    metadata = {
        'spheres': [],
        'light': {
            'center': light_params['center'].tolist(),
            'radius': float(light_params['radius']),
            'intensity': float(light_params['intensity'])
        },
        'background': {
            'type': int(bg_params['type']),
            'color1': bg_params['color1'].tolist(),
            'color2': bg_params['color2'].tolist(),
            'phase': float(bg_params['phase']),
            'noise_scale': float(bg_params['noise_scale'])
        },
        'frame_seed': int(frame_seed),
        'render_params': render_params
    }

    for i in range(spheres.shape[0]):
        sphere_data = {
            'position': spheres[i, 0, :3].tolist(),
            'radius': float(spheres[i, 0, 3]),
            'color': spheres[i, 1, :3].tolist(),
            'ior': float(spheres[i, 1, 3]),
            'ratios': spheres[i, 2, :4].tolist()
        }
        metadata['spheres'].append(sphere_data)

    with open(os.path.join(scene_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)


def validate_auxiliary_buffers(textures: dict, program: int, spheres: np.ndarray,
                               ssbo: int, light_params: dict, bg_params: dict,
                               width: int, height: int, frame_seed: int,
                               render_params: dict) -> dict:
    """補助バッファがSPPに依存しないことを検証"""
    # SPP=1でレンダリング
    render_scene(program, textures, spheres, ssbo, light_params, bg_params,
                 width, height, 1, frame_seed, **render_params)
    normal_spp1 = read_texture_float32(textures['normal'], width, height)
    depth_spp1 = read_texture_float32(textures['depth'], width, height)
    albedo_spp1 = read_texture_float32(textures['albedo'], width, height)

    # SPP=64でレンダリング
    render_scene(program, textures, spheres, ssbo, light_params, bg_params,
                 width, height, 64, frame_seed, **render_params)
    normal_spp64 = read_texture_float32(textures['normal'], width, height)
    depth_spp64 = read_texture_float32(textures['depth'], width, height)
    albedo_spp64 = read_texture_float32(textures['albedo'], width, height)

    return {
        'normal_max_diff': float(np.abs(normal_spp1 - normal_spp64).max()),
        'depth_max_diff': float(np.abs(depth_spp1 - depth_spp64).max()),
        'albedo_max_diff': float(np.abs(albedo_spp1 - albedo_spp64).max()),
    }


def main():
    parser = argparse.ArgumentParser(description='デノイジング/超解像学習用データセット生成')
    parser.add_argument('--out_dir', type=str, required=True, help='出力ディレクトリ')
    parser.add_argument('--n', type=int, default=1000, help='生成するシーン数')
    parser.add_argument('--width', type=int, default=256, help='画像幅')
    parser.add_argument('--height', type=int, default=256, help='画像高さ')
    parser.add_argument('--low_spp', type=int, nargs='+', default=[1, 2, 4, 8, 16], help='低SPPリスト')
    parser.add_argument('--high_spp', type=int, default=256, help='高SPP（Ground Truth）')
    parser.add_argument('--seed', type=int, default=42, help='乱数シード')
    parser.add_argument('--depth_far', type=float, default=5.0, help='depth正規化の最大距離')
    parser.add_argument('--shininess', type=float, default=48.0, help='スペキュラの鋭さ')
    parser.add_argument('--spec_jitter', type=float, default=0.15, help='スペキュラジッター強度')
    parser.add_argument('--aperture', type=float, default=0.0, help='DOFアパーチャ半径（0で無効）')
    parser.add_argument('--focus_dist', type=float, default=1.0, help='DOF焦点距離')
    parser.add_argument('--validate', action='store_true', help='補助バッファの一貫性を検証')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

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

    try:
        program = compileProgram(compileShader(COMPUTE_SHADER, GL_COMPUTE_SHADER))
    except RuntimeError as e:
        print(f"シェーダーコンパイルエラー:\n{e}")
        glfw.terminate()
        sys.exit(1)

    textures = {
        'color': create_texture_rgba16f(args.width, args.height),
        'normal': create_texture_rgba16f(args.width, args.height),
        'depth': create_texture_rgba16f(args.width, args.height),
        'albedo': create_texture_rgba16f(args.width, args.height),
    }

    ssbo = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo)

    render_params = {
        'depth_far': args.depth_far,
        'shininess': args.shininess,
        'spec_jitter_scale': args.spec_jitter,
        'aperture_radius': args.aperture,
        'focus_distance': args.focus_dist
    }

    print(f"データセット生成開始: {args.n}シーン, {args.width}x{args.height}")
    print(f"低SPP: {args.low_spp}, 高SPP: {args.high_spp}")
    if args.validate:
        print("補助バッファ検証モード有効")

    validation_results = []

    for scene_idx in range(args.n):
        scene_dir = os.path.join(args.out_dir, f"{scene_idx:06d}")
        os.makedirs(scene_dir, exist_ok=True)

        spheres = generate_random_spheres(rng)
        light_params = generate_random_light(rng)
        bg_params = generate_random_background(rng)
        frame_seed = rng.integers(0, 2**31)

        # 検証モード
        if args.validate:
            result = validate_auxiliary_buffers(
                textures, program, spheres, ssbo, light_params, bg_params,
                args.width, args.height, frame_seed, render_params
            )
            validation_results.append(result)

        # 高SPPレンダリング
        render_scene(program, textures, spheres, ssbo, light_params, bg_params,
                     args.width, args.height, args.high_spp, frame_seed, **render_params)

        color_high = read_texture_float32(textures['color'], args.width, args.height)
        normal = read_texture_float32(textures['normal'], args.width, args.height)
        depth = read_texture_float32(textures['depth'], args.width, args.height)
        albedo = read_texture_float32(textures['albedo'], args.width, args.height)

        np.save(os.path.join(scene_dir, "color_high.npy"), color_high)
        np.save(os.path.join(scene_dir, "normal.npy"), normal)
        np.save(os.path.join(scene_dir, "depth.npy"), depth)
        np.save(os.path.join(scene_dir, "albedo.npy"), albedo)
        save_preview_png(color_high, os.path.join(scene_dir, "preview_high.png"))

        # メタデータ保存
        save_metadata(scene_dir, spheres, light_params, bg_params, frame_seed, render_params)

        # 低SPPレンダリング
        for low_spp in args.low_spp:
            render_scene(program, textures, spheres, ssbo, light_params, bg_params,
                         args.width, args.height, low_spp, frame_seed, **render_params)
            color_low = read_texture_float32(textures['color'], args.width, args.height)
            np.save(os.path.join(scene_dir, f"color_low_spp{low_spp}.npy"), color_low)
            save_preview_png(color_low, os.path.join(scene_dir, f"preview_low_spp{low_spp}.png"))

        if (scene_idx + 1) % 10 == 0 or scene_idx == 0:
            print(f"進捗: {scene_idx + 1}/{args.n} シーン完了")

    # 検証結果出力
    if args.validate and validation_results:
        print("\n=== 補助バッファ一貫性検証結果 ===")
        max_normal = max(r['normal_max_diff'] for r in validation_results)
        max_depth = max(r['depth_max_diff'] for r in validation_results)
        max_albedo = max(r['albedo_max_diff'] for r in validation_results)
        print(f"normal max_abs_diff: {max_normal:.10f}")
        print(f"depth  max_abs_diff: {max_depth:.10f}")
        print(f"albedo max_abs_diff: {max_albedo:.10f}")
        if max_normal <= 1e-6 and max_depth <= 1e-6 and max_albedo <= 1e-6:
            print("✓ 補助バッファはSPPに依存しません")
        else:
            print("! 補助バッファに差分があります")

    print(f"データセット生成完了: {args.out_dir}")

    glDeleteTextures(4, list(textures.values()))
    glDeleteBuffers(1, [ssbo])
    glDeleteProgram(program)
    glfw.terminate()


if __name__ == "__main__":
    main()
