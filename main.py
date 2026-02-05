import sys
import glfw
import numpy as np
import cv2
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram

WIDTH, HEIGHT = 256, 256
OUTPUT_PATH = "output.png"

COMPUTE_SHADER = """
#version 450
precision highp float;
layout(local_size_x = 16, local_size_y = 16) in;
layout(rgba8, binding = 0) uniform writeonly image2D img_output;
struct Sphere {
    vec4 posRadius; // xyz: 位置, w: 半径
    vec3 color;     // rgb: 色
    float ior;      // 屈折率
    vec4 ratios;    // x: 環境, y: 拡散, z: 反射, w: 屈折
};
layout(std430, binding=1) buffer Spheres { Sphere spheres[]; };
uniform int sphereCount;
uniform vec4 light; // xyz: 方向, w: 強さ

const int MAX_SPHERES = 16;
const int MAX_DEPTH = 3;
const int MAX_STACK = 16;
const float HIT_THRESHOLD = 1.0e-4;
const float MAX_DISTANCE = 100.0;
const int MAX_MARCH_STEPS = 100;

// レイタスク構造体
struct RayTask {
    vec3 origin;
    vec3 direction;
    float weight;
    int depth;
    bool inside;  // 物体内部にいるか
};

// 距離場の計算
void fieldAt(vec3 p, out float kMax, out float weightSum, out float weights[MAX_SPHERES]) {
    float k_ref = 8.0;  // 小さいほど影響範囲が広くくっつきやすい
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

// 距離のみ計算（影判定用の軽量版）
float distanceAt(vec3 p) {
    float k_ref = 8.0;  // 小さいほど影響範囲が広くくっつきやすい
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

// 法線計算（数値微分）
vec3 calcNormal(vec3 p) {
    const float eps = 1.0e-3;
    float kTmp;
    float wTmp;
    float weightsTmp[MAX_SPHERES];
    fieldAt(p + vec3(eps, 0.0, 0.0), kTmp, wTmp, weightsTmp);
    float fxp = wTmp;
    fieldAt(p - vec3(eps, 0.0, 0.0), kTmp, wTmp, weightsTmp);
    float fxm = wTmp;
    fieldAt(p + vec3(0.0, eps, 0.0), kTmp, wTmp, weightsTmp);
    float fyp = wTmp;
    fieldAt(p - vec3(0.0, eps, 0.0), kTmp, wTmp, weightsTmp);
    float fym = wTmp;
    fieldAt(p + vec3(0.0, 0.0, eps), kTmp, wTmp, weightsTmp);
    float fzp = wTmp;
    fieldAt(p - vec3(0.0, 0.0, eps), kTmp, wTmp, weightsTmp);
    float fzm = wTmp;
    return normalize(-vec3(fxp - fxm, fyp - fym, fzp - fzm));
}

// 影判定（光源方向にレイを飛ばして遮蔽物があるか）
float calcShadow(vec3 p, vec3 lightDir) {
    vec3 pos = p + lightDir * 0.01;  // 自己交差回避
    float distance = 0.0;
    for (int i = 0; i < 50; i++) {
        float d = distanceAt(pos);
        if (d < HIT_THRESHOLD) return 0.2;  // 影の中（完全な黒ではなく暗く）
        if (distance > MAX_DISTANCE) break;
        distance += d;
        pos += lightDir * d;
    }
    return 1.0;  // 影なし
}

// 地面との交差判定
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

// 背景色
vec3 backgroundColor(vec3 ray) {
    return vec3(0.5, 0.7, 1.0);
}

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(img_output);

    if (pixel.x >= size.x || pixel.y >= size.y) return;

    // カメラ設定
    float screen_depth = 1.2071067811865475;
    vec2 uv = (vec2(pixel) + 0.5) / vec2(size) - 0.5;
    vec3 camOrigin = vec3(0.0, 0.2, 0.0);
    vec3 lookAt = vec3(0.0, 0.0, -1.0);
    vec3 upHint = vec3(0.0, 1.0, 0.0);
    vec3 forward = normalize(lookAt - camOrigin);
    vec3 right = normalize(cross(forward, upHint));
    vec3 up = normalize(cross(right, forward));
    vec3 initialRay = normalize(uv.x * right - uv.y * up + screen_depth * forward);

    // スタック初期化
    RayTask stack[MAX_STACK];
    int stackPtr = 0;
    vec3 finalColor = vec3(0.0);

    // 初期レイをプッシュ
    stack[stackPtr].origin = camOrigin;
    stack[stackPtr].direction = initialRay;
    stack[stackPtr].weight = 1.0;
    stack[stackPtr].depth = 0;
    stack[stackPtr].inside = false;
    stackPtr++;

    // メインループ
    while (stackPtr > 0) {
        // ポップ
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
                // ヒット: マテリアル情報を集約
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

                vec3 lightDir = -light.xyz;
                float shadow = calcShadow(pos, lightDir);

                // 環境光
                finalColor += task.weight * ratios.x * color * light.w;

                // 拡散光（影を考慮）
                float diff = max(dot(normal, lightDir), 0.0);
                finalColor += task.weight * ratios.y * color * diff * light.w * shadow;

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

        // ヒットしなかった場合: 地面または背景
        if (!hit) {
            float t;
            vec3 groundColor;
            if (hitGround(task.origin, ray, t, groundColor)) {
                vec3 groundPos = task.origin + ray * t;
                vec3 groundNormal = vec3(0.0, 1.0, 0.0);
                vec3 lightDir = -light.xyz;
                float shadow = calcShadow(groundPos, lightDir);
                float diff = max(dot(groundNormal, lightDir), 0.0);
                finalColor += task.weight * groundColor * (0.3 + 0.7 * diff * shadow);
            } else {
                finalColor += task.weight * backgroundColor(ray);
            }
        }
    }

    finalColor = clamp(finalColor, 0.0, 1.0);
    imageStore(img_output, pixel, vec4(finalColor, 1.0));
}
"""


def main():
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

    # Compute Shaderコンパイル
    try:
        program = compileProgram(
            compileShader(COMPUTE_SHADER, GL_COMPUTE_SHADER),
        )
    except RuntimeError as e:
        print(f"シェーダーコンパイルエラー:\n{e}")
        glfw.terminate()
        sys.exit(1)

    # 出力テクスチャ作成（DSA）
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, WIDTH, HEIGHT)

    # 球データ作成（位置, 半径, 色, 屈折率, 環境/拡散/反射/屈折）
    # メタボールらしくくっつくように近接配置
    spheres = np.array(
        [
            # posRadius(x,y,z,r), color(r,g,b,ior), ratios(ambient,diffuse,reflect,refract)
            [(0.08, 0.02, -0.9, 0.10), (1.0, 0.3, 0.3, 1.5), (0.1, 0.7, 0.2, 0.0)],   # 赤
            [(-0.06, 0.0, -0.95, 0.08), (0.3, 1.0, 0.3, 1.5), (0.1, 0.7, 0.2, 0.0)],  # 緑
            [(0.0, 0.08, -0.92, 0.07), (0.3, 0.3, 1.0, 1.5), (0.1, 0.7, 0.2, 0.0)],   # 青
        ],
        dtype=np.float32,
    )

    # SSBO作成・バインド
    ssbo = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)
    glBufferData(GL_SHADER_STORAGE_BUFFER, spheres.nbytes, spheres, GL_STATIC_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo)

    # Compute Shader実行
    glUseProgram(program)
    count_loc = glGetUniformLocation(program, "sphereCount")
    glUniform1i(count_loc, spheres.shape[0])
    light_loc = glGetUniformLocation(program, "light")
    norm_light = np.array([1.0, -1.0, -1.0])
    norm_light /= np.linalg.norm(norm_light)
    glUniform4f(light_loc, norm_light[0], norm_light[1], norm_light[2], 1.0)
    glBindImageTexture(0, texture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8)
    glDispatchCompute((WIDTH + 15) // 16, (HEIGHT + 15) // 16, 1)
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

    # テクスチャからピクセル読み取り
    glBindTexture(GL_TEXTURE_2D, texture)
    data = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE)
    image = np.frombuffer(data, dtype=np.uint8).reshape(HEIGHT, WIDTH, 3)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(OUTPUT_PATH, image)
    print(f"保存しました: {OUTPUT_PATH} ({WIDTH}x{HEIGHT})")

    glfw.terminate()


if __name__ == "__main__":
    main()
