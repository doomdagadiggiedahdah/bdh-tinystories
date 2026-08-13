/*
 * BDH TinyStories — run a byte-level language model on ESP32-S3
 *
 * Weights live in flash (int8 + per-tensor scale), memory-mapped.
 * Inference is incremental: each new token computes only its own position,
 * attending over int8-quantized caches in PSRAM (like a KV cache).
 * Weight matrices are stored so every inner loop reads flash sequentially.
 *
 * Serial REPL: type a story prompt, model continues it.
 */

#include <math.h>

#ifndef ARDUINO_HOST_TEST
#include "esp_spi_flash.h"
#endif

// Weights are flashed to this raw address — no partition table needed
#define WEIGHTS_FLASH_ADDR 0x100000

// Max sequence length (prompt + generation). Cache cost ~71KB/position.
#define T_MAX 80

// --- Per-stage profiling ---
// Wraps each stage in micros() timers so we optimize what is actually slow
// rather than what looks slow. Costs ~24 micros() calls per stage per token,
// which is noise against a ~2s token. Set to 0 for a clean build.
#define PROFILE 1

#if PROFILE && !defined(ARDUINO_HOST_TEST)
enum { PF_EMBED, PF_ENCODER, PF_ROPE, PF_ATTN, PF_ENCV, PF_DECODER, PF_LMHEAD, PF_N };
static const char* pf_names[PF_N] = {
    "embed", "encoder", "rope+quant", "attention", "encoder_v", "decoder", "lm_head"
};
static uint32_t pf_us[PF_N];
static uint32_t pf_t0;
#define PF_START()   (pf_t0 = micros())
#define PF_END(slot) (pf_us[slot] += micros() - pf_t0)
#define PF_RESET()   memset(pf_us, 0, sizeof(pf_us))
#else
#define PF_START()
#define PF_END(slot)
#define PF_RESET()
#endif

// --- Weight header (64 bytes, matches export_weights.py) ---
struct __attribute__((packed)) WeightHeader {
    uint32_t magic;      // 0x42444802 — layout v2 (transposed encoders)
    uint32_t d_model;
    uint32_t n_head;
    uint32_t n_inner;    // N = mlp_mult * D / nh
    uint32_t n_layer;
    uint32_t vocab_size;
    float embed_scale;
    float encoder_scale;
    float encoder_v_scale;
    float decoder_scale;
    float lm_head_scale;
};

// --- Model config (read from header) ---
static uint32_t D, nh, N, n_layer, vocab_size;
static uint32_t nhN;

// --- Weight pointers (memory-mapped flash, int8) ---
// Layout v2: encoder/encoder_v are TRANSPOSED to [nh][N][D] so the
// per-neuron dot product reads D consecutive bytes.
static const int8_t* w_embed;      // [vocab, D]
static const int8_t* w_encoder;    // [nh][N][D]
static const int8_t* w_encoder_v;  // [nh][N][D]
static const int8_t* w_decoder;    // [nhN][D]
static const int8_t* w_lm_head;    // [D][vocab]
static float s_embed, s_encoder, s_encoder_v, s_decoder, s_lm_head;

// Non-null if the encoder tensor was hoisted from flash into PSRAM.
static int8_t* encoder_psram = nullptr;

// --- Incremental caches (PSRAM) ---
// kr_cache: RoPE'd sparse activations, int8 per (layer, head, pos)
static int8_t* kr_cache;    // [n_layer][nh][T_MAX][N]
static float*  kr_scale;    // [n_layer][nh][T_MAX]
static float*  x_cache;     // [n_layer][T_MAX][D] — residual input per layer

// --- Per-step scratch ---
static float* cur_x;        // [D]
static float* cur_sparse;   // [nh][N]  x_sparse of current position
static float* cur_qr;       // [N]      RoPE'd sparse (one head at a time)
static float* cur_xy;       // [nh][N]  gated product
static float* cur_ykv;      // [D]
static float* cur_y;        // [D]
static float* buf_logits;   // [256]
static float* rope_freqs;   // [N]

static int seq_pos = 0;     // current sequence length

// ============================================================
//  Math helpers
// ============================================================

static void layernorm(float* out, const float* in, int len) {
    float mean = 0;
    for (int i = 0; i < len; i++) mean += in[i];
    mean /= len;
    float var = 0;
    for (int i = 0; i < len; i++) { float d = in[i] - mean; var += d * d; }
    var /= len;
    float inv = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < len; i++) out[i] = (in[i] - mean) * inv;
}

// dot of float vector with sequential int8 row
static inline float dot_f32_i8(const float* a, const int8_t* b, int K) {
    float sum = 0;
    for (int k = 0; k < K; k++) sum += a[k] * (float)b[k];
    return sum;
}

// Apply RoPE in place-compatible way (src/dst may differ), length len.
// Valid because BDH pair-quantizes freqs (freqs[2i] == freqs[2i+1]).
static void rope_apply(const float* src, float* dst, const float* freqs, int pos, int len) {
    for (int i = 0; i + 1 < len; i += 2) {
        float phase = pos * freqs[i];
        phase = (phase - floorf(phase)) * 2.0f * M_PI;
        float c = cosf(phase);
        float s = sinf(phase);
        dst[i]   = src[i] * c - src[i+1] * s;
        dst[i+1] = src[i+1] * c + src[i] * s;
    }
}

// Compute RoPE frequencies (matches Python get_freqs, theta=2^16)
static void compute_rope_freqs(float* out, int n, float theta) {
    for (int i = 0; i < n; i++) {
        float qi = floorf((float)i / 2.0f) * 2.0f;
        out[i] = (1.0f / powf(theta, qi / (float)n)) / (2.0f * M_PI);
    }
}

// ============================================================
//  Incremental BDH step
// ============================================================

// Process one token at position seq_pos; leaves next-token logits in
// buf_logits and advances seq_pos.
static void bdh_step(uint8_t token) {
    int t = seq_pos;

    // 1. Embedding + LayerNorm
    PF_START();
    {
        const int8_t* row = w_embed + (uint32_t)token * D;
        for (int d = 0; d < (int)D; d++) cur_x[d] = (float)row[d] * s_embed;
        layernorm(cur_x, cur_x, D);
    }
    PF_END(PF_EMBED);

    // 2. Layers (shared weights)
    for (int l = 0; l < (int)n_layer; l++) {
        // Cache this layer's residual input for future positions' attention
        float* xc = x_cache + (l * T_MAX + t) * D;
        memcpy(xc, cur_x, D * sizeof(float));

        // Per head: sparse projection, RoPE, attention
        for (int h = 0; h < (int)nh; h++) {
            // 2a. x_sparse = relu(x @ encoder[h]) — encoder row n is D sequential bytes
            PF_START();
            const int8_t* enc_h = w_encoder + (uint32_t)h * N * D;
            float* xs = cur_sparse + h * N;
            for (int n = 0; n < (int)N; n++) {
                float v = dot_f32_i8(cur_x, enc_h + (uint32_t)n * D, D) * s_encoder;
                xs[n] = v > 0 ? v : 0;
            }
            PF_END(PF_ENCODER);

            // 2b. RoPE current position, quantize into cache
            PF_START();
            rope_apply(xs, cur_qr, rope_freqs, t, N);
            {
                float absmax = 0;
                for (int n = 0; n < (int)N; n++) {
                    float a = fabsf(cur_qr[n]);
                    if (a > absmax) absmax = a;
                }
                float sc = absmax > 0 ? absmax / 127.0f : 1.0f;
                int8_t* krc = kr_cache + (((uint32_t)l * nh + h) * T_MAX + t) * N;
                for (int n = 0; n < (int)N; n++) {
                    int q = (int)roundf(cur_qr[n] / sc);
                    krc[n] = (int8_t)(q > 127 ? 127 : (q < -127 ? -127 : q));
                }
                kr_scale[((uint32_t)l * nh + h) * T_MAX + t] = sc;
            }
            PF_END(PF_ROPE);

            // 2c. Attention: yKV[h] = sum_{t2<t} (qr . kr[t2]) * x_cache[l][t2]
            //     (strictly causal — diagonal excluded, matches tril(-1))
            PF_START();
            for (int d = 0; d < (int)D; d++) cur_ykv[d] = 0;
            for (int t2 = 0; t2 < t; t2++) {
                const int8_t* krc = kr_cache + (((uint32_t)l * nh + h) * T_MAX + t2) * N;
                float score = dot_f32_i8(cur_qr, krc, N)
                            * kr_scale[((uint32_t)l * nh + h) * T_MAX + t2];
                const float* xv = x_cache + (l * T_MAX + t2) * D;
                for (int d = 0; d < (int)D; d++) cur_ykv[d] += score * xv[d];
            }
            layernorm(cur_ykv, cur_ykv, D);
            PF_END(PF_ATTN);

            // 2d. Gate: xy = x_sparse * relu(yKV @ encoder_v[h])
            PF_START();
            const int8_t* encv_h = w_encoder_v + (uint32_t)h * N * D;
            float* xy = cur_xy + h * N;
            for (int n = 0; n < (int)N; n++) {
                if (xs[n] == 0) { xy[n] = 0; continue; }  // gate is zero anyway
                float v = dot_f32_i8(cur_ykv, encv_h + (uint32_t)n * D, D) * s_encoder_v;
                xy[n] = v > 0 ? xs[n] * v : 0;
            }
            PF_END(PF_ENCV);
        }

        // 2e. Decoder: y[d] = sum_i xy[i] * decoder[i][d], skipping zero rows
        PF_START();
        for (int d = 0; d < (int)D; d++) cur_y[d] = 0;
        for (int i = 0; i < (int)nhN; i++) {
            float g = cur_xy[i];
            if (g == 0) continue;  // ReLU sparsity — skip the flash read entirely
            const int8_t* row = w_decoder + (uint32_t)i * D;
            for (int d = 0; d < (int)D; d++) cur_y[d] += g * (float)row[d];
        }
        for (int d = 0; d < (int)D; d++) cur_y[d] *= s_decoder;

        layernorm(cur_y, cur_y, D);
        for (int d = 0; d < (int)D; d++) cur_x[d] += cur_y[d];
        layernorm(cur_x, cur_x, D);
        PF_END(PF_DECODER);
    }

    // 3. Logits: accumulate over sequential lm_head rows
    PF_START();
    for (int v = 0; v < 256; v++) buf_logits[v] = 0;
    for (int d = 0; d < (int)D; d++) {
        float xv = cur_x[d];
        const int8_t* row = w_lm_head + (uint32_t)d * 256;
        for (int v = 0; v < 256; v++) buf_logits[v] += xv * (float)row[v];
    }
    for (int v = 0; v < 256; v++) buf_logits[v] *= s_lm_head;
    PF_END(PF_LMHEAD);

    seq_pos = t + 1;
}

static void bdh_reset() { seq_pos = 0; }

// ============================================================
//  Weight loading + buffer allocation (shared with host test)
// ============================================================

static bool bdh_init_from_blob(const void* blob) {
    const WeightHeader* hdr = (const WeightHeader*)blob;
    if (hdr->magic != 0x42444802) return false;

    D = hdr->d_model; nh = hdr->n_head; N = hdr->n_inner;
    n_layer = hdr->n_layer; vocab_size = hdr->vocab_size;
    nhN = nh * N;
    s_embed = hdr->embed_scale; s_encoder = hdr->encoder_scale;
    s_encoder_v = hdr->encoder_v_scale; s_decoder = hdr->decoder_scale;
    s_lm_head = hdr->lm_head_scale;

    const int8_t* data = (const int8_t*)blob + 64;
    w_embed = data;      data += (uint32_t)vocab_size * D;
    w_encoder = data;    data += (uint32_t)nh * N * D;
    w_encoder_v = data;  data += (uint32_t)nh * N * D;
    w_decoder = data;    data += (uint32_t)nhN * D;
    w_lm_head = data;

    const int8_t* encoder_in_flash = w_encoder;

    // Try to bring up the buffers twice: once with the encoder hoisted into
    // PSRAM, and if anything at all fails, again without it.
    //
    // Hoisting the encoder is the optimization — it's the hottest weight
    // tensor (every layer re-reads all nh*N*D bytes, ~13MB/token), and
    // PSRAM beats DIO flash, which is stuck at ~20MB/s because OPI PSRAM
    // claims the pins QIO would need. But it costs 2.11MB of an 8MB budget
    // the caches nearly fill, so it must degrade gracefully: a failed
    // allocation should cost speed, never boot. Worth ~12%; if you raise
    // T_MAX, expect this to be the thing that stops fitting.
    for (int attempt = 0; attempt < 2; attempt++) {
        bool hoist = (attempt == 0);

        if (hoist) {
            encoder_psram = (int8_t*)ps_malloc((size_t)nh * N * D);
            if (!encoder_psram) continue;   // no room at all — retry without
            memcpy(encoder_psram, encoder_in_flash, (size_t)nh * N * D);
            w_encoder = encoder_psram;
        } else {
            w_encoder = encoder_in_flash;
        }

        kr_cache   = (int8_t*)ps_malloc((size_t)n_layer * nh * T_MAX * N);
        kr_scale   = (float*)ps_malloc((size_t)n_layer * nh * T_MAX * sizeof(float));
        x_cache    = (float*)ps_malloc((size_t)n_layer * T_MAX * D * sizeof(float));
        cur_x      = (float*)ps_malloc(D * sizeof(float));
        cur_sparse = (float*)ps_malloc((size_t)nh * N * sizeof(float));
        cur_qr     = (float*)ps_malloc(N * sizeof(float));
        cur_xy     = (float*)ps_malloc((size_t)nh * N * sizeof(float));
        cur_ykv    = (float*)ps_malloc(D * sizeof(float));
        cur_y      = (float*)ps_malloc(D * sizeof(float));
        buf_logits = (float*)ps_malloc(256 * sizeof(float));
        rope_freqs = (float*)ps_malloc(N * sizeof(float));

        if (kr_cache && kr_scale && x_cache && cur_x && cur_sparse &&
            cur_qr && cur_xy && cur_ykv && cur_y && buf_logits && rope_freqs)
            break;   // all good

        if (!hoist) return false;   // even the lean layout doesn't fit

        // Give everything back and retry reading the encoder from flash.
        free(kr_cache);   free(kr_scale);  free(x_cache);   free(cur_x);
        free(cur_sparse); free(cur_qr);    free(cur_xy);    free(cur_ykv);
        free(cur_y);      free(buf_logits); free(rope_freqs);
        free(encoder_psram);
        kr_cache = nullptr; kr_scale = nullptr; x_cache = nullptr;
        cur_x = nullptr; cur_sparse = nullptr; cur_qr = nullptr;
        cur_xy = nullptr; cur_ykv = nullptr; cur_y = nullptr;
        buf_logits = nullptr; rope_freqs = nullptr;
        encoder_psram = nullptr;
    }

    compute_rope_freqs(rope_freqs, N, powf(2.0f, 16.0f));
    return true;
}

#ifndef ARDUINO_HOST_TEST

// ============================================================
//  Sampling
// ============================================================

static uint8_t sample_token(float temperature) {
    if (temperature != 1.0f) {
        for (int v = 0; v < 256; v++) buf_logits[v] /= temperature;
    }
    float maxv = buf_logits[0];
    for (int v = 1; v < 256; v++) if (buf_logits[v] > maxv) maxv = buf_logits[v];
    float sum = 0;
    for (int v = 0; v < 256; v++) { buf_logits[v] = expf(buf_logits[v] - maxv); sum += buf_logits[v]; }
    float r = ((float)esp_random() / (float)UINT32_MAX) * sum;
    float cumsum = 0;
    for (int v = 0; v < 256; v++) {
        cumsum += buf_logits[v];
        if (cumsum >= r) return (uint8_t)v;
    }
    return 255;
}

// ============================================================
//  Setup & REPL
// ============================================================

void setup() {
    Serial.begin(115200);
    while (!Serial) delay(10);
    Serial.println("\n=== BDH TinyStories ===");

    const void* mapped;
    spi_flash_mmap_handle_t handle;
    esp_err_t err = spi_flash_mmap(WEIGHTS_FLASH_ADDR, 7 * 1024 * 1024,
        SPI_FLASH_MMAP_DATA, &mapped, &handle);
    if (err != ESP_OK) {
        Serial.printf("ERROR: mmap failed: %d\n", err);
        while (1) delay(1000);
    }

    size_t psram_before = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);

    // Distinguish the two failure modes — they need opposite fixes.
    const WeightHeader* probe = (const WeightHeader*)mapped;
    if (probe->magic != 0x42444802) {
        Serial.printf("ERROR: bad weights magic 0x%08lX (expected 0x42444802)\n",
                      (unsigned long)probe->magic);
        Serial.println("Re-run export_weights.py (layout v2) and reflash weights.bin");
        while (1) delay(1000);
    }
    if (!bdh_init_from_blob(mapped)) {
        Serial.printf("ERROR: PSRAM allocation failed (%u bytes free, D=%lu N=%lu T_MAX=%d)\n",
                      (unsigned)psram_before, probe->d_model, probe->n_inner, T_MAX);
        Serial.println("The model doesn't fit. Lower T_MAX in the .ino and reflash.");
        while (1) delay(1000);
    }
    size_t psram_after = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);

    Serial.printf("Model: D=%lu nh=%lu N=%lu layers=%lu vocab=%lu\n", D, nh, N, n_layer, vocab_size);
    Serial.printf("PSRAM used: %.2f MB (T_MAX=%d)\n",
        (psram_before - psram_after) / (1024.0f * 1024.0f), T_MAX);
    Serial.printf("Encoder in %s%s\n",
        encoder_psram ? "PSRAM" : "flash",
        encoder_psram ? "" : " (didn't fit — expect ~2x slower)");
    Serial.printf("\nReady! Type a story prompt and press Enter.\n");
    Serial.printf("(max %d bytes total, temperature=0.8)\n\n", T_MAX);
    Serial.flush();
}

void loop() {
    if (!Serial.available()) return;

    String input = Serial.readStringUntil('\n');
    input.trim();
    if (input.length() == 0) return;

    Serial.printf("\n> %s", input.c_str());
    Serial.flush();

    bdh_reset();
    PF_RESET();
    int prompt_len = min((int)input.length(), T_MAX - 16);
    unsigned long t0 = millis();
    for (int i = 0; i < prompt_len; i++) {
        bdh_step((uint8_t)input.charAt(i));
    }
    unsigned long t_prompt = millis() - t0;

    int gen = 0;
    t0 = millis();
    while (seq_pos < T_MAX) {
        uint8_t next = sample_token(0.8f);
        gen++;
        if (next >= 32 && next < 127) Serial.write(next);
        else if (next == '\n') Serial.println();
        else Serial.printf("\\x%02x", next);
        Serial.flush();
        if (seq_pos == T_MAX) break;
        bdh_step(next);
    }
    unsigned long t_gen = millis() - t0;

    Serial.printf("\n[prompt %d B in %.1fs | %d tokens in %.1fs = %.2f tok/s]\n",
        prompt_len, t_prompt / 1000.0f, gen, t_gen / 1000.0f,
        gen * 1000.0f / (float)t_gen);

#if PROFILE
    {
        int steps = prompt_len + gen;
        uint32_t total = 0;
        for (int i = 0; i < PF_N; i++) total += pf_us[i];
        Serial.printf("[profile: %d steps, %.1f ms/step accounted]\n",
                      steps, total / 1000.0f / steps);
        for (int i = 0; i < PF_N; i++) {
            Serial.printf("  %-11s %7.1f ms/tok  %5.1f%%\n", pf_names[i],
                          pf_us[i] / 1000.0f / steps,
                          total ? 100.0f * pf_us[i] / total : 0.0f);
        }
    }
#endif
    Serial.println();
    Serial.flush();
}

#endif  // !ARDUINO_HOST_TEST
