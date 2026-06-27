#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NEG_INF -1e30f

#if defined(__GNUC__)
#if defined(__ANDROID__) && defined(__aarch64__)
#pragma GCC optimize("Ofast")

#elif defined(__linux__) && defined(__x86_64__)
#pragma GCC optimize("Ofast")
// #pragma GCC target("avx2,fma")

#elif defined(__linux__) && defined(__i386__)
#pragma GCC optimize("Ofast")
#pragma GCC target("sse4.2")
#endif
#endif

#if !defined(__STDC_VERSION__) || __STDC_VERSION__ < 199901L
#define restrict
#endif

void sigmoid(float *restrict x, size_t size) {
  for (size_t i = 0; i < size; i++) {
    if (x[i] >= 0) {
      float z = expf(-x[i]);
      x[i] = 1.0f / (1.0f + z);
    } else {
      float z = expf(x[i]);
      x[i] = z / (1.0f + z);
    }
  }
}

void sigmoid_derivative(float *restrict y, float *restrict dy, size_t size) {
  for (size_t i = 0; i < size; i++)
    dy[i] = y[i] * (1.0f - y[i]);
}

void relu_vec(float *restrict z, float *restrict dz, size_t size) {
  for (size_t i = 0; i < size; i++)
    dz[i] = (z[i] > 0.0f) ? z[i] : 0.0f;
}

void leaky_relu_vec(float *restrict w, float *restrict dw, float alpha,
                    size_t size) {
  for (size_t i = 0; i < size; i++)
    dw[i] = (w[i] > 0.0f) ? w[i] : alpha * w[i];
}

void relu_derivative_vec(float *restrict f, float *restrict df, size_t size) {
  for (size_t i = 0; i < size; i++)
    df[i] = (f[i] > 0.0f) ? 1.0f : 0.0f;
}

void leaky_relu_derivative_vec(float *restrict u, float *restrict du,
                               float alpha, size_t size) {
  for (size_t i = 0; i < size; i++)
    du[i] = (u[i] > 0.0f) ? 1.0f : alpha;
}

void softmax(float *restrict o, size_t size) {
  if (size == 0)
    return;
  float max = o[0];

  for (size_t i = 1; i < size; i++) {
    if (o[i] > max)
      max = o[i];
  }

  float sum_exp = 0.0f;

  for (size_t j = 0; j < size; j++) {
    o[j] = expf(o[j] - max);
    sum_exp += o[j];
  }

  if (sum_exp < 1e-12f)
    sum_exp = 1e-12f;

  for (size_t k = 0; k < size; k++) {
    o[k] /= sum_exp;
  }
}

void softmax_rows(float *restrict A, size_t rows, size_t cols) {
  for (size_t i = 0; i < rows; i++)
    softmax(A + i * cols, cols);
}

#define BS 64 // tamanho do bloco
#define MIN(a, b) ((a) < (b) ? (a) : (b))
void matmul(const float *restrict X, const float *restrict Y, float *restrict Z,
            size_t M, size_t N, size_t K) {
  // zerar Z
  for (size_t i = 0; i < M * N; i++)
    Z[i] = 0.0f;

  for (size_t ii = 0; ii < M; ii += BS)
    for (size_t kk = 0; kk < K; kk += BS)
      for (size_t jj = 0; jj < N; jj += BS) {
        size_t i_max = MIN(ii + BS, M);
        size_t k_max = MIN(kk + BS, K);
        size_t j_max = MIN(jj + BS, N);

        for (size_t i = ii; i < i_max; i++)
          for (size_t k = kk; k < k_max; k++) {
            float x = X[i * K + k];
            for (size_t j = jj; j < j_max; j++)
              Z[i * N + j] += x * Y[k * N + j];
          }
      }
}

void transpose(const float *restrict A, float *restrict AT, size_t rows,
               size_t cols) {
  for (size_t i = 0; i < rows; i++)
    for (size_t j = 0; j < cols; j++)
      AT[j * rows + i] = A[i * cols + j];
}

void scale(float *restrict A, size_t size, float factor) {
  for (size_t i = 0; i < size; i++)
    A[i] *= factor;
}

void self_attention(const float *X, const float *Wq, const float *Wk,
                    const float *Wv, float *Q, float *K, float *V, float *KT,
                    float *scores, float *output, size_t seq_len,
                    size_t d_model, size_t d_k, size_t d_head, size_t d_v) {
  matmul(X, Wq, Q, seq_len, d_k, d_model);
  matmul(X, Wk, K, seq_len, d_k, d_model);
  matmul(X, Wv, V, seq_len, d_v, d_model);

  transpose(K, KT, seq_len, d_k);
  matmul(Q, KT, scores, seq_len, seq_len, d_k);
  scale(scores, seq_len * seq_len, 1.0f / sqrtf((float)d_head));
  softmax_rows(scores, seq_len, seq_len);
  matmul(scores, V, output, seq_len, d_v, seq_len);
}

void add_cause_mask(float *restrict scores, size_t seq_len) {
  for (size_t i = 0; i < seq_len; i++) {
    for (size_t j = 0; j < seq_len; j++) {
      scores[i * seq_len + j] += (j > i) ? -NEG_INF : 0.0f;
    }
  }
}

void multi_head_attention(const float *restrict X, const float *restrict Wq,
                          const float *restrict Wk, const float *restrict Wv,
                          const float *restrict Wo, float *restrict output,
                          float *restrict temp_buffer, size_t seq_len,
                          size_t d_model, size_t num_heads) {
  if (d_model % num_heads != 0)
    return;

  size_t d_head = d_model / num_heads;

  // Buffers principais
  float *Q_all = temp_buffer;
  float *K_all = Q_all + seq_len * d_model;
  float *V_all = K_all + seq_len * d_model;
  float *concat = V_all + seq_len * d_model;

  // head
  float *Q_h = concat + seq_len * d_model;
  float *K_h = Q_h + seq_len * d_head;
  float *V_h = K_h + seq_len * d_head;
  float *KT_h = V_h + seq_len * d_head;
  float *scores = KT_h + d_head * seq_len;

  // projeçes lineares
  matmul(X, Wq, Q_all, seq_len, d_model, d_model);
  matmul(X, Wk, K_all, seq_len, d_model, d_model);
  matmul(X, Wv, V_all, seq_len, d_model, d_model);

  for (size_t h = 0; h < num_heads; h++) {
    // extrair blocos
    for (size_t i = 0; i < seq_len; i++) {
      memcpy(Q_h + i * d_head, Q_all + i * d_model + h * d_head,
             d_head * sizeof(float));

      memcpy(K_h + i * d_head, K_all + i * d_model + h * d_head,
             d_head * sizeof(float));

      memcpy(V_h + i * d_head, V_all + i * d_model + h * d_head,
             d_head * sizeof(float));
    }

    // KT = transpose(K_h)
    transpose(K_h, KT_h, seq_len, d_head);

    // scores = Q_h @ KT_h
    matmul(Q_h, KT_h, scores, seq_len, seq_len, d_head);

    // escala
    scale(scores, seq_len * seq_len, 1.0f / sqrtf((float)d_head));

    add_cause_mask(scores, seq_len);

    // softmax linha a linha
    softmax_rows(scores, seq_len, seq_len);

    // head_output = scores @ V_h
    matmul(scores, V_h,
           Q_h, // reaproveitar Q_h como tamanho
           seq_len, d_head, seq_len);

    // copiar concat
    for (size_t i = 0; i < seq_len; i++) {
      memcpy(concat + i * d_model + h * d_head, Q_h + i * d_head,
             d_head * sizeof(float));
    }
  }

  // proeja,cãp
  matmul(concat, Wo, output, seq_len, d_model, d_model);
}

void rmsnorm(float *restrict o, const float *restrict x,
             const float *restrict weight, size_t size) {
  float sum_square = 0.0f;

  for (size_t j = 0; j < size; j++)
    sum_square += x[j] * x[j];

  sum_square /= (float)size;
  sum_square += 1e-5f;

  float inv_rms = 1.0f / sqrtf(sum_square);

  for (size_t j = 0; j < size; j++)
    o[j] = weight[j] * (x[j] * inv_rms);
}

int main() {
  size_t seq_len = 2;
  size_t d_model = 4;
  size_t num_heads = 2;
  size_t d_head = d_model / num_heads;

  // entradinhas
  float X[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

  // bias
  float Wq[16] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f,
                  0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f};

  float Wk[16] = {0.2f, 0.1f, 0.4f, 0.3f, 0.6f, 0.5f, 0.8f, 0.7f,
                  1.0f, 0.9f, 1.2f, 1.1f, 1.4f, 1.3f, 1.6f, 1.5f};

  float Wv[16] = {0.5f, 0.4f, 0.3f, 0.2f, 0.1f, 0.0f, 0.9f, 0.8f,
                  0.7f, 0.6f, 0.5f, 0.4f, 0.3f, 0.2f, 0.1f, 0.0f};

  // Wo identidade
  float Wo[16] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                  0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};

  // saida na heap
  float *output_multi = calloc(seq_len * d_model, sizeof(float));
  if (!output_multi) {
    printf("Erro de alocação output\n");
    return 1;
  }

  // Isso faz alguma coisa
  size_t buf_size = seq_len * d_model * 3  // Q_all K_all V_all
                    + seq_len * d_model    // concat
                    + seq_len * d_head * 3 // Q_h K_h V_h
                    + seq_len * d_head     // KT_h
                    + seq_len * seq_len;   // scores

  float *temp_buffer = malloc(buf_size * sizeof(float));
  if (!temp_buffer) {
    printf("Erro de alocação buffer\n");
    free(output_multi);
    return 1;
  }

  // executar MHA
  multi_head_attention(X, Wq, Wk, Wv, Wo, output_multi, temp_buffer, seq_len,
                       d_model, num_heads);

  printf("Output:\n");
  for (size_t i = 0; i < seq_len; i++) {
    for (size_t j = 0; j < d_model; j++) {
      printf("%.6f ", output_multi[i * d_model + j]);
    }
    printf("\n");
  }

  free(temp_buffer);
  free(output_multi);

  return 0;
}
