const std = @import("std");
const math: type = std.math;
const assert: fn (bool) void = std.debug.assert;

const Activation: type = @import("../nn/activation.zig").Activation;
const Aljabr: type = @import("linalg.zig").Aljabr;

const EPSILON: f32 = @as(f32, 1e-5);
const NEGINF: f32 = -math.inf(f32);

pub const NeuralNet = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) NeuralNet {
        return .{
            .allocator = allocator,
        };
    }

    pub fn rmsnormalization(self: NeuralNet, X: []const f32, Weight: []const f32, size: usize) ![]f32 {
        assert(X.len >= size);
        assert(Weight.len >= size);

        const rmsnorm = try self.allocator.alloc(f32, size);

        var sum_square: f32 = 0.0;

        for (0..size) |j| sum_square += X[j] * X[j];

        sum_square /= @floatFromInt(size);
        sum_square += EPSILON;

        const invertRMS: f32 = 1.0 / math.sqrt(sum_square);
        for (0..size) |j| rmsnorm[j] = Weight[j] * (X[j] * invertRMS);

        return rmsnorm;
    }

    pub fn add_cause_mask(self: NeuralNet, score: []const f32, seq_len: usize) ![]f32 {
        assert(score.len == seq_len * seq_len);
        const mask = try self.allocator.alloc(f32, score.len);

        for (0..seq_len) |i| {
            for (0..seq_len) |j|
                mask[i * seq_len + j] = if (j > i) NEGINF else score[i * seq_len + j];
        }

        return mask;
    }

    pub fn self_attention(self: NeuralNet, X: []const f32, Wq: []const f32, Wk: []const f32, Wv: []const f32, seq_len: usize, d_model: usize, d_k: usize, d_head: usize, d_v: usize) ![]f32 {
        assert(X.len == seq_len * d_model);
        assert(Wq.len == d_model * d_k);
        assert(Wk.len == d_model * d_k);
        assert(Wv.len == d_model * d_v);

        const aljabr = Aljabr.init(self.allocator);
        const act = Activation.init(self.allocator);

        // Q = X @ Wq
        const Q = try aljabr.matmul(X, Wq, seq_len, d_k, d_model);
        // K = X @ Wk
        const K = try aljabr.matmul(X, Wk, seq_len, d_k, d_model);
        // V = X @ Wv
        const V = try aljabr.matmul(X, Wv, seq_len, d_v, d_model);

        // KT = transpose(K)
        const KT = try aljabr.transpose(K, seq_len, d_k);

        // scores -> Q @ KT
        const scores = try aljabr.matmul(Q, KT, seq_len, seq_len, d_k);

        // factor -> 1.0 / math.sqrt(d_head)
        const factor = 1.0 / math.sqrt(@as(f32, @floatFromInt(d_head)));
        const scaled = try aljabr.scale(scores, factor);

        const masked = try self.add_cause_mask(scaled, seq_len);
        const weights = try act.softmax(masked, seq_len, seq_len);

        const output = try aljabr.matmul(weights, V, seq_len, d_v, seq_len);

        return output;
    }

    pub fn multi_head_attention(self: NeuralNet, X: []const f32, Wq: []const f32, Wk: []const f32, Wv: []const f32, Wo: []const f32, seq_len: usize, d_model: usize, n_heads: usize) ![]f32 {
        std.debug.assert(d_model % n_heads == 0);
        std.debug.assert(X.len == seq_len * d_model);
        std.debug.assert(Wq.len == d_model * d_model);
        std.debug.assert(Wk.len == d_model * d_model);
        std.debug.assert(Wv.len == d_model * d_model);
        std.debug.assert(Wo.len == d_model * d_model);

        const d_head = d_model / n_heads;

        const aljabr = Aljabr.init(self.allocator);
        const act = Activation.init(self.allocator);

        const Q_all = try aljabr.matmul(X, Wq, seq_len, d_model, d_model);
        const K_all = try aljabr.matmul(X, Wk, seq_len, d_model, d_model);
        const V_all = try aljabr.matmul(X, Wv, seq_len, d_model, d_model);

        const concat = try self.allocator.alloc(f32, seq_len * d_model);

        // buffers temporários
        const Q_h = try self.allocator.alloc(f32, seq_len * d_head);
        const K_h = try self.allocator.alloc(f32, seq_len * d_head);
        const V_h = try self.allocator.alloc(f32, seq_len * d_head);

        const scale_factor = 1.0 / math.sqrt(@as(f32, @floatFromInt(d_head)));

        var h: usize = 0;
        while (h < n_heads) : (h += 1) {
            for (0..seq_len) |i| {
                const src_offset = i * d_model + h * d_head;
                const dest_offset = i * d_head;

                @memcpy(Q_h[dest_offset .. dest_offset + d_head], Q_all[src_offset .. src_offset + d_head]);
                @memcpy(K_h[dest_offset .. dest_offset + d_head], K_all[src_offset .. src_offset + d_head]);
                @memcpy(V_h[dest_offset .. dest_offset + d_head], V_all[src_offset .. src_offset + d_head]);
            }

            const KT_h = try aljabr.transpose(K_h, seq_len, d_head);
            const raw_scores = try aljabr.matmul(Q_h, KT_h, seq_len, seq_len, d_head);
            const scaled_scores = try aljabr.scale(raw_scores, scale_factor);
            const masked_scores = try self.add_cause_mask(scaled_scores, seq_len);
            const attention_weights = try act.softmax_rows(masked_scores, seq_len, seq_len);

            const head_output = try aljabr.matmul(attention_weights, V_h, seq_len, d_head, seq_len);

            for (0..seq_len) |i| {
                const dest_offset = i * d_model + h * d_head;
                const src_offset = i * d_head;

                @memcpy(concat[dest_offset .. dest_offset + d_head], head_output[src_offset .. src_offset + d_head]);
            }
        }

        const output = try aljabr.matmul(concat, Wo, seq_len, d_model, d_model);

        return output;
    }
};
