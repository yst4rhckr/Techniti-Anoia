const std = @import("std");

const analyze = @import("../analyzer/analyze.zig");
const printbar = @import("../analyzer/printbar.zig").printBar;
const Aljabr = @import("../math/linalg.zig").Aljabr;
const NeuralNet = @import("../math/ops.zig").NeuralNet;
const Activation = @import("../nn/activation.zig").Activation;

pub fn teste(init: std.process.Init) anyerror!void {
    const base_allocator = init.gpa;

    var arena = std.heap.ArenaAllocator.init(base_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const ops = Aljabr.init(allocator);
    const activation = Activation.init(allocator);
    const neural = NeuralNet.init(allocator);
    const TensorStats = analyze.TensorStats;

    const mock_a: []const f32 = &[_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const mock_y: []const f32 = &[_]f32{ 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 };
    const mock_x: []const f32 = &[_]f32{ -2.0, -0.5, 0.0, 1.5, 3.0 };
    const mock_dx: []const f32 = &[_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0 };
    const mock_logits: []const f32 = &[_]f32{ 2.0, 1.0, 0.1 };

    const seq_len: usize = 2;
    const d_model: usize = 4;
    const num_heads: usize = 2;

    const mha_x: []const f32 = &[_]f32{ 1.0, 0.0, 2.0, -1.0, 0.0, 1.0, 0.5, 1.0 };
    const identity_w: []const f32 = &[_]f32{ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };

    const result_transpose = try ops.transpose(mock_a, 2, 3);
    const result_matmul = try ops.matmul(mock_a, mock_y, 2, 2, 3);
    const result_scale = try ops.scale(result_matmul, 0.5);

    const result_sig_f = try activation.sigmoid_forward(mock_x);
    const result_relu_f = try activation.relu_forward(mock_x);
    const result_sig_b = try activation.sigmoid_backward(result_sig_f, mock_dx);
    const result_relu_b = try activation.relu_backward(mock_x, mock_dx);
    const result_softmax = try activation.softmax(mock_logits);

    const result_mha = try neural.multi_head_attention(mha_x, identity_w, identity_w, identity_w, identity_w, seq_len, d_model, num_heads);

    const stats_transpose = TensorStats.analyze(result_transpose);
    const stats_matmul = TensorStats.analyze(result_matmul);
    const stats_scale = TensorStats.analyze(result_scale);
    const stats_sig_f = TensorStats.analyze(result_sig_f);
    const stats_relu_f = TensorStats.analyze(result_relu_f);
    const stats_sig_b = TensorStats.analyze(result_sig_b);
    const stats_relu_b = TensorStats.analyze(result_relu_b);
    const stats_softmax = TensorStats.analyze(result_softmax);
    const stats_mha = TensorStats.analyze(result_mha);

    std.debug.print("⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯\n", .{});
    std.debug.print("TELEMETRIA\n", .{});
    std.debug.print("⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯\n", .{});

    TensorStats.printMetric("ops.transpose", stats_transpose);
    TensorStats.printMetric("ops.matmul", stats_matmul);
    TensorStats.printMetric("ops.scale", stats_scale);
    std.debug.print("⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯\n", .{});
    TensorStats.printMetric("activation.sigmoid_f", stats_sig_f);
    TensorStats.printMetric("activation.relu_f", stats_relu_f);
    TensorStats.printMetric("activation.sigmoid_b", stats_sig_b);
    TensorStats.printMetric("activation.relu_b", stats_relu_b);
    TensorStats.printMetric("activation.softmax", stats_softmax);
    std.debug.print("⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯\n", .{});
    TensorStats.printMetric("neural.multi_head_attn", stats_mha);

    std.debug.print("⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯\n", .{});
    std.debug.print("DENSIDADE E ESPARSIDADE DO TENSOR\n", .{});
    std.debug.print("⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯\n", .{});

    printbar("ops.transpose", stats_transpose.zeros_pct);
    printbar("ops.matmul", stats_matmul.zeros_pct);
    printbar("ops.scale", stats_scale.zeros_pct);

    std.debug.print("________________________________________________________\n", .{});
    printbar("activation.sigmoid_f", stats_sig_f.zeros_pct);
    printbar("activation.relu_f", stats_relu_f.zeros_pct);
    printbar("activation.sigmoid_b", stats_sig_b.zeros_pct);
    printbar("activation.relu_b", stats_relu_b.zeros_pct);
    printbar("activation.softmax", stats_softmax.zeros_pct);
    std.debug.print("________________________________________________________\n", .{});
    printbar("neural.multi_head_attn", stats_mha.zeros_pct);
    std.debug.print("________________________________________________________\n", .{});
}
