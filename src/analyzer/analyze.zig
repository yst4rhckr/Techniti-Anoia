const std = @import("std");

pub const TensorStats = struct {
    mean: f32,
    l2_norm: f32,
    zeros_pct: f32,
    has_nan_inf: bool,

    pub fn analyze(data: []const f32) TensorStats {
        if (data.len == 0) return .{ .mean = 0, .l2_norm = 0, .zeros_pct = 0, .has_nan_inf = false };

        var sum: f64 = 0;
        var sum_sq: f64 = 0;
        var zero_count: usize = 0;
        var nan_inf = false;

        for (data) |val| {
            if (std.math.isNan(val) or std.math.isInf(val)) {
                nan_inf = true;
                continue;
            }
            sum += val;
            sum_sq += @as(f64, val) * @as(f64, val);
            if (val == 0.0) zero_count += 1;
        }

        const len_f = @as(f64, @floatFromInt(data.len));
        return .{
            .mean = @as(f32, @floatCast(sum / len_f)),
            .l2_norm = @as(f32, @floatCast(std.math.sqrt(sum_sq))),
            .zeros_pct = @as(f32, @floatCast((@as(f64, @floatFromInt(zero_count)) / len_f) * 100.0)),
            .has_nan_inf = nan_inf,
        };
    }

    pub fn printMetric(label: []const u8, stats: TensorStats) void {
        const status = if (stats.has_nan_inf) "NaN" else "Ok";
        std.debug.print("  {s: <22} | Média: {d:>6.3} | Norma L2: {d:>6.3} | Zeros: {d:>5.1}% | {s}\n", .{ label, stats.mean, stats.l2_norm, stats.zeros_pct, status });
    }
};
