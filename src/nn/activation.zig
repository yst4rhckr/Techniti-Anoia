const std: type = @import("std");
const math: type = std.math;
const assert: fn (bool) void = std.debug.assert;

const EPSILON: comptime_float = 1e-12;

pub const Activation = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Activation {
        return .{
            .allocator = allocator,
        };
    }

    pub fn sigmoid_forward(self: Activation, x: []const f32) ![]f32 {
        const forward = try self.allocator.alloc(f32, x.len);
        for (x, 0..) |value, i| {
            if (value >= 0.0) {
                const z: f32 = math.exp(-value);
                forward[i] = 1.0 / (1.0 + z);
            } else {
                const z: f32 = math.exp(value);
                forward[i] = z / (1.0 + z);
            }
        }

        return forward;
    }

    pub fn sigmoid_backward(self: Activation, x: []const f32, dx: []const f32) ![]f32 {
        const backward = try self.allocator.alloc(f32, x.len);
        for (x, 0..) |_, i|
            backward[i] = dx[i] * x[i] * (1.0 - x[i]);
        return backward;
    }

    pub fn relu_forward(self: Activation, x: []const f32) ![]f32 {
        const forward: []f32 = try self.allocator.alloc(f32, x.len);
        for (x, 0..) |value, i|
            forward[i] = if (value > 0.0) value else 0.0;
        return forward;
    }

    pub fn relu_backward(self: Activation, x: []const f32, dx: []const f32) ![]f32 {
        const backward = try self.allocator.alloc(f32, x.len);
        for (x, 0..) |value, i|
            backward[i] = if (value > 0.0) dx[i] else 0.0;
        return backward;
    }

    pub fn leaky_relu(self: Activation, x: []const f32, a: f32) ![]f32 {
        const leaky: []f32 = try self.allocator.alloc(f32, x.len);

        for (x, 0..) |value, i|
            leaky[i] = if (value > 0.0) value else a * value;
        return leaky;
    }

    pub fn leaky_relu_b(self: Activation, x: []const f32, dx: []const f32, a: f32) ![]f32 {
        const leaky_b = try self.allocator.alloc(f32, x.len);

        for (x, 0..) |value, i|
            leaky_b[i] = if (value > 0.0) dx[i] else a * dx[i];
        return leaky_b;
    }

    pub fn softmax(self: Activation, output: []const f32) ![]f32 {
        const soft = try self.allocator.alloc(f32, output.len);

        if (output.len == 0) return soft;

        var max: f32 = output[0];
        for (output) |value| {
            if (value > max) max = value;
        }

        var sum: f32 = 0.0;
        for (output, 0..) |value, j| {
            soft[j] = math.exp(value - max);
            sum += soft[j];
        }

        if (sum < EPSILON) sum = EPSILON;

        for (soft, 0..) |_, k|
            soft[k] /= sum;

        return soft;
    }

    pub fn softmax_rows(self: Activation, A: []const f32, cols: usize, rows: usize) ![]f32 {
        assert(A.len == cols * rows);
        const softrows = try self.allocator.alloc(f32, A.len);

        var r: usize = 0;
        while (r < rows) : (r += 1) {
            const input: []const f32 = A[r * cols .. r * cols + cols];
            const output: []f32 = try self.softmax(input);
            @memcpy(softrows[r * cols .. r * cols + cols], output);
        }
        return softrows;
    }
};
