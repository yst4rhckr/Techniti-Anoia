const std: type = @import("std");
const math: type = std.math;
const assert: fn (bool) void = std.debug.assert;

const BS: usize = 64;

pub inline fn min(comptime T: type, a: T, b: T) T {
    return if (a < b) a else b;
}

pub const Aljabr: type = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Aljabr {
        return .{
            .allocator = allocator,
        };
    }

    pub fn matmul(self: Aljabr, X: []const f32, Y: []const f32, M: usize, N: usize, K: usize) ![]f32 {
        assert(X.len == M * K);
        assert(Y.len == K * N);

        const Z = try self.allocator.alloc(f32, M * N);
        @memset(Z, 0.0);

        var ii: usize = 0;
        while (ii < M) : (ii += BS) {
            var kk: usize = 0;
            while (kk < K) : (kk += BS) {
                var jj: usize = 0;
                while (jj < N) : (jj += BS) {
                    const i_max: usize = min(usize, ii + BS, M);
                    const k_max: usize = min(usize, kk + BS, K);
                    const j_max: usize = min(usize, jj + BS, N);

                    for (ii..i_max) |i| {
                        for (kk..k_max) |k| {
                            const x: f32 = X[i * K + k];
                            for (jj..j_max) |j| {
                                Z[i * N + j] += x * Y[k * N + j];
                            }
                        }
                    }
                }
            }
        }

        return Z;
    }

    pub fn transpose(self: Aljabr, A: []const f32, rows: usize, cols: usize) ![]f32 {
        assert(A.len == cols * rows);

        const AT = try self.allocator.alloc(f32, A.len);
        @memset(AT, 0.0);

        for (0..rows) |i| {
            for (0..cols) |j|
                AT[j * rows + i] = A[i * cols + j];
        }

        return AT;
    }

    pub fn scale(self: Aljabr, A: []const f32, factor: f32) ![]f32 {
        const calc = try self.allocator.alloc(f32, A.len);

        for (A, 0..) |value, i|
            calc[i] = value * factor;
        return calc;
    }
};
