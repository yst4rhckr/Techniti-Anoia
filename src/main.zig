const std = @import("std");

const test_mod = @import("test/test.zig");

pub fn main(init: std.process.Init) !void {
    try test_mod.teste(init);
}
