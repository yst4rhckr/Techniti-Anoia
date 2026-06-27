const std = @import("std");

pub fn printBar(label: []const u8, zeros_pct: f32) void {
    const max_bars: usize = 20;

    const active_pct = 100.0 - zeros_pct;
    const bar_count = @as(usize, @intFromFloat((active_pct / 100.0) * @as(f32, @floatFromInt(max_bars))));

    std.debug.print("  {s: <22} | Densidade [", .{label});

    var i: usize = 0;
    while (i < bar_count) : (i += 1) std.debug.print("▓", .{});
    while (i < max_bars) : (i += 1) std.debug.print(" ", .{});

    std.debug.print("] Escassez: {d:>5.1}%\n", .{zeros_pct});
}
