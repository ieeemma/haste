const std = @import("std");

const Lexer = @import("Lexer.zig");

pub fn main() !void {
    std.log.info("All your codebase are belong to us.", .{});
}

test {
    _ = Lexer;
}
