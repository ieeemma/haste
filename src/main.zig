const std = @import("std");

const Lexer = @import("Lexer.zig");
const Parser = @import("Parser.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var parser = try Parser.init(allocator, "98 + 100(101, 102)");
    defer parser.deinit();
    errdefer {
        parser.mod.imports.deinit(allocator);
        parser.mod.ir.deinit(allocator);
        parser.mod.locs.deinit(allocator);
    }

    const mod = parser.parseFile() catch |err| switch (err) {
        error.ParseError => {
            std.debug.print("{s}\n", .{parser.err.msg});
            return err;
        },
        else => return err,
    };

    defer allocator.free(mod.imports);
    defer allocator.free(mod.ir);
    defer allocator.free(mod.locs);

    for (mod.ir) |insn| {
        std.debug.print("{}\n", .{insn});
    }
}

test {
    _ = Lexer;
}
