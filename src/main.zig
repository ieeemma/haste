const std = @import("std");

const Lexer = @import("Lexer.zig");
const Parser = @import("Parser.zig");

pub fn main() !u8 {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();

    const stderr = std.io.getStdErr().writer();

    _ = args.skip();

    const path = args.next() orelse {
        _ = try stderr.write("Error: no file name provided\n");
        return 1;
    };
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    const src = try file.readToEndAlloc(allocator, std.math.maxInt(usize));
    defer allocator.free(src);

    var p = try Parser.init(allocator, src);
    defer p.deinit();

    const mod = p.parse() catch |err| switch (err) {
        error.ParseError => {
            _ = try stderr.print("Error: {s}\n", .{p.err.msg});
            return 1;
        },
        else => return err,
    };
    defer allocator.free(mod.ir);

    for (mod.ir) |ir| {
        _ = try std.io.getStdOut().writer().print("{}\n", .{ir});
    }

    return 0;
}

test {
    _ = Lexer;
    _ = Parser;
}
