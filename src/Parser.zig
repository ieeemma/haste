const std = @import("std");

const Lexer = @import("Lexer.zig");
const Token = Lexer.Token;
const ir = @import("ir.zig");

const Parser = @This();

allocator: std.mem.Allocator,

tokens: std.MultiArrayList(Token),
idx: u32 = 0,

src: []const u8,
src_idx: u32 = 0,

mod: struct {
    imports: std.ArrayListUnmanaged([]const u8) = .{},
    ir: std.ArrayListUnmanaged(ir.IR) = .{},
    locs: std.ArrayListUnmanaged(u32) = .{},
} = .{},

err: struct {
    msg: ?[]const u8 = undefined,
    idx: u32 = 0,
} = .{},

const E = std.fmt.AllocPrintError || error{ParseError};

pub fn init(allocator: std.mem.Allocator, src: []const u8) !Parser {
    return Parser{
        .allocator = allocator,
        .tokens = try Lexer.tokenize(allocator, src),
        .src = src,
    };
}

pub fn deinit(self: *Parser) void {
    self.tokens.deinit(self.allocator);
    if (self.err.msg) |msg| self.allocator.free(msg);
}

pub fn parseFile(self: *Parser) E!ir.Module {

    // TODO: parse imports here

    while (!self.eof()) {
        try self.parseExpr();
    }

    return ir.Module{
        .imports = self.mod.imports.toOwnedSlice(self.allocator),
        .ir = self.mod.ir.toOwnedSlice(self.allocator),
        .locs = self.mod.locs.toOwnedSlice(self.allocator),
    };
}

fn parseExpr(self: *Parser) E!void {
    // TODO: parse control flow here

    return self.parseOp(0);
}

const Prec = struct {
    op: []const u8,
    insn: ir.Insn,
};

const precedence = [_][]const Prec{
    &.{
        .{ .op = "==", .insn = .eq },
        .{ .op = "!=", .insn = .neq },
    },
    &.{
        .{ .op = "<", .insn = .lt },
        .{ .op = "<=", .insn = .le },
        .{ .op = ">", .insn = .gt },
        .{ .op = ">=", .insn = .ge },
    },
    &.{
        .{ .op = "+", .insn = .add },
        .{ .op = "-", .insn = .sub },
    },
    &.{ .{ .op = "*", .insn = .mul }, .{ .op = "/", .insn = .div } },
};

fn parseOp(self: *Parser, comptime prec_idx: usize) E!void {

    // Parse lhs
    if (prec_idx + 1 < precedence.len) {
        try self.parseOp(prec_idx + 1);
    } else {
        try self.parseCall();
    }

    // If the next token is an operator...
    if (try self.peek()) |tok| {
        if (tok.tag == .op) {
            const got = self.src[self.src_idx .. self.src_idx + tok.len];

            for (precedence[prec_idx]) |prec| {
                if (std.mem.eql(u8, prec.op, got)) {

                    // Skip past this token
                    self.advance(tok);

                    // Parse rhs
                    try self.parseOp(prec_idx);

                    // Emit op
                    try self.emit(.{ .insn = prec.insn });
                }
            }
        }
    }
}

fn parseCall(self: *Parser) E!void {
    try self.parseAtom();

    // If next is opening paren, continue parsing `<expression> COMMA`
    // until closing paren
    if (try self.getIf(.lparen)) |_| {
        var nargs: u32 = 0;

        while (true) {
            try self.parseExpr();
            nargs += 1;
            switch ((try self.expect(&.{ .rparen, .comma })).tag) {
                .rparen => break,
                .comma => if (try self.getIf(.rparen)) |_| break,
                else => unreachable,
            }
        }

        try self.emit(.{ .insn = .call });
        try self.emit(.{ .uint = nargs });
    }
}

fn parseAtom(self: *Parser) E!void {
    const tok = try self.get();
    switch (tok.tag) {

        // Parenthesized expression
        .lparen => {
            try self.parseExpr();
            _ = try self.expect(&.{.rparen});
        },

        // Block
        .lbrace => {
            // Continue parsing `<expression> SEMICOLON` until closing brace
            // Emit `pop` after all but last expression
            while (true) {
                try self.parseExpr();
                _ = try self.expect(&.{.semi_colon});
                if (try self.getIf(.rbrace)) |_| {
                    break;
                } else {
                    try self.emit(.{ .insn = .pop });
                }
            }
        },

        .symbol => {
            // TODO: there needs to be actual logic here
            try self.emit(.{ .insn = .load });
            try self.emit(.{ .uint = 0 });
        },

        // Integer literal
        .int => {
            // Parse integer, then emit `push_int` instruction
            // TODO: + and - prefixes
            var n: i32 = 0;
            var i: usize = self.src_idx - tok.len;
            while (i < self.src_idx) : (i += 1) {
                const ch = self.src[i];
                n = n * 10 + (ch - '0');
            }
            try self.emit(.{ .insn = .push_int });
            try self.emit(.{ .int = n });
        },

        // TODO: parse other literals here

        else => |tag| return self.parseError("Unexpected token {}", .{tag}),
    }
}

inline fn emit(self: *Parser, data: ir.IR) E!void {
    try self.mod.ir.append(self.allocator, data);
}

fn eof(self: Parser) bool {
    return self.idx >= self.tokens.len;
}

fn peek(self: *Parser) E!?Token {
    while (!self.eof()) {
        const tok = self.tokens.get(self.idx);
        switch (tok.tag) {
            .ws, .comment => self.advance(tok),
            .invalid => return self.parseError("Invalid token", .{}),
            else => return tok,
        }
    }
    return null;
}

fn advance(self: *Parser, tok: Token) void {
    self.idx += 1;
    self.src_idx += tok.len;
}

fn get(self: *Parser) E!Token {
    if (try self.peek()) |tok| {
        self.advance(tok);
        return tok;
    } else {
        return self.parseError("Unexpected EOF", .{});
    }
}

fn getIf(self: *Parser, comptime tag: Token.Tag) E!?Token {
    if (try self.peek()) |tok| {
        if (tok.tag == tag) {
            self.advance(tok);
            return tok;
        }
    }
    return null;
}

fn expect(self: *Parser, comptime expected: []const Token.Tag) E!Token {
    const tok = try self.get();
    for (expected) |tag| {
        if (tok.tag == tag) return tok;
    }

    // TODO: why does this type error?
    // return self.parseError("Invalid token: expected one of {any}, but got {}", .{ expect, tok.tag });

    return error.ParseError;
}

fn parseError(self: *Parser, comptime fmt: []const u8, args: anytype) E {
    // TODO: add source line
    self.err = .{
        .msg = try std.fmt.allocPrint(self.allocator, fmt, args),
        .idx = self.src_idx,
    };
    return error.ParseError;
}

fn testParserSuccess(comptime src: []const u8, comptime expected: []const ir.IR) !void {
    const allocator = std.testing.allocator;

    var parser = try Parser.init(allocator, src);
    defer parser.deinit();
    errdefer {
        parser.mod.imports.deinit(allocator);
        parser.mod.ir.deinit(allocator);
        parser.mod.locs.deinit(allocator);
    }

    const mod = try parser.parseFile();
    defer allocator.free(mod.imports);
    defer allocator.free(mod.ir);
    defer allocator.free(mod.locs);

    // std.debug.print("\n", .{});
    // for (mod.ir) |x, i| {
    //     std.debug.print("{} {}\n", .{ i, x });
    // }

    try std.testing.expectEqualSlices(ir.IR, expected, mod.ir);
}

test "operator precedence" {
    try testParserSuccess(
        "1 + 2 * 3",
        &.{
            .{ .insn = .push_int },
            .{ .int = 1 },
            .{ .insn = .push_int },
            .{ .int = 2 },
            .{ .insn = .push_int },
            .{ .int = 3 },
            .{ .insn = .mul },
            .{ .insn = .add },
        },
    );
}

test "function call" {
    try testParserSuccess(
        "1(2, 3)",
        &.{
            .{ .insn = .push_int },
            .{ .int = 1 },
            .{ .insn = .push_int },
            .{ .int = 2 },
            .{ .insn = .push_int },
            .{ .int = 3 },
            .{ .insn = .call },
            .{ .uint = 2 },
        },
    );
}
