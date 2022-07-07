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

break_offsets: std.ArrayListUnmanaged(usize) = .{},
continue_offsets: std.ArrayListUnmanaged(usize) = .{},

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
    self.break_offsets.deinit(self.allocator);
    self.continue_offsets.deinit(self.allocator);
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
    if (try self.getIf(.kw_if)) |_| {
        try self.parseIf();
    } else if (try self.getIf(.kw_while)) |_| {
        try self.parseWhile();
    } else if (try self.getIf(.kw_break)) |_| {
        try self.parseBreak();
    } else if (try self.getIf(.kw_continue)) |_| {
        try self.parseContinue();
    } else {
        try self.parseOp(0);
    }
}

fn parseIf(self: *Parser) E!void {
    _ = try self.expect(&.{.lparen});

    // Parse condition
    try self.parseExpr();

    // If true, jump ahead by 2 instruction slots to skip jump+offset
    try self.emit(.{ .insn = .jump_if });
    try self.emit(.{ .offset = 2 });

    // If false, jump to end of block
    // Push a temporary offset, save the index, and fill in this value later
    try self.emit(.{ .insn = .jump });
    const offset_a = self.mod.ir.items.len;
    try self.emit(.{ .offset = undefined });

    _ = try self.expect(&.{.rparen});

    // Parse body
    try self.parseExpr();

    if (try self.getIf(.kw_else)) |_| {

        // Push a temporary offset, save the index, and fill in this value later
        try self.emit(.{ .insn = .jump });
        const offset_b = self.mod.ir.items.len;
        try self.emit(.{ .offset = undefined });

        // Fill in temporary offset_a
        self.mod.ir.items[offset_a] = .{ .offset = @intCast(i32, self.mod.ir.items.len - offset_a - 1) };

        // Parse else body
        try self.parseExpr();

        // Fill in temporary offset_b
        self.mod.ir.items[offset_b] = .{ .offset = @intCast(i32, self.mod.ir.items.len - offset_b - 1) };
    } else {
        // Fill in temporary offset_a
        self.mod.ir.items[offset_a] = .{ .offset = @intCast(i32, self.mod.ir.items.len - offset_a - 1) };
    }
}

fn parseWhile(self: *Parser) E!void {
    _ = try self.expect(&.{.lparen});

    // Start of condition
    const cond = self.mod.ir.items.len;

    // Parse condition
    try self.parseExpr();

    _ = try self.expect(&.{.rparen});

    // If the condition is false, jump ahead
    try self.emit(.{ .insn = .jump_if_not });
    const offset_a = self.mod.ir.items.len;
    try self.emit(.{ .offset = undefined });

    const break_len_before = self.break_offsets.items.len;
    const continue_len_before = self.continue_offsets.items.len;

    // Parse body
    try self.parseExpr();

    const break_len_after = self.break_offsets.items.len;
    const continue_len_after = self.continue_offsets.items.len;

    // Jump back to condition
    try self.emit(.{ .insn = .jump });
    try self.emit(.{ .offset = @intCast(i32, cond) - @intCast(i32, self.mod.ir.items.len) - 1 });

    // Fill in temporary offset_a
    self.mod.ir.items[offset_a] = .{ .offset = @intCast(i32, self.mod.ir.items.len - offset_a - 1) };

    // Fill in any break statement offsets inside while body
    for (self.break_offsets.items[break_len_before..break_len_after]) |idx| {
        self.mod.ir.items[idx] = .{ .offset = @intCast(i32, self.mod.ir.items.len - idx - 1) };
    }
    try self.break_offsets.resize(self.allocator, break_len_before);

    // Fill in any continue statement offsets inside while body
    for (self.continue_offsets.items[continue_len_before..continue_len_after]) |idx| {
        self.mod.ir.items[idx] = .{ .offset = @intCast(i32, cond) - @intCast(i32, idx) - 1 };
    }
    try self.continue_offsets.resize(self.allocator, continue_len_before);
}

fn parseBreak(self: *Parser) E!void {
    // Jump to the end of the while block
    // The offset is unknown, so push a temp value
    try self.emit(.{ .insn = .jump });
    const offset = self.mod.ir.items.len;
    try self.emit(.{ .offset = undefined });

    try self.break_offsets.append(self.allocator, offset);
}

fn parseContinue(self: *Parser) E!void {
    // Jump to the start of the while block
    try self.emit(.{ .insn = .jump });
    const offset = self.mod.ir.items.len;
    try self.emit(.{ .offset = undefined });

    try self.continue_offsets.append(self.allocator, offset);
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

test "block" {
    try testParserSuccess(
        "{ 1; 2; 3; }",
        &.{
            .{ .insn = .push_int },
            .{ .int = 1 },
            .{ .insn = .pop },
            .{ .insn = .push_int },
            .{ .int = 2 },
            .{ .insn = .pop },
            .{ .insn = .push_int },
            .{ .int = 3 },
        },
    );
}

test "if" {
    try testParserSuccess(
        "if (3) 4",
        &.{
            .{ .insn = .push_int },
            .{ .int = 3 },

            .{ .insn = .jump_if },
            .{ .offset = 2 },
            .{ .insn = .jump },
            .{ .offset = 2 },

            .{ .insn = .push_int },
            .{ .int = 4 },
        },
    );
}

test "if else" {
    try testParserSuccess(
        "if (3) { 4; 5; } else 6",
        &.{
            .{ .insn = .push_int },
            .{ .int = 3 },

            .{ .insn = .jump_if },
            .{ .offset = 2 },
            .{ .insn = .jump },
            .{ .offset = 7 },

            .{ .insn = .push_int },
            .{ .int = 4 },
            .{ .insn = .pop },
            .{ .insn = .push_int },
            .{ .int = 5 },
            .{ .insn = .jump },
            .{ .offset = 2 },

            .{ .insn = .push_int },
            .{ .int = 6 },
        },
    );
}

test "while" {
    try testParserSuccess(
        "while (1) { 2; 3; }",
        &.{
            .{ .insn = .push_int },
            .{ .int = 1 },

            .{ .insn = .jump_if_not },
            .{ .offset = 7 },

            .{ .insn = .push_int },
            .{ .int = 2 },
            .{ .insn = .pop },
            .{ .insn = .push_int },
            .{ .int = 3 },

            .{ .insn = .jump },
            .{ .offset = -11 },
        },
    );
}

test "break" {
    try testParserSuccess(
        "while (1) { 2; break; 3; }",
        &.{
            .{ .insn = .push_int },
            .{ .int = 1 },

            .{ .insn = .jump_if_not },
            .{ .offset = 10 },

            .{ .insn = .push_int },
            .{ .int = 2 },
            .{ .insn = .pop },
            .{ .insn = .jump },
            .{ .offset = 5 },
            .{ .insn = .pop },
            .{ .insn = .push_int },
            .{ .int = 3 },

            .{ .insn = .jump },
            .{ .offset = -14 },
        },
    );
}

test "continue" {
    try testParserSuccess(
        "while (1) { 2; continue; 3; }",
        &.{
            .{ .insn = .push_int },
            .{ .int = 1 },

            .{ .insn = .jump_if_not },
            .{ .offset = 10 },

            .{ .insn = .push_int },
            .{ .int = 2 },
            .{ .insn = .pop },
            .{ .insn = .jump },
            .{ .offset = -9 },
            .{ .insn = .pop },
            .{ .insn = .push_int },
            .{ .int = 3 },

            .{ .insn = .jump },
            .{ .offset = -14 },
        },
    );
}
