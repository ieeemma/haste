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
    ir: ir.IR,
};

const precedence = [_][]const Prec{
    &.{
        .{ .op = "==", .ir = .eq },
        .{ .op = "!=", .ir = .neq },
    },
    &.{
        .{ .op = "<", .ir = .lt },
        .{ .op = "<=", .ir = .le },
        .{ .op = ">", .ir = .gt },
        .{ .op = ">=", .ir = .ge },
    },
    &.{
        .{ .op = "+", .ir = .add },
        .{ .op = "-", .ir = .sub },
    },
    &.{ .{ .op = "*", .ir = .mul }, .{ .op = "/", .ir = .div } },
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
                    try self.mod.ir.append(self.allocator, prec.ir);
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

        try self.mod.ir.append(self.allocator, .call);
        try self.mod.ir.append(self.allocator, @intToEnum(ir.IR, nargs));
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
                    try self.mod.ir.append(self.allocator, .pop);
                }
            }
        },

        // TODO: parse symbols here

        // Integer literal
        .int => {
            // Parse integer, then emit `push_int` instruction
            var n: u32 = 0;
            var i: usize = self.src_idx - tok.len;
            while (i < self.src_idx) : (i += 1) {
                const ch = self.src[i];
                n = n * 10 + (ch - '0');
            }
            try self.mod.ir.append(self.allocator, .push_int);
            try self.mod.ir.append(self.allocator, @intToEnum(ir.IR, n));
        },

        // TODO: parse other literals here

        else => |tag| return self.parseError("Unexpected token {}", .{tag}),
    }
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
