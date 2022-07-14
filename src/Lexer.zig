const std = @import("std");

const Lexer = @This();

src: []const u8,
pos: usize = 0,

pub fn tokenize(allocator: std.mem.Allocator, src: []const u8) !std.MultiArrayList(Token) {
    var toks: std.MultiArrayList(Token) = .{};
    var l = Lexer{ .src = src };
    while (l.next()) |tok| {
        try toks.append(allocator, tok);
    }
    return toks;
}

fn next(self: *Lexer) ?Token {
    if (self.eof()) return null;

    const start = self.pos;
    const t = switch (self.src[self.pos]) {
        '(' => self.single(.lparen),
        ')' => self.single(.rparen),
        '{' => self.single(.lbrace),
        '}' => self.single(.rbrace),
        ';' => self.single(.semi_colon),
        ',' => self.single(.comma),

        'a'...'z', 'A'...'Z', '_' => self.symbol(),
        '+', '-', '*', '/', '%', '<', '>', '=', '!' => self.operator(),
        '0'...'9' => self.num(),
        '"' => self.string(),
        '\'' => self.string(),

        ' ', '\n', '\t' => self.ws(),
        '#' => self.comment(),

        else => self.single(.invalid),
    };
    return Token{ .len = @intCast(u16, self.pos - start), .tag = t };
}

const keywords = std.ComptimeStringMap(Token.Tag, .{
    .{ "fn", .kw_fn },
    .{ "if", .kw_if },
    .{ "else", .kw_else },
    .{ "while", .kw_while },
    .{ "break", .kw_break },
    .{ "continue", .kw_continue },
    .{ "return", .kw_return },
    .{ "let", .kw_let },
    .{ "end", .kw_end },
});

pub const ops = std.ComptimeStringMap(void, .{
    .{ "+", {} },
    .{ "-", {} },
    .{ "*", {} },
    .{ "/", {} },

    .{ "==", {} },
    .{ "!=", {} },
    .{ "<", {} },
    .{ "<=", {} },
    .{ ">", {} },
    .{ ">=", {} },

    .{ "+=", {} },
    .{ "-=", {} },
    .{ "*=", {} },
    .{ "/=", {} },
});

inline fn single(self: *Lexer, comptime t: Token.Tag) Token.Tag {
    self.pos += 1;
    return t;
}

fn symbol(self: *Lexer) Token.Tag {
    const start = self.pos;
    self.pos += 1;

    while (!self.eof()) : (self.pos += 1) {
        switch (self.src[self.pos]) {
            'a'...'z', 'A'...'Z', '_' => {},
            else => break,
        }
    }

    return keywords.get(self.src[start..self.pos]) orelse .symbol;
}

fn operator(self: *Lexer) Token.Tag {
    const start = self.pos;
    self.pos += 1;

    while (!self.eof()) : (self.pos += 1) {
        switch (self.src[self.pos]) {
            '+', '-', '*', '/', '%', '<', '>', '=' => {},
            else => break,
        }
    }

    const op = self.src[start..self.pos];

    // Special case for equality
    if (op.len == 1 and op[0] == '=') return .equals;

    // TODO: zig pr, make `has` into `contains`
    return if (ops.has(op)) .op else .invalid;
}

fn num(self: *Lexer) Token.Tag {
    self.pos += 1;
    var dot = false;
    while (!self.eof()) : (self.pos += 1) {
        switch (self.src[self.pos]) {
            '0'...'9' => {},
            '.' => if (dot) {
                return .invalid;
            } else {
                dot = true;
            },
            else => break,
        }
    }
    return if (dot) .float else .int;
}

fn string(self: *Lexer) Token.Tag {
    const delim = self.src[self.pos];
    self.pos += 1;
    var esc = false;
    while (!self.eof()) : (self.pos += 1) {
        const c = self.src[self.pos];
        if (esc) {
            esc = false;
        } else if (c == '\\') {
            esc = true;
        } else if (c == delim) {
            self.pos += 1;
            break;
        }
    } else {
        return .invalid;
    }

    return switch (delim) {
        '"' => .string,
        '\'' => .rune,
        else => unreachable,
    };
}

fn ws(self: *Lexer) Token.Tag {
    self.pos += 1;
    while (!self.eof()) : (self.pos += 1) {
        switch (self.src[self.pos]) {
            ' ', '\n', '\t' => {},
            else => break,
        }
    }
    return .ws;
}

fn comment(self: *Lexer) Token.Tag {
    self.pos += 1;
    while (!self.eof() and self.src[self.pos] != '\n') : (self.pos += 1) {}
    return .comment;
}

inline fn eof(self: Lexer) bool {
    return self.pos >= self.src.len;
}

pub const Token = struct {
    len: u16,
    tag: Tag,

    pub const Tag = enum(u8) {
        // Punctuation
        lparen,
        rparen,
        lbrace,
        rbrace,
        equals,
        semi_colon,
        comma,
        // Keywords
        kw_fn,
        kw_if,
        kw_else,
        kw_while,
        kw_break,
        kw_continue,
        kw_return,
        kw_let,
        kw_end,
        // Complex
        symbol,
        op,
        int,
        float,
        string,
        rune,
        // Misc
        ws,
        comment,
        invalid,
    };
};

test "tokenize numbers" {
    try testTok(
        "31.2 14 8.",
        &.{ .float, .ws, .int, .ws, .float },
    );
}

test "tokenize strings" {
    try testTok(
        \\"Hello, world" "" "escaped\"string"
    ,
        &.{ .string, .ws, .string, .ws, .string },
    );
}

test "tokenize example" {
    try testTok("fn foo(x) if x 5\n# This is a comment!", &.{ .kw_fn, .ws, .symbol, .lparen, .symbol, .rparen, .ws, .kw_if, .ws, .symbol, .ws, .int, .ws, .comment });
}

fn testTok(
    src: []const u8,
    expected: []const Token.Tag,
) !void {
    var toks = try tokenize(std.testing.allocator, src);
    defer toks.deinit(std.testing.allocator);

    var pos: u32 = 0;
    const s = toks.slice();
    for (s.items(.tag)) |tok, i| {
        std.testing.expectEqual(expected[i], tok) catch |err| {
            std.debug.print("{s}\n", .{src[pos .. pos + s.items(.len)[i]]});
            return err;
        };
        pos += s.items(.len)[i];
    }

    try std.testing.expectEqual(expected.len, toks.len);
    try std.testing.expectEqual(src.len, pos);
}
