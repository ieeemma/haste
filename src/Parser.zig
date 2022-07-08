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

stack_size: usize = 0,

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

// TODO: would a hash-assisted association list be faster than a hashmap here?
const Scope = std.StringHashMapUnmanaged(usize);

const Env = std.SinglyLinkedList(Scope);

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

fn parse(self: *Parser) E!ir.Module {
    try self.newScope(null, Parser.parseFile);

    return ir.Module{
        .imports = self.mod.imports.toOwnedSlice(self.allocator),
        .ir = self.mod.ir.toOwnedSlice(self.allocator),
        .locs = self.mod.locs.toOwnedSlice(self.allocator),
    };
}

fn parseFile(self: *Parser, scope: *Env.Node) E!void {

    // TODO: parse imports here

    while (!self.eof()) {
        try self.parseExpr(scope);
    }
}

fn parseExpr(self: *Parser, scope: *Env.Node) E!void {
    const size_before = self.stack_size;

    if (try self.getIf(.kw_if)) |_| {
        try self.parseIf(scope);
    } else if (try self.getIf(.kw_while)) |_| {
        try self.parseWhile(scope);
    } else if (try self.getIf(.kw_break)) |_| {
        try self.parseBreak();
    } else if (try self.getIf(.kw_continue)) |_| {
        try self.parseContinue();
    } else if (try self.getIf(.kw_let)) |_| {
        try self.parseLet(scope);
    } else {
        try self.parseOp(scope, 0);
    }

    if (self.stack_size != size_before + 1) {
        std.debug.panic(
            "Expression did not produce stack one value!\n{s}\n{any}",
            .{ self.src, self.mod.ir.items },
        );
    }
}

fn parseIf(self: *Parser, scope: *Env.Node) E!void {
    _ = try self.expect(&.{.lparen});

    // Parse condition
    try self.parseExpr(scope);

    // If true, jump ahead by 2 instruction slots to skip jump+offset
    try self.emitInsn(.jump_if, -1);
    try self.emitData(.{ .offset = 2 });

    // If false, jump to end of block
    // Push a temporary offset, save the index, and fill in this value later
    try self.emitInsn(.jump, 0);
    const offset_a = self.mod.ir.items.len;
    try self.emitData(.{ .offset = undefined });

    _ = try self.expect(&.{.rparen});

    // Parse body
    try self.newScope(scope, Parser.parseExpr);

    if (try self.getIf(.kw_else)) |_| {

        // Restore stack size
        self.stack_size -= 1;

        // Push a temporary offset, save the index, and fill in this value later
        try self.emitInsn(.jump, 0);
        const offset_b = self.mod.ir.items.len;
        try self.emitData(.{ .offset = undefined });

        // Fill in temporary offset_a
        self.mod.ir.items[offset_a] = .{ .offset = @intCast(i32, self.mod.ir.items.len - offset_a - 1) };

        // Parse else body
        try self.newScope(scope, Parser.parseExpr);

        // Fill in temporary offset_b
        self.mod.ir.items[offset_b] = .{ .offset = @intCast(i32, self.mod.ir.items.len - offset_b - 1) };
    } else {
        // Fill in temporary offset_a
        self.mod.ir.items[offset_a] = .{ .offset = @intCast(i32, self.mod.ir.items.len - offset_a - 1) };
    }
}

fn parseWhile(self: *Parser, scope: *Env.Node) E!void {
    _ = try self.expect(&.{.lparen});

    // Start of condition
    const cond = self.mod.ir.items.len;

    // Parse condition
    try self.parseExpr(scope);

    _ = try self.expect(&.{.rparen});

    // If the condition is false, jump ahead
    try self.emitInsn(.jump_if_not, -1);
    const offset_a = self.mod.ir.items.len;
    try self.emitData(.{ .offset = undefined });

    const break_len_before = self.break_offsets.items.len;
    const continue_len_before = self.continue_offsets.items.len;

    // Parse body
    try self.newScope(scope, Parser.parseExpr);

    // Restore stack size
    self.stack_size -= 1;

    const break_len_after = self.break_offsets.items.len;
    const continue_len_after = self.continue_offsets.items.len;

    // Jump back to condition
    try self.emitInsn(.jump, 0);
    try self.emitData(.{ .offset = @intCast(i32, cond) - @intCast(i32, self.mod.ir.items.len) - 1 });

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

    // Return `void`
    // TODO: `break <value>` for result of while expr
    try self.emitInsn(.push_void, 1);
}

fn parseBreak(self: *Parser) E!void {
    // Jump to the end of the while block
    // The offset is unknown, so push a temp value
    try self.emitInsn(.jump, 0);
    const offset = self.mod.ir.items.len;
    try self.emitData(.{ .offset = undefined });

    try self.break_offsets.append(self.allocator, offset);

    // For sake of stack size logic, push a dummy value
    // TODO: is there a smarter way to handle 'valueless values' like `break` and `continue`?
    try self.emitInsn(.push_void, 1);
}

fn parseContinue(self: *Parser) E!void {
    // Jump to the start of the while block
    try self.emitInsn(.jump, 0);
    const offset = self.mod.ir.items.len;
    try self.emitData(.{ .offset = undefined });

    try self.continue_offsets.append(self.allocator, offset);

    // For sake of stack size logic, push a dummy value
    try self.emitInsn(.push_void, 1);
}

fn parseLet(self: *Parser, scope: *Env.Node) E!void {
    const name_tok = try self.expect(&.{.symbol});
    const name = self.src[self.src_idx - name_tok.len .. self.src_idx];

    const offset = self.mod.ir.items.len;
    try self.emitInsn(.alloc, 1);
    try scope.data.put(self.allocator, name, offset);

    _ = try self.expect(&.{.equals});
    try self.parseExpr(scope);
    try self.emitInsn(.dup, 1);
    try self.emitInsn(.store, -1);
    // TODO: need to know stack size to know offset here
    try self.emitData(.{ .int = unreachable });
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

fn parseOp(self: *Parser, scope: *Env.Node, comptime prec_idx: usize) E!void {

    // Parse lhs
    if (prec_idx + 1 < precedence.len) {
        try self.parseOp(scope, prec_idx + 1);
    } else {
        try self.parseCall(scope);
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
                    try self.parseOp(scope, prec_idx);

                    // Emit op
                    try self.emitInsn(prec.insn, -1);
                }
            }
        }
    }
}

fn parseCall(self: *Parser, scope: *Env.Node) E!void {
    try self.parseAtom(scope);

    // If next is opening paren, continue parsing `<expression> COMMA`
    // until closing paren
    if (try self.getIf(.lparen)) |_| {
        var nargs: u32 = 0;

        while (true) {
            try self.parseExpr(scope);
            nargs += 1;
            switch ((try self.expect(&.{ .rparen, .comma })).tag) {
                .rparen => break,
                .comma => if (try self.getIf(.rparen)) |_| break,
                else => unreachable,
            }
        }

        try self.emitInsn(.call, -@intCast(i32, nargs));
        try self.emitData(.{ .uint = nargs });
    }
}

fn parseAtom(self: *Parser, scope: *Env.Node) E!void {
    const tok = try self.get();
    switch (tok.tag) {

        // Parenthesized expression
        .lparen => {
            try self.parseExpr(scope);
            _ = try self.expect(&.{.rparen});
        },

        .lbrace => try self.newScope(scope, Parser.parseBlock),

        .symbol => {
            _ = scope;

            // TODO: there needs to be actual logic here
            try self.emitInsn(.load, 1);
            try self.emitData(.{ .uint = 0 });
        },

        .int => {
            const int_str = self.src[self.src_idx - tok.len .. self.src_idx];
            const n = std.fmt.parseInt(i32, int_str, 0) catch {
                return self.parseError("Invalid integer literal", .{});
            };
            try self.emitInsn(.push_int, 1);
            try self.emitData(.{ .int = n });
        },

        .float => {
            const float_str = self.src[self.src_idx - tok.len .. self.src_idx];
            const n = std.fmt.parseFloat(f32, float_str) catch {
                return self.parseError("Invalid float literal", .{});
            };
            try self.emitInsn(.push_float, 1);
            try self.emitData(.{ .float = n });
        },

        // TODO: parse other literals here

        else => |tag| return self.parseError("Unexpected token {}", .{tag}),
    }
}

fn parseBlock(self: *Parser, scope: *Env.Node) E!void {
    // Continue parsing `<expression> SEMICOLON` until closing brace
    // Emit `pop` after all but last expression
    while (true) {
        try self.parseExpr(scope);
        _ = try self.expect(&.{.semi_colon});
        if (try self.getIf(.rbrace)) |_| {
            break;
        } else {
            try self.emitInsn(.pop, -1);
        }
    }
}

fn newScope(self: *Parser, scope: ?*Env.Node, comptime f: fn (*Parser, *Env.Node) E!void) E!void {
    var new = Env.Node{ .next = scope, .data = .{} };
    defer new.data.deinit(self.allocator);
    try f(self, &new);
}

inline fn emitData(self: *Parser, data: ir.IR) E!void {
    try self.mod.ir.append(self.allocator, data);
}

inline fn emitInsn(self: *Parser, insn: ir.Insn, diff: i32) E!void {
    try self.emitData(.{ .insn = insn });
    self.stack_size = @intCast(usize, @intCast(isize, self.stack_size) + diff);
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

    const mod = try parser.parse();
    defer allocator.free(mod.imports);
    defer allocator.free(mod.ir);
    defer allocator.free(mod.locs);

    // std.debug.print("\n", .{});
    // for (expected) |x, i| {
    //     std.debug.print("{} {}\n", .{ i, x });
    // }

    // std.debug.print("\n", .{});
    // for (mod.ir) |x, i| {
    //     std.debug.print("{} {}\n", .{ i, x });
    // }

    try std.testing.expectEqualSlices(ir.IR, expected, mod.ir);
}

test "int" {
    try testParserSuccess(
        "32",
        &.{ .{ .insn = .push_int }, .{ .int = 32 } },
    );
    try testParserSuccess(
        "0",
        &.{ .{ .insn = .push_int }, .{ .int = 0 } },
    );
}

test "float" {
    try testParserSuccess(
        "5.32",
        &.{ .{ .insn = .push_float }, .{ .float = 5.32 } },
    );
    try testParserSuccess(
        "0.01",
        &.{ .{ .insn = .push_float }, .{ .float = 0.01 } },
    );
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

            .{ .insn = .push_void },
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
            .{ .offset = 11 },

            .{ .insn = .push_int },
            .{ .int = 2 },
            .{ .insn = .pop },
            .{ .insn = .jump },
            .{ .offset = 6 },
            .{ .insn = .push_void },
            .{ .insn = .pop },
            .{ .insn = .push_int },
            .{ .int = 3 },

            .{ .insn = .jump },
            .{ .offset = -15 },

            .{ .insn = .push_void },
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
            .{ .offset = 11 },

            .{ .insn = .push_int },
            .{ .int = 2 },
            .{ .insn = .pop },
            .{ .insn = .jump },
            .{ .offset = -9 },
            .{ .insn = .push_void },
            .{ .insn = .pop },
            .{ .insn = .push_int },
            .{ .int = 3 },

            .{ .insn = .jump },
            .{ .offset = -15 },

            .{ .insn = .push_void },
        },
    );
}
