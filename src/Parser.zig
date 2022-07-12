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

stack_size: u32 = 0,

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
const Scope = struct {
    base: u32,
    map: std.StringArrayHashMapUnmanaged(void),
    captures: ?std.StringArrayHashMapUnmanaged(void),
};

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

pub fn parse(self: *Parser) E!ir.Module {
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
    const tok = (try self.peek()).?;
    switch (tok.tag) {
        .kw_if => try self.parseIf(scope),
        .kw_while => try self.parseWhile(scope),
        .kw_break => try self.parseBreak(),
        .kw_continue => try self.parseContinue(),
        .kw_let => try self.parseLet(scope),
        .kw_fn => try self.parseFn(scope),
        else => try self.parseAssign(scope),
    }
}

fn parseIf(self: *Parser, scope: *Env.Node) E!void {
    _ = try self.get();

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
    _ = try self.get();

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
    _ = try self.get();

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
    _ = try self.get();

    // Jump to the start of the while block
    try self.emitInsn(.jump, 0);
    const offset = self.mod.ir.items.len;
    try self.emitData(.{ .offset = undefined });

    try self.continue_offsets.append(self.allocator, offset);

    // For sake of stack size logic, push a dummy value
    try self.emitInsn(.push_void, 1);
}

fn parseLet(self: *Parser, scope: *Env.Node) E!void {
    _ = try self.get();

    const name_tok = try self.expect(&.{.symbol});
    const name = self.tokenToStr(name_tok);

    const offset = try self.let_var(scope, name);

    _ = try self.expect(&.{.equals});
    try self.parseExpr(scope);

    try self.store(scope, offset);
}

fn parseFn(self: *Parser, scope: *Env.Node) E!void {
    _ = try self.get();

    const name = if (try self.getIf(.symbol)) |tok| self.tokenToStr(tok) else null;
    _ = name;

    _ = try self.expect(&.{.lparen});

    var new = Env.Node{
        .next = scope,
        .data = .{
            .base = 0,
            .map = .{},
            .captures = .{},
        },
    };
    defer new.data.map.deinit(self.allocator);
    defer new.data.captures.?.deinit(self.allocator);

    var nargs: u32 = 0;

    while (true) {
        const tok = try self.expect(&.{.symbol});
        const arg = self.tokenToStr(tok);
        nargs += 1;

        try new.data.map.put(self.allocator, arg, {});

        switch ((try self.expect(&.{ .rparen, .comma })).tag) {
            .rparen => break,
            .comma => if (try self.getIf(.rparen)) |_| break,
            else => unreachable,
        }
    }

    // Push a temporary function size
    try self.emitInsn(.func, 1);
    const offset = self.mod.ir.items.len;
    try self.emitData(.{ .uint = undefined });

    // Push number of arg allocations
    if (nargs > 0) {
        try self.emitInsn(.alloc, @intCast(i32, nargs));
        try self.emitData(.{ .uint = nargs });
    }

    const var_offset = if (name) |n| try self.let_var(scope, n) else null;

    try self.parseExpr(&new);

    // Deallocate args
    if (nargs > 0) {
        try self.emitInsn(.dealloc, -@intCast(i32, nargs));
        try self.emitData(.{ .uint = nargs });
    }

    // Fill in function IR size
    self.mod.ir.items[offset] = .{ .uint = @intCast(u32, self.mod.ir.items.len - offset - 1) };

    const cap = new.data.captures.?;
    const nclos = cap.count();
    if (nclos > 0) {
        var it = cap.iterator();
        while (it.next()) |entry| {
            try self.locateVariable(entry.key_ptr.*, scope);
        }
        try self.emitInsn(.closure, -@intCast(i32, nclos) - 1);
        try self.emitData(.{ .uint = @intCast(u32, nclos) });
    }

    if (var_offset) |off| {
        try self.store(scope, off);
    }
}

fn let_var(self: *Parser, scope: *Env.Node, name: []const u8) E!u32 {
    // Push a new variable declaration to the scope
    const offset = scope.data.map.count();
    try scope.data.map.put(self.allocator, name, {});
    self.stack_size += 1;

    return @intCast(u32, offset);
}

fn store(self: *Parser, scope: *Env.Node, offset: u32) E!void {
    // Duplicate the initial value, then store one copy in the variable slot
    try self.emitInsn(.dup, 1);
    try self.emitInsn(.store, -1);
    try self.emitData(.{ .uint = @intCast(u32, scope.data.base + offset) });
}

const assigns = std.ComptimeStringMap(ir.Insn, .{
    .{ "+=", .add },
    .{ "-=", .sub },
    .{ "*=", .mul },
    .{ "/=", .div },
});

fn parseAssign(self: *Parser, scope: *Env.Node) E!void {

    // IR size before parsing expr
    const before = self.mod.ir.items.len;

    try self.parseOp(scope, 0);

    // This is a hack!
    // To parse assignment, reinterpret the parsed IR as an lvaue

    // Get the assignment operator, if any
    const tok = (try self.peek()) orelse return;
    var op: []const u8 = undefined;
    switch (tok.tag) {
        .equals => op = "=",
        .op => op = self.src[self.src_idx .. self.src_idx + tok.len],
        else => return,
    }
    _ = try self.get();

    // Get a slice of the IR from the last expr
    const len = self.mod.ir.items.len - before;
    const rvalue = try self.allocator.alloc(ir.IR, len);
    defer self.allocator.free(rvalue);
    std.mem.copy(ir.IR, rvalue, self.mod.ir.items[before..self.mod.ir.items.len]);

    switch (op[0]) {
        '+', '-', '*', '/' => {},
        // Pop previous IR
        '=' => self.mod.ir.shrinkRetainingCapacity(before),
        else => unreachable,
    }

    // Parse value
    try self.parseExpr(scope);

    // Push arithmetic op, if any
    switch (op[0]) {
        '+', '-', '*', '/' => try self.emitInsn(assigns.get(op).?, 1),
        '=' => {},
        else => unreachable,
    }

    // Convert rvalue to lvalue
    try self.rvalueToLvalue(rvalue);
}

fn rvalueToLvalue(self: *Parser, rvalue: []const ir.IR) E!void {
    // TODO: add cases for attributes/list index etc
    if (rvalue.len == 2 and rvalue[0].insn == .load) {
        try self.emitInsn(.store, -1);
        try self.emitData(rvalue[1]);
    } else {
        return self.parseError("Invalid assignment target", .{});
    }
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

        .symbol => try self.locateVariable(self.tokenToStr(tok), scope),

        .int => {
            const int_str = self.tokenToStr(tok);
            const n = std.fmt.parseInt(i32, int_str, 0) catch {
                return self.parseError("Invalid integer literal", .{});
            };
            try self.emitInsn(.push_int, 1);
            try self.emitData(.{ .int = n });
        },

        .float => {
            const float_str = self.tokenToStr(tok);
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

fn locateVariable(self: *Parser, name: []const u8, s: *Env.Node) E!void {
    var scope: ?*Env.Node = s;

    while (scope) |current| : (scope = current.next) {
        if (current.data.map.getIndex(name)) |offset| {
            try self.emitInsn(.load, 1);
            try self.emitData(.{ .uint = @intCast(u32, current.data.base + offset) });
            return;
        }

        if (current.data.captures) |*cap| {
            try cap.put(self.allocator, name, {});
            try self.emitInsn(.cload, 1);
            try self.emitData(.{ .uint = @intCast(u32, cap.count() - 1) });

            while (scope) |current2| : (scope = current2.next) {
                if (current2.data.map.getIndex(name)) |_| return;
            }
        }
    }

    return self.parseError(
        "Undefined symbol '{s}'",
        .{name},
    );
}

fn newScope(self: *Parser, scope: ?*Env.Node, comptime f: fn (*Parser, *Env.Node) E!void) E!void {
    // TODO: add `expect` op for stack size and use `assumeCapacity` variants of stack ops in vm

    var new = Env.Node{
        .next = scope,
        .data = .{
            .base = self.stack_size,
            .map = .{},
            .captures = null,
        },
    };
    defer new.data.map.deinit(self.allocator);

    // Emit temporary offset
    try self.emitInsn(.alloc, 0);
    const offset = self.mod.ir.items.len;
    try self.emitData(.{ .uint = undefined });

    // Call sub parser with new scope node
    try f(self, &new);

    // If any vars were allocated in that expression, modify offset and emit `deinit`
    // Otherwise, replace `alloc` insn with `nop`s
    const n_vars = @intCast(u32, new.data.map.count());

    if (n_vars > 0) {
        self.mod.ir.items[offset] = .{ .uint = n_vars };
        try self.emitInsn(.dealloc, -@intCast(i32, n_vars));
        try self.emitData(.{ .uint = n_vars });
    } else {
        self.mod.ir.items[offset - 1] = .{ .insn = .nop };
        self.mod.ir.items[offset - 0] = .{ .insn = .nop };
    }
}

inline fn emitData(self: *Parser, data: ir.IR) E!void {
    try self.mod.ir.append(self.allocator, data);
}

inline fn emitInsn(self: *Parser, insn: ir.Insn, diff: i32) E!void {
    try self.emitData(.{ .insn = insn });
    self.stack_size = @intCast(u32, @intCast(i32, self.stack_size) + diff);
}

fn eof(self: Parser) bool {
    return self.idx >= self.tokens.len;
}

fn tokenToStr(self: Parser, tok: Token) []const u8 {
    return self.src[self.src_idx - tok.len .. self.src_idx];
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

    return self.parseError("Invalid token: expected one of {any}, but got {}", .{ expected, tok.tag });
}

fn parseError(self: *Parser, comptime fmt: []const u8, args: anytype) E {
    // TODO: add source line
    self.err = .{
        .msg = try std.fmt.allocPrint(self.allocator, fmt, args),
        .idx = self.src_idx,
    };
    return error.ParseError;
}

fn testParserSuccess(comptime src: []const u8, comptime exp: []const ir.IR) !void {
    const allocator = std.testing.allocator;

    const expected = [_]ir.IR{.{ .insn = .nop }} ** 2 ++ exp;

    var parser = try Parser.init(allocator, src);
    defer parser.deinit();
    errdefer {
        parser.mod.imports.deinit(allocator);
        parser.mod.ir.deinit(allocator);
        parser.mod.locs.deinit(allocator);
    }

    const mod = parser.parse() catch |err| switch (err) {
        error.ParseError => {
            std.debug.panic("{s}", .{parser.err.msg});
        },
        else => return err,
    };
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
            .{ .insn = .nop },
            .{ .insn = .nop },

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
            .{ .offset = 4 },

            .{ .insn = .nop },
            .{ .insn = .nop },
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
            .{ .offset = 11 },

            .{ .insn = .nop },
            .{ .insn = .nop },
            .{ .insn = .nop },
            .{ .insn = .nop },
            .{ .insn = .push_int },
            .{ .int = 4 },
            .{ .insn = .pop },
            .{ .insn = .push_int },
            .{ .int = 5 },
            .{ .insn = .jump },
            .{ .offset = 4 },

            .{ .insn = .nop },
            .{ .insn = .nop },
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
            .{ .offset = 11 },

            .{ .insn = .nop },
            .{ .insn = .nop },
            .{ .insn = .nop },
            .{ .insn = .nop },
            .{ .insn = .push_int },
            .{ .int = 2 },
            .{ .insn = .pop },
            .{ .insn = .push_int },
            .{ .int = 3 },

            .{ .insn = .jump },
            .{ .offset = -15 },

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
            .{ .offset = 15 },

            .{ .insn = .nop },
            .{ .insn = .nop },
            .{ .insn = .nop },
            .{ .insn = .nop },
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
            .{ .offset = -19 },

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
            .{ .offset = 15 },

            .{ .insn = .nop },
            .{ .insn = .nop },
            .{ .insn = .nop },
            .{ .insn = .nop },
            .{ .insn = .push_int },
            .{ .int = 2 },
            .{ .insn = .pop },
            .{ .insn = .jump },
            .{ .offset = -13 },
            .{ .insn = .push_void },
            .{ .insn = .pop },
            .{ .insn = .push_int },
            .{ .int = 3 },

            .{ .insn = .jump },
            .{ .offset = -19 },

            .{ .insn = .push_void },
        },
    );
}

test "let" {
    try testParserSuccess(
        "{ let x = 5; x + 5; }",
        &.{
            .{ .insn = .alloc },
            .{ .uint = 1 },

            .{ .insn = .push_int },
            .{ .int = 5 },
            .{ .insn = .dup },
            .{ .insn = .store },
            .{ .uint = 0 },
            .{ .insn = .pop },

            .{ .insn = .load },
            .{ .uint = 0 },
            .{ .insn = .push_int },
            .{ .int = 5 },
            .{ .insn = .add },

            .{ .insn = .dealloc },
            .{ .uint = 1 },
        },
    );
}

test "nested scopes" {
    try testParserSuccess(
        "{ let x = { let y = 5; let z = 10; y + z; }; x / 2; }",
        &.{
            .{ .insn = .alloc },
            .{ .uint = 1 },

            .{ .insn = .alloc },
            .{ .uint = 2 },

            .{ .insn = .push_int },
            .{ .int = 5 },
            .{ .insn = .dup },
            .{ .insn = .store },
            .{ .uint = 1 },
            .{ .insn = .pop },

            .{ .insn = .push_int },
            .{ .int = 10 },
            .{ .insn = .dup },
            .{ .insn = .store },
            .{ .uint = 2 },
            .{ .insn = .pop },

            .{ .insn = .load },
            .{ .uint = 1 },
            .{ .insn = .load },
            .{ .uint = 2 },
            .{ .insn = .add },

            .{ .insn = .dealloc },
            .{ .uint = 2 },

            .{ .insn = .dup },
            .{ .insn = .store },
            .{ .uint = 0 },
            .{ .insn = .pop },

            .{ .insn = .load },
            .{ .uint = 0 },
            .{ .insn = .push_int },
            .{ .int = 2 },
            .{ .insn = .div },

            .{ .insn = .dealloc },
            .{ .uint = 1 },
        },
    );
}

test "assign" {
    try testParserSuccess(
        "{ let x = 5; x = 10; }",
        &.{
            .{ .insn = .alloc },
            .{ .uint = 1 },

            .{ .insn = .push_int },
            .{ .int = 5 },
            .{ .insn = .dup },
            .{ .insn = .store },
            .{ .uint = 0 },
            .{ .insn = .pop },

            .{ .insn = .push_int },
            .{ .int = 10 },
            .{ .insn = .store },
            .{ .uint = 0 },

            .{ .insn = .dealloc },
            .{ .uint = 1 },
        },
    );
}

test "arithmetic assign" {
    try testParserSuccess(
        "{ let x = 5; x += 3; }",
        &.{
            .{ .insn = .alloc },
            .{ .uint = 1 },

            .{ .insn = .push_int },
            .{ .int = 5 },
            .{ .insn = .dup },
            .{ .insn = .store },
            .{ .uint = 0 },
            .{ .insn = .pop },

            .{ .insn = .load },
            .{ .uint = 0 },
            .{ .insn = .push_int },
            .{ .int = 3 },
            .{ .insn = .add },
            .{ .insn = .store },
            .{ .uint = 0 },

            .{ .insn = .dealloc },
            .{ .uint = 1 },
        },
    );
}

test "fn" {
    try testParserSuccess(
        \\ {
        \\     fn inc(x) {
        \\         x + 1;
        \\     };
        \\     inc(5);
        \\ }
    ,
        &.{
            .{ .insn = .alloc },
            .{ .uint = 1 },

            .{ .insn = .func },
            .{ .uint = 11 },

            .{ .insn = .alloc },
            .{ .uint = 1 },

            .{ .insn = .nop },
            .{ .insn = .nop },
            .{ .insn = .load },
            .{ .uint = 0 },
            .{ .insn = .push_int },
            .{ .int = 1 },
            .{ .insn = .add },

            .{ .insn = .dealloc },
            .{ .uint = 1 },

            .{ .insn = .dup },
            .{ .insn = .store },
            .{ .uint = 0 },
            .{ .insn = .pop },

            .{ .insn = .load },
            .{ .uint = 0 },
            .{ .insn = .push_int },
            .{ .int = 5 },
            .{ .insn = .call },
            .{ .uint = 1 },

            .{ .insn = .dealloc },
            .{ .uint = 1 },
        },
    );
}

test "capturing fn" {
    try testParserSuccess(
        \\ {
        \\     let x = 5;
        \\     fn addx(y) {
        \\         x + y;
        \\     };
        \\ }
    ,
        &.{
            .{ .insn = .alloc },
            .{ .uint = 2 },

            .{ .insn = .push_int },
            .{ .int = 5 },
            .{ .insn = .dup },
            .{ .insn = .store },
            .{ .uint = 0 },
            .{ .insn = .pop },

            .{ .insn = .func },
            .{ .uint = 11 },

            .{ .insn = .alloc },
            .{ .uint = 1 },

            .{ .insn = .nop },
            .{ .insn = .nop },

            .{ .insn = .cload },
            .{ .uint = 0 },
            .{ .insn = .load },
            .{ .uint = 0 },
            .{ .insn = .add },

            .{ .insn = .dealloc },
            .{ .uint = 1 },

            .{ .insn = .load },
            .{ .uint = 0 },
            .{ .insn = .closure },
            .{ .uint = 1 },

            .{ .insn = .dup },
            .{ .insn = .store },
            .{ .uint = 1 },

            .{ .insn = .dealloc },
            .{ .uint = 2 },
        },
    );
}
