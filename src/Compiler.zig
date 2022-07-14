const std = @import("std");

const Lexer = @import("Lexer.zig");
const Token = Lexer.Token;
const ir = @import("ir.zig");

const Compiler = @This();

allocator: std.mem.Allocator,

tokens: std.MultiArrayList(Token),
idx: u32 = 0,

src: []const u8,
src_idx: u32 = 0,

stack_size: u32 = 0,

break_offsets: std.ArrayListUnmanaged(usize) = .{},
continue_offsets: std.ArrayListUnmanaged(usize) = .{},

ir: std.ArrayListUnmanaged(ir.IR) = .{},
locs: std.ArrayListUnmanaged(u32) = .{},

err_msg: ?[]const u8 = undefined,

const Scope = struct {
    next: ?*Scope,
    // Base offset (0 for func scope)
    base: u32,
    // Local `let` vars
    locals: std.StringArrayHashMapUnmanaged(void) = .{},
    // Function captures (func scope if set)
    captures: ?std.StringHashMapUnmanaged(void) = null,
    // IR offset to write alloc amount to
    offset: u32,

    fn deinit(self: *Scope, allocator: std.mem.Allocator) void {
        self.locals.deinit(allocator);
    }

    fn begin(comp: *Compiler, next: ?*Scope, base: u32) Error!Scope {
        const self = Scope{
            .next = next,
            .base = base,
            .offset = @intCast(u32, comp.ir.items.len + 1),
        };

        // Emit temporary offset
        try comp.emitInsn(.alloc, 0);
        try comp.emitData(.{ .uint = undefined });

        return self;
    }

    fn end(self: *Scope, comp: *Compiler) Error!void {
        // If any vars were allocated in the scope, modify offset and emit `deinit`
        // Otherwise, replace `alloc` insn with `nop`s
        const nvars = @intCast(u32, self.locals.count());

        if (nvars > 0) {
            comp.ir.items[self.offset].uint = nvars;
            try comp.emitInsn(.dealloc, -@intCast(i32, nvars));
            try comp.emitData(.{ .uint = nvars });
        } else {
            comp.ir.items[self.offset - 1] = .{ .insn = .nop };
            comp.ir.items[self.offset - 0] = .{ .insn = .nop };
        }

        self.deinit(comp.allocator);
    }

    fn load(self: *Scope, comp: *Compiler, name: []const u8) Error!void {
        var scope: ?*Scope = self;

        while (scope) |current| : (scope = current.next) {
            if (current.locals.getIndex(name)) |offset| {
                try comp.emitInsn(.load, 1);
                try comp.emitData(.{ .uint = @intCast(u32, current.base + offset) });
                return;
            }

            if (current.captures) |*caps| {
                try caps.put(comp.allocator, name, {});
                try comp.emitInsn(.cload, 1);
                try comp.emitData(.{ .uint = @intCast(u32, caps.count() - 1) });

                while (scope) |current2| : (scope = current2.next) {
                    if (current2.locals.getIndex(name)) |_| return;
                }
            }
        }

        return comp.parseError("Undefined symbol '{s}'", .{name});
    }

    fn new(self: *Scope, comp: *Compiler, name: []const u8) Error!u32 {
        // Push a new variable declaration to the scope
        const offset = self.locals.count();
        try self.locals.put(comp.allocator, name, {});
        comp.stack_size += 1;

        return @intCast(u32, offset);
    }

    fn bind(self: *Scope, comp: *Compiler, offset: u32) Error!void {
        // Duplicate the initial value, then store one copy in the variable slot
        //try comp.emitInsn(.dup, 1);
        try comp.emitInsn(.store, -1);
        try comp.emitData(.{ .uint = @intCast(u32, self.base + offset) });
    }
};

const Error = error{ OutOfMemory, ParseError };

pub fn init(allocator: std.mem.Allocator, src: []const u8) !Compiler {
    return Compiler{
        .allocator = allocator,
        .tokens = try Lexer.tokenize(allocator, src),
        .src = src,
    };
}

pub fn deinit(self: *Compiler) void {
    self.tokens.deinit(self.allocator);
    self.break_offsets.deinit(self.allocator);
    self.continue_offsets.deinit(self.allocator);
}

pub fn compile(self: *Compiler) Error!ir.Module {
    errdefer self.deinit();

    var scope = try Scope.begin(self, null, 0);
    errdefer scope.deinit(self.allocator);

    try self.compileFile(&scope);

    try scope.end(self);

    return ir.Module{
        .imports = undefined,
        .ir = self.ir.toOwnedSlice(self.allocator),
        .locs = self.locs.toOwnedSlice(self.allocator),
    };
}

fn compileFile(self: *Compiler, scope: *Scope) Error!void {

    // TODO: parse imports here

    while (!self.eof()) {
        try self.compileStmtOpt(scope, true);
    }
}

fn compileStmtOpt(self: *Compiler, scope: *Scope, comptime in_block: bool) Error!void {
    const tok = try self.peekNoEof();
    switch (tok.tag) {
        // Compound statements (no semicolon)
        .kw_fn => return self.compileFnStmt(scope),
        .kw_if => return self.compileIfStmt(scope),
        .kw_while => return self.compileWhile(scope),
        .lbrace => return self.compileBlockStmt(scope),

        // Simple statements
        .kw_let => try self.compileLet(scope),
        .kw_return => try self.compileReturn(scope),
        .kw_break => try self.compileBreak(),
        .kw_continue => try self.compileContinue(),
        else => {
            const start_idx = self.ir.items.len;
            try self.compileExpr(scope);
            if (!try self.compileAssign(scope, start_idx)) {
                try self.emitInsn(.pop, -1);
            }
        },
    }
    if (in_block) _ = try self.expect(&.{.semi_colon});
}

fn compileStmt(self: *Compiler, scope: *Scope) Error!void {
    return self.compileStmtOpt(scope, false);
}

inline fn compileFnStmt(self: *Compiler, scope: *Scope) Error!void {
    return self.compileFn(scope, false);
}

inline fn compileIfStmt(self: *Compiler, scope: *Scope) Error!void {
    return self.compileIf(scope, false);
}

fn compileWhile(self: *Compiler, scope: *Scope) Error!void {

    // Skip `while` keyword
    _ = try self.get();

    _ = try self.expect(&.{.lparen});

    // Start of cond index
    const cond_idx = self.ir.items.len;

    try self.compileExpr(scope);

    _ = try self.expect(&.{.rparen});

    // If condition is false, jump past body
    // Fill in offset later
    try self.emitInsn(.jump_if_not, -1);
    const end_idx = self.ir.items.len;
    try self.emitData(.{ .offset = undefined });

    const break_len_before = self.break_offsets.items.len;
    const continue_len_before = self.continue_offsets.items.len;

    // Parse body
    try self.compileStmt(scope);

    const break_len_after = self.break_offsets.items.len;
    const continue_len_after = self.continue_offsets.items.len;

    // Jump back to condition
    try self.emitInsn(.jump, 0);
    try self.emitData(.{ .offset = @intCast(i32, cond_idx) - @intCast(i32, self.ir.items.len) - 1 });

    // Fill in end idx
    self.ir.items[end_idx].offset = @intCast(i32, self.ir.items.len - end_idx - 1);

    // Fill in break statement offsets
    for (self.break_offsets.items[break_len_before..break_len_after]) |idx| {
        self.ir.items[idx].offset = @intCast(i32, self.ir.items.len - idx - 1);
    }
    try self.break_offsets.resize(self.allocator, break_len_before);

    // Fill in continue statement offsets
    for (self.continue_offsets.items[continue_len_before..continue_len_after]) |idx| {
        self.ir.items[idx].offset = @intCast(i32, cond_idx) - @intCast(i32, idx) - 1;
    }
    try self.continue_offsets.resize(self.allocator, continue_len_before);
}

fn compileBlockStmt(self: *Compiler, scope: *Scope) Error!void {
    var new = try Scope.begin(self, scope, self.stack_size);
    errdefer new.deinit(self.allocator);

    _ = try self.get();
    while (true) {
        if (try self.getIf(.rbrace)) |_| break;
        try self.compileStmtOpt(&new, true);
    }

    try new.end(self);
}

fn compileLet(self: *Compiler, scope: *Scope) Error!void {

    // Skip `let` keyword
    _ = try self.get();

    const name = self.tokenString(try self.expect(&.{.symbol}));

    // Define a new variable slot
    const offset = try scope.new(self, name);

    _ = try self.expect(&.{.equals});

    try self.compileExpr(scope);

    // Bind to previous slot
    try scope.bind(self, offset);
}

fn compileReturn(self: *Compiler, scope: *Scope) Error!void {
    _ = try self.get();
    _ = self;
    _ = scope;
    unreachable;
}

fn compileBreak(self: *Compiler) Error!void {

    // Skip `break` keyword
    _ = try self.get();

    // Jump to the end of the while block
    // The offset is unknown, so push a temp value
    try self.emitInsn(.jump, 0);
    const idx = self.ir.items.len;
    try self.emitData(.{ .offset = undefined });

    try self.break_offsets.append(self.allocator, idx);
}

fn compileContinue(self: *Compiler) Error!void {

    // Skip `continue` keyword
    _ = try self.get();

    // Jump to the start of the while block
    // The offset is unknown, so push a temp value
    try self.emitInsn(.jump, 0);
    const offset = self.ir.items.len;
    try self.emitData(.{ .offset = undefined });

    try self.continue_offsets.append(self.allocator, offset);
}

const assigns = std.ComptimeStringMap(ir.Insn, .{
    .{ "+=", .add },
    .{ "-=", .sub },
    .{ "*=", .mul },
    .{ "/=", .div },
});

fn compileAssign(self: *Compiler, scope: *Scope, start_idx: usize) Error!bool {
    // This is a big hack!
    // To compile an assignment, reinterpret the compiled IR for the target as an lvalue

    // Get assignment op, or null for plain equals
    const next = (try self.peek()) orelse return false;

    if (next.tag == .equals or next.tag == .op) {
        self.advance(next);
    }
    const insn = switch (next.tag) {
        .equals => null,
        .op => assigns.get(self.tokenString(next)) orelse return false,
        else => return false,
    };

    // Get slice of rvalue ir
    const current = self.ir.items.len;
    const len = current - start_idx;
    const rvalue = try self.allocator.alloc(ir.IR, len);
    defer self.allocator.free(rvalue);
    std.mem.copy(ir.IR, rvalue, self.ir.items[start_idx..current]);

    // If the assignemnt is an arithmetic op, keep the rvalue on the stack
    // Else, remove that ir
    if (insn == null) self.ir.shrinkRetainingCapacity(start_idx);

    // Compile assignment value
    try self.compileExpr(scope);

    // Push the arithmetic instruction, if applicable
    if (insn) |i| try self.emitInsn(i, 1);

    // Convert rvalue to lvalue
    try self.rvalueToLvalue(rvalue);

    return true;
}

fn rvalueToLvalue(self: *Compiler, rvalue: []const ir.IR) Error!void {
    // TODO: add cases for attributes/list index etc
    if (rvalue.len == 2 and rvalue[0].insn == .load) {
        try self.emitInsn(.store, -1);
        try self.emitData(rvalue[1]);
    } else {
        return self.parseError("Invalid assignment target", .{});
    }
}

fn compileExpr(self: *Compiler, scope: *Scope) Error!void {
    const tok = try self.peekNoEof();
    switch (tok.tag) {
        .kw_fn => try self.compileFnExpr(scope),
        .kw_if => try self.compileIfExpr(scope),
        .lbrace => try self.compileBlockExpr(scope, false),
        else => try self.compileOp(scope, 0),
    }
}

inline fn compileIfExpr(self: *Compiler, scope: *Scope) Error!void {
    return self.compileIf(scope, true);
}

inline fn compileFnExpr(self: *Compiler, scope: *Scope) Error!void {
    return self.compileFn(scope, true);
}

fn compileBlockExpr(self: *Compiler, scope: *Scope, comptime keep_scope: bool) Error!void {

    // Skip lbrace token
    _ = try self.get();

    // In case block is function body, dont make a new scope
    var new: *Scope = scope;
    if (!keep_scope) {
        new = &try Scope.begin(self, scope, self.stack_size);
    }

    // Continue parsing statements until rbrace or end keyword
    while (true) {
        const tok = try self.peekNoEof();
        switch (tok.tag) {
            .rbrace => {
                _ = try self.get();
                try self.emitInsn(.push_void, 1);
                break;
            },
            .kw_end => {
                _ = try self.get();
                try self.compileExpr(new);
                _ = try self.expect(&.{.semi_colon});
                _ = try self.expect(&.{.rbrace});
                break;
            },
            else => try self.compileStmtOpt(new, true),
        }
    }

    if (!keep_scope) try new.end(self);
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

fn compileOp(self: *Compiler, scope: *Scope, comptime prec_idx: usize) Error!void {

    // Parse lhs
    if (prec_idx + 1 < precedence.len) {
        try self.compileOp(scope, prec_idx + 1);
    } else {
        try self.compileCall(scope);
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
                    try self.compileOp(scope, prec_idx);

                    // Emit op
                    try self.emitInsn(prec.insn, -1);
                }
            }
        }
    }
}

fn compileCall(self: *Compiler, scope: *Scope) Error!void {

    // Compile target
    try self.compileAtom(scope);

    // If next token is opening paren...
    if (try self.getIf(.lparen)) |_| {

        // Parse args until closing paren
        var nargs: u32 = 0;
        while (true) {
            try self.compileExpr(scope);
            nargs += 1;
            const after = try self.expect(&.{ .rparen, .comma });
            switch (after.tag) {
                .rparen => break,
                .comma => if (try self.getIf(.rparen)) |_| break,
                else => unreachable,
            }
        }

        try self.emitInsn(.call, -@intCast(i32, nargs));
        try self.emitData(.{ .uint = nargs });
    }
}

fn compileAtom(self: *Compiler, scope: *Scope) Error!void {
    const tok = try self.get();
    switch (tok.tag) {
        .lparen => {
            try self.compileExpr(scope);
            _ = try self.expect(&.{.rparen});
        },

        .symbol => try scope.load(self, self.tokenString(tok)),

        .int => {
            const int_str = self.tokenString(tok);
            const n = std.fmt.parseInt(i32, int_str, 0) catch {
                return self.parseError("Invalid integer literal", .{});
            };
            try self.emitInsn(.push_int, 1);
            try self.emitData(.{ .int = n });
        },

        .float => {
            const float_str = self.tokenString(tok);
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

fn compileFn(self: *Compiler, scope: *Scope, comptime is_expr: bool) Error!void {

    // Skip `fn` keyword
    _ = try self.get();

    // Parse fn name
    var name: []const u8 = undefined;
    if (!is_expr) {
        const tok = try self.expect(&.{.symbol});
        name = self.tokenString(tok);
    }

    // Emit size of function as undefined
    try self.emitInsn(.func, 1);
    const size_idx = self.ir.items.len;
    try self.emitData(.{ .uint = undefined });

    var new = try Scope.begin(self, scope, 0);
    new.captures = .{};
    errdefer new.deinit(self.allocator);
    defer new.captures.?.deinit(self.allocator);

    _ = try self.expect(&.{.lparen});

    // Parse arguments
    var nargs: u32 = 0;
    while (true) {
        if (try self.getIf(.symbol)) |arg| {
            try new.locals.put(self.allocator, self.tokenString(arg), {});
            nargs += 1;
        }
        const after = try self.expect(&.{ .comma, .rparen });
        if (after.tag == .rparen) break;
    }

    // Compile body, keeping `new` as current scope in case of block
    // Special case for blocks
    const tok = try self.peekNoEof();
    if (tok.tag == .lbrace) {
        try self.compileBlockExpr(&new, true);
    } else {
        try self.compileExpr(&new);
    }

    try new.end(self);

    // Fill in size of function
    self.ir.items[size_idx].uint = @intCast(u32, self.ir.items.len - size_idx + 1);

    // Load any closure variables and create closure object
    const caps = new.captures.?;
    const ncaps = caps.count();
    if (ncaps > 0) {
        var it = caps.keyIterator();
        while (it.next()) |cap| {
            try scope.load(self, cap.*);
        }
        try self.emitInsn(.closure, -@intCast(i32, ncaps));
        try self.emitData(.{ .uint = ncaps });
    }

    // Bind to name if not expr
    if (!is_expr) {
        const offset = scope.locals.getIndex(name) orelse try scope.new(self, name);
        try scope.bind(self, @intCast(u32, offset));
    }
}

fn compileIf(self: *Compiler, scope: *Scope, comptime is_expr: bool) Error!void {

    // Skip `if` keyword
    _ = try self.get();

    _ = try self.expect(&.{.lparen});

    // Compile condition
    try self.compileExpr(scope);

    _ = try self.expect(&.{.rparen});

    // If condition is truthy, jump to body
    try self.emitInsn(.jump_if, -1);
    try self.emitData(.{ .offset = 2 });

    // Otherwise, jump to `else` (offset undefined, fill in later)
    try self.emitInsn(.jump, 0);
    const else_idx = self.ir.items.len;
    try self.emitData(.{ .offset = undefined });

    const f = comptime if (is_expr) Compiler.compileExpr else Compiler.compileStmt;

    // Compile body
    try f(self, scope);

    // Jump to skip `else` block
    try self.emitInsn(.jump, 0);
    const end_idx = self.ir.items.len;
    try self.emitData(.{ .offset = undefined });

    // Fill in `else` jump
    self.ir.items[else_idx].offset = @intCast(i32, self.ir.items.len - else_idx - 1);

    if (try self.getIf(.kw_else)) |_| {

        // Restore stack size
        if (is_expr) self.stack_size -= 1;

        // Compile else body
        try f(self, scope);
    } else if (is_expr) {
        try self.emitInsn(.push_void, 1);
    } else {
        // If statement and no `else` block, pop last jump insn, correct
        // else offset, and skip filling in end offset
        _ = self.ir.pop();
        _ = self.ir.pop();
        self.ir.items[else_idx].offset -= 2;
        return;
    }

    // Fill in `end` jump
    self.ir.items[end_idx].offset = @intCast(i32, self.ir.items.len - end_idx - 1);
}

inline fn eof(self: Compiler) bool {
    return self.idx >= self.tokens.len;
}

inline fn emitData(self: *Compiler, data: ir.IR) Error!void {
    try self.ir.append(self.allocator, data);
}

inline fn emitInsn(self: *Compiler, insn: ir.Insn, diff: i32) Error!void {
    try self.emitData(.{ .insn = insn });
    self.stack_size = @intCast(u32, @intCast(i32, self.stack_size) + diff);
}

fn tokenString(self: Compiler, tok: Token) []const u8 {
    return self.src[self.src_idx - tok.len .. self.src_idx];
}

fn peek(self: *Compiler) Error!?Token {
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

fn peekNoEof(self: *Compiler) Error!Token {
    if (try self.peek()) |tok| {
        return tok;
    } else {
        return self.parseError("Unexpected EOF", .{});
    }
}

fn advance(self: *Compiler, tok: Token) void {
    self.idx += 1;
    self.src_idx += tok.len;
}

fn get(self: *Compiler) Error!Token {
    const tok = try self.peekNoEof();
    self.advance(tok);
    return tok;
}

fn getIf(self: *Compiler, comptime tag: Token.Tag) Error!?Token {
    if (try self.peek()) |tok| {
        if (tok.tag == tag) {
            self.advance(tok);
            return tok;
        }
    }
    return null;
}

fn expect(self: *Compiler, comptime expected: []const Token.Tag) Error!Token {
    const tok = try self.get();
    for (expected) |tag| {
        if (tok.tag == tag) return tok;
    }

    return self.parseError("Invalid token: expected one of {any}, but got {}", .{ expected, tok.tag });
}

fn parseError(self: *Compiler, comptime fmt: []const u8, args: anytype) Error {
    // TODO: add source line
    self.err_msg = try std.fmt.allocPrint(self.allocator, fmt, args);
    return error.ParseError;
}

const Mode = enum { expr, stmt, file };

fn testCompiler(
    comptime mode: Mode,
    comptime src: []const u8,
    comptime exp: []const ir.IR,
) !void {
    const allocator = std.testing.allocator;

    var compiler = try Compiler.init(allocator, src);
    defer compiler.deinit();
    errdefer {
        compiler.ir.deinit(allocator);
        compiler.locs.deinit(allocator);
    }

    var scope = try Scope.begin(&compiler, null, 0);
    errdefer scope.deinit(allocator);

    const f = switch (mode) {
        .expr => Compiler.compileExpr,
        .stmt => Compiler.compileStmt,
        .file => Compiler.compileFile,
    };

    const expected = switch (mode) {
        .expr, .stmt => [_]ir.IR{.{ .insn = .nop }} ** 2 ++ exp,
        .file => exp,
    };

    f(&compiler, &scope) catch |err| switch (err) {
        error.ParseError => {
            std.debug.print("{s}\n", .{compiler.err_msg});
            return err;
        },
        else => return err,
    };

    try scope.end(&compiler);

    try std.testing.expect(compiler.eof());

    const insns = compiler.ir.toOwnedSlice(allocator);
    defer allocator.free(insns);

    std.testing.expectEqualSlices(ir.IR, expected, insns) catch |err| {
        for (insns) |x, i| std.debug.print("{} {}\n", .{ i, x });
        return err;
    };
}

test "int" {
    try testCompiler(
        .expr,
        "32",
        &.{ .{ .insn = .push_int }, .{ .int = 32 } },
    );

    try testCompiler(
        .expr,
        "0",
        &.{ .{ .insn = .push_int }, .{ .int = 0 } },
    );
}

test "float" {
    try testCompiler(
        .expr,
        "5.32",
        &.{ .{ .insn = .push_float }, .{ .float = 5.32 } },
    );

    try testCompiler(
        .expr,
        "0.01",
        &.{ .{ .insn = .push_float }, .{ .float = 0.01 } },
    );
}

test "operator precedence" {
    try testCompiler(
        .expr,
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
    try testCompiler(
        .expr,
        "1(2, 3)",
        &.{
            // Target
            .{ .insn = .push_int },
            .{ .int = 1 },
            // Args
            .{ .insn = .push_int },
            .{ .int = 2 },
            .{ .insn = .push_int },
            .{ .int = 3 },
            // Call
            .{ .insn = .call },
            .{ .uint = 2 },
        },
    );
}

test "if else stmt" {
    try testCompiler(
        .stmt,
        "if (1) 2 else 3",
        &.{
            // Cond
            .{ .insn = .push_int },
            .{ .int = 1 },

            // Test cond
            .{ .insn = .jump_if },
            .{ .offset = 2 },
            .{ .insn = .jump },
            .{ .offset = 5 },

            // Body
            .{ .insn = .push_int },
            .{ .int = 2 },
            .{ .insn = .pop },
            .{ .insn = .jump },
            .{ .offset = 3 },

            // Else
            .{ .insn = .push_int },
            .{ .int = 3 },
            .{ .insn = .pop },
        },
    );
}

test "block stmt" {
    try testCompiler(
        .stmt,
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
            .{ .insn = .pop },
        },
    );
}

test "block expr" {
    try testCompiler(
        .expr,
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
            .{ .insn = .pop },

            .{ .insn = .push_void },
        },
    );
}

test "block expr end" {
    try testCompiler(
        .expr,
        "{ 1; 2; end 3; }",
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

test "if stmt" {
    try testCompiler(
        .stmt,
        "if (1) 2",
        &.{
            // Cond
            .{ .insn = .push_int },
            .{ .int = 1 },

            // Test cond
            .{ .insn = .jump_if },
            .{ .offset = 2 },
            .{ .insn = .jump },
            .{ .offset = 3 },

            // Body
            .{ .insn = .push_int },
            .{ .int = 2 },
            .{ .insn = .pop },
        },
    );
}

test "while stmt" {
    try testCompiler(
        .stmt,
        "while (1) { 2; 3; }",
        &.{
            // Cond
            .{ .insn = .push_int },
            .{ .int = 1 },
            .{ .insn = .jump_if_not },
            .{ .offset = 10 },

            // Body
            .{ .insn = .nop },
            .{ .insn = .nop },
            .{ .insn = .push_int },
            .{ .int = 2 },
            .{ .insn = .pop },
            .{ .insn = .push_int },
            .{ .int = 3 },
            .{ .insn = .pop },

            // Jump back to cond
            .{ .insn = .jump },
            .{ .offset = -14 },
        },
    );
}

test "break" {
    try testCompiler(
        .stmt,
        "while (1) { 2; break; 3; }",
        &.{
            // Cond
            .{ .insn = .push_int },
            .{ .int = 1 },
            .{ .insn = .jump_if_not },
            .{ .offset = 12 },

            // Body
            .{ .insn = .nop },
            .{ .insn = .nop },
            .{ .insn = .push_int },
            .{ .int = 2 },
            .{ .insn = .pop },
            // Break
            .{ .insn = .jump },
            .{ .offset = 5 },
            .{ .insn = .push_int },
            .{ .int = 3 },
            .{ .insn = .pop },

            // Jump back to cond
            .{ .insn = .jump },
            .{ .offset = -16 },
        },
    );
}

test "continue" {
    try testCompiler(
        .stmt,
        "while (1) { 2; continue; 3; }",
        &.{
            // Cond
            .{ .insn = .push_int },
            .{ .int = 1 },
            .{ .insn = .jump_if_not },
            .{ .offset = 12 },

            // Body
            .{ .insn = .nop },
            .{ .insn = .nop },
            .{ .insn = .push_int },
            .{ .int = 2 },
            .{ .insn = .pop },
            // Continue
            .{ .insn = .jump },
            .{ .offset = -11 },
            .{ .insn = .push_int },
            .{ .int = 3 },
            .{ .insn = .pop },

            // Jump back to cond
            .{ .insn = .jump },
            .{ .offset = -16 },
        },
    );
}

test "let" {
    try testCompiler(
        .file,
        \\ let x = 5;
        \\ x + 5;
    ,
        &.{
            .{ .insn = .alloc },
            .{ .uint = 1 },

            // Let
            .{ .insn = .push_int },
            .{ .int = 5 },
            .{ .insn = .store },
            .{ .uint = 0 },

            // Load
            .{ .insn = .load },
            .{ .uint = 0 },
            .{ .insn = .push_int },
            .{ .int = 5 },
            .{ .insn = .add },
            .{ .insn = .pop },

            .{ .insn = .dealloc },
            .{ .uint = 1 },
        },
    );
}

test "nested scopes" {
    try testCompiler(
        .file,
        \\ let x = {
        \\     let y = 5;
        \\     let z = 10;
        \\     end y + z;
        \\ };
        \\ x / 2;
    ,
        &.{
            .{ .insn = .alloc },
            .{ .uint = 1 },

            // Inner block
            .{ .insn = .alloc },
            .{ .uint = 2 },

            // Let y
            .{ .insn = .push_int },
            .{ .int = 5 },
            .{ .insn = .store },
            .{ .uint = 1 },

            // Let z
            .{ .insn = .push_int },
            .{ .int = 10 },
            .{ .insn = .store },
            .{ .uint = 2 },

            // End
            .{ .insn = .load },
            .{ .uint = 1 },
            .{ .insn = .load },
            .{ .uint = 2 },
            .{ .insn = .add },

            .{ .insn = .dealloc },
            .{ .uint = 2 },

            // Let x
            .{ .insn = .store },
            .{ .uint = 0 },

            // Load
            .{ .insn = .load },
            .{ .uint = 0 },
            .{ .insn = .push_int },
            .{ .int = 2 },
            .{ .insn = .div },
            .{ .insn = .pop },

            .{ .insn = .dealloc },
            .{ .uint = 1 },
        },
    );
}

test "assign" {
    try testCompiler(
        .file,
        \\ let x = 5;
        \\ x = 10;
    ,
        &.{
            .{ .insn = .alloc },
            .{ .uint = 1 },

            // Let
            .{ .insn = .push_int },
            .{ .int = 5 },
            .{ .insn = .store },
            .{ .uint = 0 },

            // Assign
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
    try testCompiler(
        .file,
        \\ let x = 5;
        \\ x += 3;
    ,
        &.{
            .{ .insn = .alloc },
            .{ .uint = 1 },

            // Let
            .{ .insn = .push_int },
            .{ .int = 5 },
            .{ .insn = .store },
            .{ .uint = 0 },

            // Assign
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
    try testCompiler(
        .file,
        \\ fn inc(x) {
        \\     end x + 1;
        \\ }
        \\ inc(5);
    ,
        &.{
            // Outer scope
            .{ .insn = .alloc },
            .{ .uint = 1 },

            .{ .insn = .func },
            .{ .uint = 11 },

            // Function scope (arg)
            .{ .insn = .alloc },
            .{ .uint = 1 },

            .{ .insn = .load },
            .{ .uint = 0 },
            .{ .insn = .push_int },
            .{ .int = 1 },
            .{ .insn = .add },

            .{ .insn = .dealloc },
            .{ .uint = 1 },

            .{ .insn = .store },
            .{ .uint = 0 },

            .{ .insn = .load },
            .{ .uint = 0 },
            .{ .insn = .push_int },
            .{ .int = 5 },
            .{ .insn = .call },
            .{ .uint = 1 },
            .{ .insn = .pop },

            .{ .insn = .dealloc },
            .{ .uint = 1 },
        },
    );
}

test "capturing fn" {
    try testCompiler(
        .file,
        \\ let x = 5;
        \\ fn addx(y) {
        \\     end x + y;
        \\ }
    ,
        &.{
            .{ .insn = .alloc },
            .{ .uint = 2 },

            // Let
            .{ .insn = .push_int },
            .{ .int = 5 },
            .{ .insn = .store },
            .{ .uint = 0 },

            // Func
            .{ .insn = .func },
            .{ .uint = 11 },

            .{ .insn = .alloc },
            .{ .uint = 1 },

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

            .{ .insn = .store },
            .{ .uint = 1 },

            .{ .insn = .dealloc },
            .{ .uint = 2 },
        },
    );
}
