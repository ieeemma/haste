const std = @import("std");
const builtin = @import("builtin");

pub const Insn = enum(u32) {

    // Control flow
    jump,
    jump_if,
    jump_if_not,

    // Vars
    alloc,
    dealloc,
    load,
    cload,
    store,

    // Literals
    push_void,
    push_int,
    push_float,

    // Functions
    func,
    closure,
    call,

    // Arithmetic and comparison
    eq,
    neq,
    lt,
    le,
    gt,
    ge,
    add,
    sub,
    mul,
    div,

    // Stack manipulation
    pop,
    dup,

    // Unused
    nop,
    _,
};

pub const IR = if (builtin.is_test or builtin.mode == .Debug) union(enum) {
    insn: Insn,
    uint: u32,
    int: i32,
    float: f32,
    offset: i32,
} else union {
    insn: Insn,
    uint: u32,
    int: i32,
    float: f32,
    offset: i32,
};

pub const Module = struct {
    arena: std.heap.ArenaAllocator,
    imports: []const []const u8,
    ir: []const IR,
    locs: []const u32,

    pub fn deinit(self: Module) void {
        self.arena.deinit();
    }
};
