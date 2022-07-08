const std = @import("std");
const builtin = @import("builtin");

pub const Insn = enum(u32) {

    // Control flow
    jump,
    jump_if,
    jump_if_not,

    // Vars
    alloc,
    load,
    store,

    // Literals
    push_int,
    push_float,

    // Misc
    toplevel_func,
    func,
    call,

    // Ops,
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
    _,
};

pub const IR = if (builtin.is_test) union(enum) {
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
    imports: []const []const u8,
    ir: []const IR,
    locs: []const u32,
};
