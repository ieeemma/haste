const std = @import("std");

pub const IR = enum(u32) {

    // Control flow
    jump,
    jumpc,

    // Vars
    load,
    store,

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

    // Unused
    _,
};

const Module = struct {
    imports: []const []const u8,
    ir: []const IR,
    locs: []const u32,
};
