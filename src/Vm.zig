const std = @import("std");

const ir = @import("ir.zig");

const Vm = @This();

allocator: std.mem.Allocator,
ip: u32,
stack: std.ArrayListUnmanaged(Value),

const Value = u64;

const FrameInfo = struct {
    closure: void,
    base: u32,
};

// This must be kept in sync with `ir.Insn`
// TODO: use a `std.enums.directEnumArray` when stage2 is better at handling dependency loops
const insns = [_]fn (*Vm, FrameInfo) void{

    // Control flow
    op.jump,
    op.jumpIf,
    op.jumpIfNot,

    // Vars
    op.alloc,
    op.dealloc,
    op.load,
    op.cload,
    op.store,

    // Literals
    op.pushVoid,
    op.pushInt,
    op.pushFloat,

    // Functions
    op.func,
    op.closure,
    op.call,

    // Arithmetic and comparison
    op.eq,
    op.neq,
    op.lt,
    op.le,
    op.gt,
    op.ge,
    op.add,
    op.sub,
    op.mul,
    op.div,

    // Stack manipulation
    op.pop,
    op.dup,

    // Unused
    op.nop,
};

const op = opaque {

    // Control flow

    fn jump(self: *Vm, fi: FrameInfo) void {
        self.ip += self.load(.offset);
        return @call(.{ .modifier = .always_tail }, insns[self.ip], .{ self, fi });
    }

    fn jumpIf(self: *Vm, fi: FrameInfo) void {
        const off = self.load(.offset);
        const tos = self.pop();
        if (self.truthy(tos)) self.ip += off;
        return @call(.{ .modifier = .always_tail }, insns[self.ip], .{ self, fi });
    }

    fn jumpIfNot(self: *Vm, fi: FrameInfo) void {
        const off = self.load(.offset);
        const tos = self.pop();
        if (!self.truthy(tos)) self.ip += off;
        return @call(.{ .modifier = .always_tail }, insns[self.ip], .{ self, fi });
    }

    // Vars

    fn alloc(self: *Vm, fi: FrameInfo) void {
        self.stack.items.len += self.load(.uint);
        // self.stack.appendNTimesAssumeCapacity(0);
        return @call(.{ .modifier = .always_tail }, insns[self.ip], .{ self, fi });
    }

    fn dealloc(self: *Vm, fi: FrameInfo) void {
        self.stack.items.len -= self.load(.uint);
        return @call(.{ .modifier = .always_tail }, insns[self.ip], .{ self, fi });
    }

    fn load(self: *Vm, fi: FrameInfo) void {
        const idx = self.load(.uint);
        self.stack.appendAssumeCapacity(self.stack.items[fi.base + idx]);
        return @call(.{ .modifier = .always_tail }, insns[self.ip], .{ self, fi });
    }

    fn cload(self: *Vm, fi: FrameInfo) void {
        unreachable;
    }

    fn store(self: *Vm, fi: FrameInfo) void {
        const idx = self.read(.uint);
        self.stack[fi.base + idx] = self.pop();
        return @call(.{ .modifier = .always_tail }, insns[self.ip], .{ self, fi });
    }

    // Literals

    fn pushVoid(self: *Vm, fi: FrameInfo) void {
        unreachable;
    }

    fn pushInt(self: *Vm, fi: FrameInfo) void {
        const n = self.read(.int);
        unreachable;
    }

    fn pushFloat(self: *Vm, fi: FrameInfo) void {
        const n = self.read(.float);
        unreachable;
    }

    // Functions

    fn func(self: *Vm, fi: FrameInfo) void {
        const ninsns = self.load(.uint);
        unreachable;
    }

    fn closure(self: *Vm, fi: FrameInfo) void {
        const nvars = self.load(.uint);
        unreachable;
    }

    fn call(self: *Vm, fi: FrameInfo) void {
        const nargs = self.read(.uint);
        const func = self.stack.items[self.stack.items.len - 1 - nargs];
        // assert that nargs == func.nargs

        const new = FrameInfo{ .closure = undefined, .base = self.stack.items.len - 1 };

        unreachable;
    }

    // Stack manipulation

    fn pop(self: *Vm, fi: FrameInfo) void {
        _ = self.stack.pop();
        return @call(.{ .modifier = .always_tail }, insns[self.ip], .{ self, fi });
    }

    fn dup(self: *Vm, fi: FrameInfo) void {
        self.stack.appendAssumeCapacity(self.stack.items[self.stack.items - 1]);
        return @call(.{ .modifier = .always_tail }, insns[self.ip], .{ self, fi });
    }

    // Unused

    fn nop(self: *Vm, fi: FrameInfo) void {
        return @call(.{ .modifier = .always_tail }, insns[self.ip], .{ self, fi });
    }
};

fn load(self: *Vm, comptime tag: std.meta.FieldEnum(ir.IR)) std.meta.fieldInfo(ir.IR, tag).field_type {
    const i = self.insns[self.ip];
    self.ip += 1;
    return @field(i, @tagName(tag));
}

fn truthy(self: *Vm, value: Value) bool {
    unreachable;
}
