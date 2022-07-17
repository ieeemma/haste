const std = @import("std");

const ir = @import("ir.zig");
const PackedValue = @import("value.zig").PackedValue;
const Value = @import("value.zig").Value;

const Vm = @This();

allocator: std.mem.Allocator,
ip: u32,
insns: []const ir.IR,
stack: std.ArrayListUnmanaged(PackedValue),

const FrameInfo = struct {
    closure: void,
    base: u32,
};

// This must be kept in sync with `ir.Insn`
// TODO: use a `std.enums.directEnumArray` when stage2 is better at handling dependency loops
const op_fns = [_]fn (*Vm, FrameInfo) void{

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
        self.offset(self.read(.offset));
        return @call(.{ .modifier = .always_tail }, op_fns[self.ip], .{ self, fi });
    }

    fn jumpIf(self: *Vm, fi: FrameInfo) void {
        const off = self.read(.offset);
        const tos = self.stack.pop();
        if (tos.toValue().truthy()) self.offset(off);
        return @call(.{ .modifier = .always_tail }, op_fns[self.ip], .{ self, fi });
    }

    fn jumpIfNot(self: *Vm, fi: FrameInfo) void {
        const off = self.read(.offset);
        const tos = self.stack.pop();
        if (!tos.toValue().truthy()) self.offset(off);
        return @call(.{ .modifier = .always_tail }, op_fns[self.ip], .{ self, fi });
    }

    // Vars

    fn alloc(self: *Vm, fi: FrameInfo) void {
        self.stack.items.len += self.read(.uint);
        // self.stack.appendNTimesAssumeCapacity(0);
        return @call(.{ .modifier = .always_tail }, op_fns[self.ip], .{ self, fi });
    }

    fn dealloc(self: *Vm, fi: FrameInfo) void {
        self.stack.items.len -= self.read(.uint);
        return @call(.{ .modifier = .always_tail }, op_fns[self.ip], .{ self, fi });
    }

    fn load(self: *Vm, fi: FrameInfo) void {
        const idx = self.read(.uint);
        self.stack.appendAssumeCapacity(self.stack.items[fi.base + idx]);
        return @call(.{ .modifier = .always_tail }, op_fns[self.ip], .{ self, fi });
    }

    fn cload(self: *Vm, fi: FrameInfo) void {
        _ = self;
        _ = fi;
        unreachable;
    }

    fn store(self: *Vm, fi: FrameInfo) void {
        const idx = self.read(.uint);
        self.stack.items[fi.base + idx] = self.stack.pop();
        return @call(.{ .modifier = .always_tail }, op_fns[self.ip], .{ self, fi });
    }

    // Literals

    fn pushVoid(self: *Vm, fi: FrameInfo) void {
        _ = self;
        _ = fi;
        unreachable;
    }

    fn pushInt(self: *Vm, fi: FrameInfo) void {
        _ = fi;
        const n = self.read(.int);
        _ = n;
        unreachable;
    }

    fn pushFloat(self: *Vm, fi: FrameInfo) void {
        _ = fi;
        const n = self.read(.float);
        _ = n;
        unreachable;
    }

    // Functions

    fn func(self: *Vm, fi: FrameInfo) void {
        _ = fi;
        const ninsns = self.read(.uint);
        _ = ninsns;
        unreachable;
    }

    fn closure(self: *Vm, fi: FrameInfo) void {
        _ = fi;
        const nvars = self.read(.uint);
        _ = nvars;
        unreachable;
    }

    fn call(self: *Vm, fi: FrameInfo) void {
        _ = fi;
        const nargs = self.read(.uint);
        const f = self.stack.items[self.stack.items.len - 1 - nargs];
        _ = f;
        // assert that nargs == func.nargs

        const new = FrameInfo{
            .closure = undefined,
            .base = @intCast(u32, self.stack.items.len - 1),
        };
        _ = new;

        unreachable;
    }

    const eq = undefined;
    const neq = undefined;
    const lt = undefined;
    const le = undefined;
    const gt = undefined;
    const ge = undefined;
    const add = undefined;
    const sub = undefined;
    const mul = undefined;
    const div = undefined;

    // Stack manipulation

    fn pop(self: *Vm, fi: FrameInfo) void {
        _ = self.stack.pop();
        return @call(.{ .modifier = .always_tail }, op_fns[self.ip], .{ self, fi });
    }

    fn dup(self: *Vm, fi: FrameInfo) void {
        self.stack.appendAssumeCapacity(self.stack.items[self.stack.items.len - 1]);
        return @call(.{ .modifier = .always_tail }, op_fns[self.ip], .{ self, fi });
    }

    // Unused

    fn nop(self: *Vm, fi: FrameInfo) void {
        return @call(.{ .modifier = .always_tail }, op_fns[self.ip], .{ self, fi });
    }
};

fn read(self: *Vm, comptime tag: std.meta.FieldEnum(ir.IR)) std.meta.fieldInfo(ir.IR, tag).field_type {
    const i = self.insns[self.ip];
    self.ip += 1;
    return @field(i, @tagName(tag));
}

fn offset(self: *Vm, off: i32) void {
    self.ip = @intCast(u32, @intCast(i64, self.ip) + off);
}

test {
    _ = op_fns;
}
