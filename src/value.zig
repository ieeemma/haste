const std = @import("std");

pub const PackedValue = packed union {
    float: f64,
    int: packed struct {
        value: i51,
        exp: u12 = 0xfff,
        sign: bool = true,
    },
    tagged: packed struct {
        ptr: u48,
        tag: Tag,
        exp: u12 = 0xfff,
        sign: bool = false,
    },

    comptime {
        std.debug.assert(@bitSizeOf(PackedValue) == 64);
    }

    const Tag = enum(u3) {
        object,
        array,
        map,
        set,
    };

    pub fn toValue(self: PackedValue) Value {
        if (!std.math.isNan(self.float)) {
            return .{ .float = self.float };
        }
        if (self.int.sign) {
            return .{ .int = self.int.value };
        }

        if (self.tagged.ptr == 0) {
            // Sentinel value
            return switch (self.tagged.tag) {
                .object => .void,
                .array => .{ .bool = true },
                .map => .{ .bool = false },
                .set => .{ .float = std.math.nan(f64) },
            };
        } else {
            // Pointer value
            return switch (self.tagged.tag) {
                .object => unreachable,
                .array => .{ .array = @intToPtr(*Array, self.tagged.ptr) },
                .map => .{ .map = @intToPtr(*Map, self.tagged.ptr) },
                .set => .{ .set = @intToPtr(*Set, self.tagged.ptr) },
            };
        }
    }
};

pub const Value = union(enum) {
    float: f64,
    int: i51,
    void: void,
    bool: bool,
    object: *Object,
    array: *Array,
    map: *Map,
    set: *Set,

    pub fn hash(self: Value) u64 {
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(&.{@enumToInt(self)});
        switch (self) {
            .float => |n| std.hash.autoHash(&hasher, @bitCast(u64, n)),
            .int => |n| std.hash.autoHash(&hasher, n),
            .void => {},
            .bool => |b| hasher.update(&.{@boolToInt(b)}),
            // TODO: immutable variants of compound types
            else => unreachable,
        }
        return hasher.final();
    }

    pub fn eql(self: Value, other: Value) bool {
        if (self == .int and other == .float) {
            return @intToFloat(f64, self.int) == other.float;
        } else if (self == .float and other == .int) {
            return self.float == @intToFloat(f64, other.int);
        }

        if (std.meta.activeTag(self) != other) return false;

        switch (self) {
            .float => |f| return f == other.float,
            .int => |i| return i == other.int,
            .void => return true,
            .bool => |b| return b == other.bool,
            .object => unreachable,
            .array => |a| {
                if (a.items.len != other.array.items.len) return false;
                for (a.items) |item, i| {
                    if (!item.toValue().eql(other.array.items[i].toValue())) return false;
                }
                return true;
            },
            .map => |m| {
                if (m.count() != other.map.count()) return false;
                var it = m.iterator();
                while (it.next()) |kv| {
                    if (other.map.get(kv.key_ptr.*)) |item| {
                        if (kv.value_ptr.toValue().eql(item.toValue())) continue;
                    }
                    return false;
                }
                return true;
            },
            .set => |s| {
                if (s.count() != other.set.count()) return false;
                var it = s.iterator();
                while (it.next()) |kv| {
                    if (other.map.get(kv.key_ptr.*) == null) return false;
                }
                return true;
            },
        }
    }

    pub fn truthy(self: Value) bool {
        return switch (self) {
            .float => |f| f != 0,
            .int => |i| i != 0,
            .void => false,
            .bool => |b| b,
            .object => unreachable,
            .array => |a| a.items.len > 0,
            .map => |m| m.count() > 0,
            .set => |s| s.count() > 0,
        };
    }
};

pub const Object = void;

pub const Array = std.ArrayListUnmanaged(PackedValue);

pub const Map = std.ArrayHashMapUnmanaged(
    PackedValue,
    PackedValue,
    Ctx,
    true,
);

pub const Set = std.ArrayHashMapUnmanaged(
    PackedValue,
    void,
    Ctx,
    true,
);

const Ctx = struct {
    pub fn hash(_: Ctx, v: PackedValue) u32 {
        return @truncate(u32, v.toValue().hash());
    }

    pub fn eql(_: Ctx, a: PackedValue, b: PackedValue, _: usize) bool {
        return a.toValue().eql(b.toValue());
    }
};
