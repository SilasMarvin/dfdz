const std = @import("std");

pub fn allocateTypeLengthValue(comptime T: type, length: usize, value: T, allocator: std.mem.Allocator) std.mem.Allocator.Error![]T {
    var ret_data = try allocator.alloc(T, length);
    for (ret_data) |*item| {
        item.* = value;
    }
    return ret_data;
}

pub fn elementWiseMultiply(comptime T: type, d1: []T, d2: []T) void {
    for (d1) |_, i| {
        d1[i] *= d2[i];
    }
}

pub fn elementWiseAdd(comptime T: type, d1: []T, d2: []T) void {
    for (d1) |_, i| {
        d1[i] += d2[i];
    }
}
