const std = @import("std");
const expect = std.testing.expect;
const testing = std.testing;

const c = @cImport({
    @cInclude("cblas.h");
});

const tensor = @import("tensor.zig");
const operations = @import("tensor_operations/operations.zig");
const gradients = @import("gradients.zig");

test "Main tests" {
    var a1 = [_]u8{ 1, 2 };
    var a2 = [_]u8{ 3, 4 };
    const v = c.sdot(a1, 2, 8, a2, 8);
    std.debug.print("{:?}", v);
    // _ = @import("tensor_operations/operations.zig");
}
