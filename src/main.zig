const std = @import("std");
const expect = std.testing.expect;
const testing = std.testing;

const tensor = @import("tensor.zig");
const operations = @import("tensor_operations/operations.zig");
const gradients = @import("gradients.zig");

test "Main tests" {
    _ = @import("tensor_operations/operations.zig");
}
