const std = @import("std");
const tensor = @import("../tensor.zig");
const gradients = @import("../gradients.zig");
const helpers = @import("helpers.zig");

pub fn add(t1: anytype, t2: anytype, allocator: std.mem.Allocator, optional_tape: ?*gradients.Tape(@TypeOf(t1).Type)) !tensor.TensorBuilder(&@TypeOf(t1).shape, @TypeOf(t1).Type) {
    // Comptime checks
    comptime {
        if (@TypeOf(t1).data_len != @TypeOf(t2).data_len) {
            @panic("Cannot add two tensors with different data lengths");
        } else if (@TypeOf(t1).Type != @TypeOf(t2).Type) {
            @panic("Cannot add two tensors of different types");
        }
    }

    // Actual operation
    var new_tensor_data = try allocator.alloc(@TypeOf(t1).Type, @TypeOf(t1).data_len);
    for (new_tensor_data) |*item, i| {
        item.* = t1.data[i] + t2.data[i];
    }
    const new_tensor = tensor.TensorBuilder(&@TypeOf(t1).shape, @TypeOf(t1).Type).init(new_tensor_data[0..]);

    // Prep backwards
    if (optional_tape) |tape| {
        const temp = struct {
            fn backwards(t1_id: usize, t2_id: ?usize, tnew_id: usize, grads: *gradients.HashMap(usize, []@TypeOf(t1).Type), i_allocator: std.mem.Allocator) std.mem.Allocator.Error!void {
                if (grads.get(tnew_id)) |parent_grads| {
                    var t1_grads = if (grads.get(t1_id)) |t1_g| t1_g else try helpers.allocateTypeLengthValue(@TypeOf(t1).Type, @TypeOf(t1).data_len, 1, i_allocator);
                    helpers.elementWiseAdd(@TypeOf(t1).Type, t1_grads, parent_grads);
                    try grads.put(t1_id, t1_grads);
                    if (t2_id) |t_id| {
                        var t2_grads = if (grads.get(t_id)) |t2_g| t2_g else try helpers.allocateTypeLengthValue(@TypeOf(t2).Type, @TypeOf(t2).data_len, 1, i_allocator);
                        helpers.elementWiseAdd(@TypeOf(t1).Type, t2_grads, parent_grads);
                        try grads.put(t_id, t2_grads);
                    }
                } else {
                    var t1_grads = if (grads.get(t1_id)) |t1_g| t1_g else try helpers.allocateTypeLengthValue(@TypeOf(t1).Type, @TypeOf(t1).data_len, 1, i_allocator);
                    try grads.put(t1_id, t1_grads);
                    if (t2_id) |t_id| {
                        var t2_grads = if (grads.get(t_id)) |t2_g| t2_g else try helpers.allocateTypeLengthValue(@TypeOf(t2).Type, @TypeOf(t2).data_len, 1, i_allocator);
                        try grads.put(t_id, t2_grads);
                    }
                }
            }
        };
        const new_operation = gradients.Operation(@TypeOf(t1).Type){
            .t1_id = t1.id,
            .t1_data = t1.data,
            .t2_id = t2.id,
            .t2_data = t2.data,
            .tnew_id = new_tensor.id,
            .backwards_function = gradients.BackwardsFunctionTypes(@TypeOf(t1).Type){ .without_data = temp.backwards },
        };
        try tape.append(new_operation);
    }

    // Return the new tensor
    return new_tensor;
}
