const std = @import("std");
const tensor = @import("../tensor.zig");
const gradients = @import("../gradients.zig");
const helpers = @import("helpers.zig");

fn multiplyTensorConstant(t1: anytype, c: @TypeOf(t1).Type, allocator: std.mem.Allocator, optional_tape: ?*gradients.Tape(@TypeOf(t1).Type)) !tensor.TensorBuilder(&@TypeOf(t1).shape, @TypeOf(t1).Type) {
    // Actual Operation
    var new_tensor_data = try allocator.alloc(@TypeOf(t1).Type, @TypeOf(t1).data_len);
    for (new_tensor_data) |*item, i| {
        item.* = t1.data[i] * c;
    }
    const new_tensor = tensor.TensorBuilder(&@TypeOf(t1).shape, @TypeOf(t1).Type).init(new_tensor_data[0..]);

    // Prep backwards
    if (optional_tape) |tape| {
        const temp = struct {
            fn backwards(t1_id: usize, constant: @TypeOf(t1).Type, tnew_id: usize, grads: *gradients.HashMap(usize, []@TypeOf(t1).Type), i_allocator: std.mem.Allocator) std.mem.Allocator.Error!void {
                const parent_grads = grads.get(tnew_id);

                if (grads.get(t1_id)) |old_t1_grads| {
                    if (parent_grads) |pg| {
                        for (old_t1_grads) |*item, i| item.* += constant * pg[i];
                    } else {
                        for (old_t1_grads) |*item| item.* += constant;
                    }
                } else {
                    var t1_grads = try i_allocator.alloc(@TypeOf(t1).Type, @TypeOf(t1).data_len);
                    if (parent_grads) |pg| {
                        for (t1_grads) |*item, i| item.* = constant * pg[i];
                    } else {
                        for (t1_grads) |*item| item.* = constant;
                    }
                    try grads.put(t1_id, t1_grads);
                }
            }
        };
        const new_operation = gradients.Operation(@TypeOf(t1).Type){
            .t1_id = t1.id,
            .tnew_id = new_tensor.id,
            .op = gradients.OperationTypes(@TypeOf(t1).Type){ .SingleTensorWithConstant = .{
                .c = c,
                .backwards_function = temp.backwards,
            } },
        };
        try tape.append(new_operation);
    }

    return new_tensor;
}

pub fn multiplyTensorTensor(t1: anytype, t2: @TypeOf(t1), allocator: std.mem.Allocator, optional_tape: ?*gradients.Tape(@TypeOf(t1).Type)) !tensor.TensorBuilder(&@TypeOf(t1).shape, @TypeOf(t1).Type) {
    // Actual operation
    var new_tensor_data = try allocator.alloc(@TypeOf(t1).Type, @TypeOf(t1).data_len);
    for (new_tensor_data) |*item, i| {
        item.* = t1.data[i] * t2.data[i];
    }
    const new_tensor = tensor.TensorBuilder(&@TypeOf(t1).shape, @TypeOf(t1).Type).init(new_tensor_data[0..]);

    // Prep backwards
    if (optional_tape) |tape| {
        const temp = struct {
            fn backwards(t1_id: usize, t1_data: []@TypeOf(t1).Type, t2_id: usize, t2_data: []@TypeOf(t2).Type, tnew_id: usize, grads: *gradients.HashMap(usize, []@TypeOf(t1).Type), i_allocator: std.mem.Allocator) std.mem.Allocator.Error!void {
                const parent_grads = grads.get(tnew_id);

                if (grads.get(t1_id)) |old_t1_grads| {
                    if (parent_grads) |pg| {
                        for (old_t1_grads) |*item, i| item.* += t2_data[i] * pg[i];
                    } else {
                        for (old_t1_grads) |*item, i| item.* += t2_data[i];
                    }
                } else {
                    var t1_grads = try i_allocator.alloc(@TypeOf(t1).Type, @TypeOf(t1).data_len);
                    if (parent_grads) |pg| {
                        for (t1_grads) |*item, i| item.* = t2_data[i] * pg[i];
                    } else {
                        for (t1_grads) |*item, i| item.* = t2_data[i];
                    }
                    try grads.put(t1_id, t1_grads);
                }

                if (grads.get(t2_id)) |old_t2_grads| {
                    if (parent_grads) |pg| {
                        for (old_t2_grads) |*item, i| item.* += t1_data[i] * pg[i];
                    } else {
                        for (old_t2_grads) |*item, i| item.* += t1_data[i];
                    }
                } else {
                    var t2_grads = try i_allocator.alloc(@TypeOf(t1).Type, @TypeOf(t1).data_len);
                    if (parent_grads) |pg| {
                        for (t2_grads) |*item, i| item.* = t1_data[i] * pg[i];
                    } else {
                        for (t2_grads) |*item, i| item.* = t1_data[i];
                    }
                    try grads.put(t2_id, t2_grads);
                }
            }
        };
        const new_operation = gradients.Operation(@TypeOf(t1).Type){
            .t1_id = t1.id,
            .tnew_id = new_tensor.id,
            .op = gradients.OperationTypes(@TypeOf(t1).Type){ .DualTensorWithData = .{
                .t1_data = t1.data,
                .t2_id = t2.id,
                .t2_data = t2.data,
                .backwards_function = temp.backwards,
            } },
        };
        try tape.append(new_operation);
    }

    // Return the new tensor
    return new_tensor;
}

fn multiply(t1: anytype, t2: anytype, allocator: std.mem.Allocator, optional_tape: ?*gradients.Tape(@TypeOf(t1).Type)) !tensor.TensorBuilder(&@TypeOf(t1).shape, @TypeOf(t1).Type) {
    if (@TypeOf(t1) == @TypeOf(t2)) {
        return multiplyTensorTensor(t1, t2, allocator, optional_tape);
    } else {
        return multiplyTensorConstant(t1, t2, allocator, optional_tape);
    }
}

test "tensor x tensor multiply 1" {
    var allocator = std.testing.allocator;
    var tape = gradients.Tape(f32).new(allocator);
    defer tape.deinit();

    var t1_data = [_]f32{ 1.1, 2.2, 3.3 };
    var t1 = tensor.TensorBuilder(&[_]u8{3}, f32).init(&t1_data);

    var t2_data = [_]f32{ 3.3, 3.3, 3.3 };
    var t2 = tensor.TensorBuilder(&[_]u8{3}, f32).init(&t2_data);

    var t3 = try multiply(t1, t2, allocator, &tape);
    defer t3.deinit(allocator);

    var grads = try tape.backwards(allocator);
    defer grads.deinit();

    const t1_grads = if (grads.fetchRemove(t1.id)) |g| g.value else unreachable;
    defer allocator.free(t1_grads);
    try std.testing.expectEqualSlices(f32, t1_grads, &t2_data);

    const t2_grads = if (grads.fetchRemove(t2.id)) |g| g.value else unreachable;
    defer allocator.free(t2_grads);
    try std.testing.expectEqualSlices(f32, t2_grads, &t1_data);
}

test "tensor x constant multiply 1" {
    var allocator = std.testing.allocator;
    var tape = gradients.Tape(f32).new(allocator);
    defer tape.deinit();

    var t1_data = [_]f32{ 1.1, 2.2, 3.3 };
    var t1 = tensor.TensorBuilder(&[_]u8{3}, f32).init(&t1_data);

    const x: f32 = 10.0;
    var t2 = try multiply(t1, x, allocator, &tape);
    defer t2.deinit(allocator);

    var grads = try tape.backwards(allocator);
    defer grads.deinit();

    const t1_grads = if (grads.fetchRemove(t1.id)) |g| g.value else unreachable;
    defer allocator.free(t1_grads);
    for (t1_grads) |g| {
        try std.testing.expectApproxEqRel(g, 10.0, 0.0000001);
    }
}

// test "tensor x tensor multiply 2" {
//     var allocator = std.testing.allocator;
//     var tape = gradients.Tape(f32).new(allocator);
//     defer tape.deinit();
//
//     var t1_data = [_]f32{ 1.1, 2.2, 3.3 };
//     var t1 = tensor.TensorBuilder(&[_]u8{3}, f32).init(&t1_data);
//
//     var t2_data = [_]f32{ 3.3, 3.3, 3.3 };
//     var t2 = tensor.TensorBuilder(&[_]u8{3}, f32).init(&t2_data);
//
//     var t3_data = [_]f32{ 3.75, 3.65, 3.45 };
//     var t3 = tensor.TensorBuilder(&[_]u8{3}, f32).init(&t3_data);
//
//     var t4 = try multiply(t1, t2, allocator, &tape);
//     defer t4.deinit(allocator);
//
//     var t5 = try multiply(t4, t3, allocator, &tape);
//     defer t5.deinit(allocator);
//
//     var grads = try tape.backwards(allocator);
//     defer grads.deinit();
//
//     const t1_grads = if (grads.fetchRemove(t1.id)) |g| g.value else unreachable;
//     defer allocator.free(t1_grads);
//     // try std.testing.expectEqualSlices(f32, t1_grads, &t2_data);
//
//     const t2_grads = if (grads.fetchRemove(t2.id)) |g| g.value else unreachable;
//     defer allocator.free(t2_grads);
//     // try std.testing.expectEqualSlices(f32, t2_grads, &t1_data);
//
//     const t3_grads = if (grads.fetchRemove(t3.id)) |g| g.value else unreachable;
//     defer allocator.free(t3_grads);
//     // try std.testing.expectEqualSlices(f32, t3_grads, &t2_data);
//
//     const t4_grads = if (grads.fetchRemove(t4.id)) |g| g.value else unreachable;
//     defer allocator.free(t4_grads);
//     // try std.testing.expectEqualSlices(f32, t4_grads, &t1_data);
// }
