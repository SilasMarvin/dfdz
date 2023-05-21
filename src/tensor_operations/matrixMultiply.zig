const tensor = @import("../tensor.zig");

fn buildMatMulShape(comptime t1_shape: []const u8, comptime t2_shape: []const u8) [2]u8 {
    comptime {
        if (t1_shape[1] != t2_shape[0]) {
            @panic("Cannot matrix multiply tensors with the current shape");
        }
    }
    const shape = [2]u8{ t2_shape[1], t2_shape[0] };
    return shape;
}

pub fn matrixMultiply(t1: anytype, t2: anytype) tensor.TensorBuilder(&buildMatMulShape(&@TypeOf(t1).shape, &@TypeOf(t2).shape), @TypeOf(t1).Type) {
    return tensor.TensorBuilder(&buildMatMulShape(&@TypeOf(t1).shape, &@TypeOf(t2).shape), @TypeOf(t1).Type).init(&[_]f32{ 1, 2, 3, 4, 5, 6 });
}
