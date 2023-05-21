const std = @import("std");

var tensorId: usize = 1;

pub fn TensorBuilder(comptime shape: []const u8, comptime T: type) type {
    comptime var copied_shape = [_]u8{0} ** 5;
    comptime {
        for (shape) |s, i| {
            copied_shape[i] = s;
        }
    }
    return Tensor(copied_shape, T);
}

pub fn Tensor(comptime t_shape: [5]u8, comptime T: type) type {
    comptime var temp_data_len: usize = 1;
    comptime {
        for (t_shape) |s| {
            if (s != 0) {
                temp_data_len *= s;
            }
        }
    }

    return struct {
        data: []T,
        id: usize,

        pub const shape = t_shape;
        pub const data_len = temp_data_len;
        pub const Type = T;
        pub const Self = @This();

        pub fn init(set_data: []T) Self {
            if (data_len != set_data.len) {
                @panic("Tensor data is not the correct shape");
            }
            const tensor_id = @atomicRmw(usize, &tensorId, std.builtin.AtomicRmwOp.Add, 1, std.builtin.AtomicOrder.AcqRel);
            return Self{
                .data = set_data,
                .id = tensor_id,
            };
        }

        pub fn debug(self: *const Self) void {
            std.debug.print("Id: {any}\n", .{self.id});
            std.debug.print("Shape: {any}\n", .{shape});
            std.debug.print("data_len: {any}\n", .{data_len});
            std.debug.print("Data: {any}\n", .{self.data});
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            allocator.free(self.data);
        }
    };
}
