const std = @import("std");
const ArrayList = std.ArrayList;

pub const HashMap = std.AutoHashMap;

const OperationTypesTags = enum { SingleTensorWithData, SingleTensorWithConstant, SingleTensor, DualTensorWithData, DualTensor };
pub fn OperationTypes(comptime T: type) type {
    return union(OperationTypesTags) { SingleTensorWithData: struct {
        t1_data: []T,
        backwards_function: *const fn (t1_id: usize, t1_data: []T, tnew_id: usize, grads: *HashMap(usize, []T), std.mem.Allocator) std.mem.Allocator.Error!void,
    }, SingleTensorWithConstant: struct {
        c: T,
        backwards_function: *const fn (t1_id: usize, constant: T, tnew_id: usize, grads: *HashMap(usize, []T), std.mem.Allocator) std.mem.Allocator.Error!void,
    }, SingleTensor: struct {
        backwards_function: *const fn (t1_id: usize, tnew_id: usize, grads: *HashMap(usize, []T), std.mem.Allocator) std.mem.Allocator.Error!void,
    }, DualTensorWithData: struct {
        t1_data: []T,
        t2_id: usize,
        t2_data: []T,
        backwards_function: *const fn (t1_id: usize, t1_data: []T, t2_id: usize, t2_data: []T, tnew_id: usize, grads: *HashMap(usize, []T), i_allocator: std.mem.Allocator) std.mem.Allocator.Error!void,
    }, DualTensor: struct {
        t2_id: usize,
        backwards_function: *const fn (t1_id: usize, t2_id: usize, tnew_id: usize, grads: *HashMap(usize, []T), i_allocator: std.mem.Allocator) std.mem.Allocator.Error!void,
    } };
}

pub fn Operation(comptime T: type) type {
    return struct {
        t1_id: usize,
        tnew_id: usize,
        op: OperationTypes(T),
        const Self = @This();

        fn do_backwards(self: *const Self, grads: *HashMap(usize, []T), allocator: std.mem.Allocator) std.mem.Allocator.Error!void {
            switch (self.op) {
                OperationTypesTags.SingleTensorWithData => |op| try op.backwards_function(self.t1_id, op.t1_data, self.tnew_id, grads, allocator),
                OperationTypesTags.SingleTensorWithConstant => |op| try op.backwards_function(self.t1_id, op.c, self.tnew_id, grads, allocator),
                OperationTypesTags.SingleTensor => |op| try op.backwards_function(self.t1_id, self.tnew_id, grads, allocator),
                OperationTypesTags.DualTensorWithData => |op| try op.backwards_function(self.t1_id, op.t1_data, op.t2_id, op.t2_data, self.tnew_id, grads, allocator),
                OperationTypesTags.DualTensor => |op| try op.backwards_function(self.t1_id, op.t2_id, self.tnew_id, grads, allocator),
            }
        }

        fn sortAsc(context: void, lhs: Self, rhs: Self) bool {
            _ = context;
            return lhs.tnew_id > rhs.tnew_id;
        }
    };
}

pub fn Tape(comptime T: type) type {
    return struct {
        operations: ArrayList(Operation(T)),
        const Self = @This();

        pub fn new(allocator: std.mem.Allocator) Self {
            return Self{
                .operations = ArrayList(Operation(T)).init(allocator),
            };
        }

        pub fn append(self: *Self, op: Operation(T)) !void {
            try self.operations.append(op);
        }

        pub fn backwards(self: *Self, allocator: std.mem.Allocator) std.mem.Allocator.Error!HashMap(usize, []T) {
            var grads = HashMap(usize, []T).init(allocator);
            var ops = self.operations.toOwnedSlice();
            std.sort.sort(Operation(T), ops, {}, Operation(T).sortAsc);
            for (ops) |op| {
                try op.do_backwards(&grads, allocator);
            }
            allocator.free(ops);
            return grads;
        }

        pub fn deinit(self: *Self) void {
            self.operations.deinit();
        }
    };
}
