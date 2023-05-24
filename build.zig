const std = @import("std");

pub fn build(b: *std.build.Builder) void {
    // Standard release options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall.
    const mode = b.standardReleaseOptions();

    // const openblas_include_dir = "/home/olabian/opt/openblas/include";
    // const openblas_lib_dir = "/home/olabian/opt/openblas/lib";

    const lib = b.addStaticLibrary("dfdz", "src/main.zig");
    lib.setBuildMode(mode);
    // lib.linkSystemLibrary("openblas");
    lib.linkSystemLibrary("/nix/store/d6aabmvg0avs71shmxjnvln94sprqg4w-openblas-0.3.20/lib");
    lib.install();

    const main_tests = b.addTest("src/main.zig");
    main_tests.setBuildMode(mode);

    const test_step = b.step("test", "Run library tests");
    test_step.dependOn(&main_tests.step);
}
