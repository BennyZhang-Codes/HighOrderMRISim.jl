# Optional host-side inference/allocation analysis. Development tools are loaded
# from HIGHORDERMRISIM_TOOL_ENV, not added as package runtime dependencies.
haskey(ENV, "HIGHORDERMRISIM_TOOL_ENV") && push!(LOAD_PATH, ENV["HIGHORDERMRISIM_TOOL_ENV"])
using HighOrderMRISim
using JET
using AllocCheck
using Profile
include("performance_workload.jl")

function analyze_reference()
    JET.JET_AVAILABLE || error("JET does not support this Julia version.")
    workload = high_order_workload(; nspins=1024)
    args = (workload.obj, workload.hoseqd, workload.signal,
            workload.state, BlochHighOrder("1111"))
    operation = HighOrderMRISim.run_spin_precession!
    operation(args...)
    println("Reference CPU bytes: ", @allocated operation(args...))
    println("JET reference analysis:")
    display(JET.report_opt(operation, Base.typesof(args...); target_modules=(HighOrderMRISim,)))
    println("AllocCheck reference analysis:")
    issues = AllocCheck.check_allocs(operation, Base.typesof(args...))
    println("Potential allocation/dynamic-call sites: ", length(issues))
    foreach(println, Iterators.take(issues, 5))
    Profile.Allocs.clear()
    Profile.Allocs.@profile sample_rate=0.1 operation(args...)
    Profile.Allocs.print()
    println("REFERENCE_STATIC_ANALYSIS_OK")
    return nothing
end

function analyze_kernel_math()
    basis = HighOrderMRISim.high_order_basis(0.02, -0.03, 0.04)
    coefficients = ntuple(_ -> fill(1e-5, 3), 16)
    gradients = ntuple(_ -> fill(2e-4, 3), 3)
    for (operation, args) in (
        (HighOrderMRISim.high_order_basis, (0.02, -0.03, 0.04)),
        (HighOrderMRISim.high_order_field,
         (basis, coefficients, gradients, 20.0, 2, Val((true, true, true, true, true)))),
        (HighOrderMRISim.rotate_high_order_spin, (0.3+0.6im, 0.5, 8e-6+2e-6im, 2e-5, 1e-5)),
    )
        println("Kernel math: ", operation)
        display(JET.report_opt(operation, Base.typesof(args...)))
        issues = AllocCheck.check_allocs(operation, Base.typesof(args...))
        println("AllocCheck sites: ", length(issues))
        isempty(issues) || error("Kernel scalar math may allocate or dispatch dynamically.")
    end
    println("KERNEL_MATH_STATIC_ANALYSIS_OK")
    return nothing
end

isempty(ARGS) ? analyze_reference() : only(ARGS) == "kernel" ? analyze_kernel_math() : error("Expected no arguments or kernel.")
