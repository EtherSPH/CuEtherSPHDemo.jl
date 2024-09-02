#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2024/08/29 16:54:18
  @ license: MIT
  @ description:
 =#

module CuEtherSPHDemo

using CUDA
using ExportAll
using WriteVTK
using Dates
using ProgressBars
using PyCall

const IntType = Int32
const RealType = Float32
const BoolType = Bool

const FLUID_TAG::IntType = 1
const WALL_TAG::IntType = 2

include("NeighbourCellSystem.jl")
include("ParticleSystem.jl")
include("NeighbourSearch.jl")
include("SPHKernel.jl")
include("ContinuityAndPressureExtrapolation.jl")
include("UpdateDensityAndPressure.jl")
include("Momentum.jl")
include("AccelerateAndMove.jl")
include("DensityFilter.jl")
include("VTPWriter.jl")
include("VTPReader.jl")

@inline function greet()::Nothing
    println("Hello, CuEtherSPHDemo!")
    println("    CUDA Version Information: $(CUDA.versioninfo())")
    return nothing
end

ExportAll.@exportAll()

end # module CuEtherSPHDemo
