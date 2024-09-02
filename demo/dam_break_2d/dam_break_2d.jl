#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2024/09/02 15:07:10
  @ license: MIT
  @ description:
 =#

using CuEtherSPHDemo
using ProgressBars
using CUDA

CUDA.device!(1)
const n_threads = 512

const dim::IntType = 2
const dr::RealType = 0.01
const gap = dr
const h = 3 * dr
const gx::RealType = 0.0
const gy::RealType = -9.8
const kernel = WendlandC2{dim}(h)

const x_0::RealType = 0.0 - h
const y_0::RealType = 0.0 - h
const x_1::RealType = 4.0 + h
const y_1::RealType = 3.0 + h

neighbour_cell_system = NeighbourCellSystem(x_0, y_0, x_1, y_1, h)

const input_particle_file_name = "demo/dam_break_2d/dam_break_2d.vtk"
@inline expandPolicy(n::IntType) = n
const max_neighbour_count::IntType = 120

particle_system = readVTP(input_particle_file_name; n_neighbours = max_neighbour_count, capacityPolicy = expandPolicy);

const sound_speed::RealType = 120.0
const sound_speed_square = sound_speed * sound_speed
const original_density::RealType = 1e3

particle_system.cu_sound_speed_ .= sound_speed

@inline @fastmath function eos(density::RealType)::RealType
    return sound_speed_square * (density - original_density)
end

const dt::RealType = 0.1 * h / sound_speed
const total_time::RealType = 4.0
const total_step = ceil(IntType, total_time / dt)
const output_intervel = 100
const density_filter_interval = 10

vtp_writer = VTPWriter()
vtp_writer.output_path_ = "demo/results/dam_break_2d"
vtp_writer.file_name_ = "dam_break_2d_"

assurePathExist(vtp_writer)

@inline function eachStep!(step::Int, t::Real)::Nothing
    continuityAndPressureExtrapolation!(particle_system, kernel, gx, gy; numthreads = n_threads)
    updateDensityAndPressure!(particle_system, dt, eos; numthreads = n_threads)
    momentum!(particle_system, kernel; numthreads = n_threads)
    accelerateAndMove!(particle_system, dt, gx, gy; numthreads = n_threads)
    findNeighbours!(particle_system, neighbour_cell_system; numthreads = n_threads)
    if round(Int, step % density_filter_interval) == 0
        densityFilterInteraction!(particle_system, kernel; numthreads = n_threads)
        densityFilterSelfaction!(particle_system, kernel; numthreads = n_threads)
    end
    if round(Int, step % output_intervel) == 0
        writeVTP!(vtp_writer, particle_system, step, t)
    end
    return nothing
end

@inline function main()::Nothing
    t::RealType = 0.0
    writeVTP!(vtp_writer, particle_system, 0, t)
    for step in ProgressBar(1:total_step)
        eachStep!(step, t)
        t += dt
    end
    waitAllWriteTasks(vtp_writer)
    return nothing
end
