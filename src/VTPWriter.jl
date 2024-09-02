#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2024/09/02 00:12:20
  @ license: MIT
  @ description:
 =#

const kFileExtension = ".vtp"
const kWallTimeFormat = "yyyy_mm_dd_HH_MM_SS.SS"
const kDensityString = "Density"
const kMassString = "Mass"
const kTypeString = "Type"
const kVelocityString = "Velocity"
const kPressureString = "Pressure"

@inline function assurePathExist(path::String)::Nothing
    if !isdir(path)
        @info "Create directory: $path"
        mkpath(path)
    else
        @info "Remove all files in directory: $path"
        for file in readdir(path)
            rm(joinpath(path, file))
        end
    end
    return nothing
end

@inline function getWallTime()::String
    return Dates.format(now(), kWallTimeFormat)
end

@kwdef mutable struct VTPWriter
    output_count_::Int64 = 0
    step_digit_::Int64 = 4 # 0001, 0002, ..., 9999
    file_name_::String = "result"
    output_path_::String = "example/results"
    write_task_list_::Vector{Base.Task} = Task[]
end

@inline function assurePathExist(vtp_writer::VTPWriter)::Nothing
    assurePathExist(vtp_writer.output_path_)
    return nothing
end

@inline function getOutputFileName(vtp_writer::VTPWriter)::String
    return joinpath(
        vtp_writer.output_path_,
        string(vtp_writer.file_name_, string(vtp_writer.output_count_, pad = vtp_writer.step_digit_), kFileExtension),
    )
end

@inline function transferGPUToCPU(particle_system::ParticleSystem)
    position = Array(particle_system.cu_position_[particle_system.cu_is_alive_, :])
    density = Array(particle_system.cu_density_[particle_system.cu_is_alive_])
    mass = Array(particle_system.cu_mass_[particle_system.cu_is_alive_])
    type = Array(particle_system.cu_type_[particle_system.cu_is_alive_])
    pressure = Array(particle_system.cu_pressure_[particle_system.cu_is_alive_])
    velocity = Array(particle_system.cu_velocity_[particle_system.cu_is_alive_, :])
    return position, density, mass, type, pressure, velocity
end

@inline function writeVTP!(
    file_name::String,
    vtp_writer::VTPWriter,
    step::Int,
    simulation_time::Real,
    position,
    density,
    mass,
    type,
    pressure,
    velocity,
)::Nothing
    position = position' |> Array
    velocity = velocity' |> Array
    n_particles = length(type)
    cells = [MeshCell(PolyData.Verts(), [i]) for i in 1:n_particles]
    vtp_file = vtk_grid(file_name, position, cells)
    vtp_file["TMSTEP"] = step
    vtp_file["TimeValue"] = simulation_time
    vtp_file["WallTime"] = getWallTime()
    vtp_file[kTypeString] = type
    vtp_file[kDensityString] = density
    vtp_file[kMassString] = mass
    vtp_file[kPressureString] = pressure
    vtp_file[kVelocityString] = velocity
    vtk_save(vtp_file)
    return nothing
end

@inline function writeVTP!(
    vtp_writer::VTPWriter,
    particle_system::ParticleSystem,
    step::Int,
    simulation_time::Real,
)::Nothing
    position, density, mass, type, pressure, velocity = transferGPUToCPU(particle_system)
    vtp_writer.output_count_ += 1
    file_name = getOutputFileName(vtp_writer)
    task = @async writeVTP!(
        file_name,
        vtp_writer,
        step,
        simulation_time,
        position,
        density,
        mass,
        type,
        pressure,
        velocity,
    )
    push!(vtp_writer.write_task_list_, task)
    return nothing
end

@inline function waitAllWriteTasks(vtp_writer::VTPWriter)::Nothing
    for i in ProgressBar(1:length(vtp_writer.write_task_list_))
        wait(vtp_writer.write_task_list_[i])
    end
    return nothing
end
