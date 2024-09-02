#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2024/08/29 20:21:29
  @ license: MIT
  @ description:
 =#

@inline function resetNeighbour!(ncs::NeighbourCellSystem)::Nothing
    CUDA.fill!(ncs.cu_cell_contained_index_count_, IntType(0))
    return nothing
end

@inline function resetNeighbour!(ps::ParticleSystem)::Nothing
    CUDA.fill!(ps.cu_neighbour_count_, IntType(0))
    return nothing
end

@inline function twoDimensionIndexToLinearIndex(i_x::IntType, i_y::IntType, n_x::IntType)::IntType
    return i_x + (i_y - IntType(1)) * n_x
end

@inline function linearIndexToTwoDimensionIndex(i_cell::IntType, n_x::IntType)::Tuple{IntType, IntType}
    i_x = mod(i_cell, n_x)
    i_y = cld(i_cell, n_x)
    return i_x, i_y
end

@inline function cuAddParticleIndexIntoCells!(
    n_particles::IntType,
    cu_is_alive,
    cu_cell_index,
    cu_position,
    n_x::IntType,
    n_y::IntType,
    start_x::RealType,
    start_y::RealType,
    last_x::RealType,
    last_y::RealType,
    delta_x_inv::RealType,
    delta_y_inv::RealType,
    cu_cell_contained_index_count,
    cu_cell_contained_index_list,
)::Nothing
    index = (blockIdx().x - IntType(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i_particle in index:stride:n_particles
        if i_particle > n_particles
            break
        end
        @inbounds x = cu_position[i_particle, 1]
        @inbounds y = cu_position[i_particle, 2]
        dx = x - start_x
        dy = y - start_y
        if dx < 0 || dy < 0 || x > last_x || y > last_y
            @inbounds cu_is_alive[i_particle] = false
        end
        @inbounds if cu_is_alive[i_particle] == true
            i_x = max(IntType(1), ceil(IntType, dx * delta_x_inv))
            i_y = max(IntType(1), ceil(IntType, dy * delta_y_inv))
            i_x = min(i_x, n_x)
            i_y = min(i_y, n_y)
            i_cell = twoDimensionIndexToLinearIndex(i_x, i_y, n_x)
            i_loc = CUDA.atomic_add!(pointer(cu_cell_contained_index_count, i_cell), IntType(1))
            i_loc += IntType(1)
            @inbounds cu_cell_contained_index_list[i_cell, i_loc] = i_particle
            @inbounds cu_cell_index[i_particle] = i_cell
        end
    end
    return nothing
end

@inline function addParticleIndexIntoCells!(
    particle_system::ParticleSystem,
    neighbour_cell_system::NeighbourCellSystem;
    numthreads = 256,
)::Nothing
    numblocks = ceil(Int, particle_system.n_particles_ / numthreads)
    CUDA.@sync begin
        @cuda threads = numthreads blocks = numblocks cuAddParticleIndexIntoCells!(
            particle_system.n_particles_,
            particle_system.cu_is_alive_,
            particle_system.cu_cell_index_,
            particle_system.cu_position_,
            neighbour_cell_system.n_x_,
            neighbour_cell_system.n_y_,
            neighbour_cell_system.start_x_,
            neighbour_cell_system.start_y_,
            neighbour_cell_system.last_x_,
            neighbour_cell_system.last_y_,
            neighbour_cell_system.delta_x_inv_,
            neighbour_cell_system.delta_y_inv_,
            neighbour_cell_system.cu_cell_contained_index_count_,
            neighbour_cell_system.cu_cell_contained_index_list_,
        )
    end
    return nothing
end

@inline function cuCreateParticleNeighbourIndexList!(
    n_particles::IntType,
    cu_is_alive,
    cu_cell_index,
    cu_position,
    cu_neighbour_count,
    cu_neighbour_index_list,
    cu_neighbour_position_list,
    cu_neighbour_distance_list,
    n_x::IntType,
    n_y::IntType,
    radius_square::RealType,
    cu_cell_contained_index_count,
    cu_cell_contained_index_list,
)::Nothing
    index = (blockIdx().x - IntType(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i_particle in index:stride:n_particles
        if i_particle > n_particles
            break
        end
        @inbounds if cu_is_alive[i_particle] == true
            @inbounds cell_index = cu_cell_index[i_particle]
            cell_i_x, cell_i_y = linearIndexToTwoDimensionIndex(cell_index, n_x)
            for d_x in -1:1, d_y in -1:1
                i_x = cell_i_x + IntType(d_x)
                i_y = cell_i_y + IntType(d_y)
                if i_x < 1 || i_y < 1 || i_x > n_x || i_y > n_y
                    nothing
                else
                    cell_linear_index = twoDimensionIndexToLinearIndex(i_x, i_y, n_x)
                    @inbounds cell_contained_index_count = cu_cell_contained_index_count[cell_linear_index]
                    if cell_contained_index_count > 0
                        for i_neighbour in 1:cell_contained_index_count
                            @inbounds neighbour_particle_index =
                                cu_cell_contained_index_list[cell_linear_index, i_neighbour]
                            if neighbour_particle_index != i_particle
                                @inbounds dx = cu_position[i_particle, 1] - cu_position[neighbour_particle_index, 1]
                                @inbounds dy = cu_position[i_particle, 2] - cu_position[neighbour_particle_index, 2]
                                dr_square = dx * dx + dy * dy
                                if dr_square <= radius_square
                                    @inbounds cu_neighbour_count[i_particle] += IntType(1)
                                    @inbounds neighbour_count = cu_neighbour_count[i_particle]
                                    @inbounds cu_neighbour_index_list[i_particle, neighbour_count] =
                                        neighbour_particle_index
                                    @inbounds cu_neighbour_position_list[i_particle, neighbour_count, 1] = dx
                                    @inbounds cu_neighbour_position_list[i_particle, neighbour_count, 2] = dy
                                    @inbounds cu_neighbour_distance_list[i_particle, neighbour_count] = sqrt(dr_square)
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    return nothing
end

@inline function createParticleNeighbourIndexList!(
    particle_system::ParticleSystem,
    neighbour_cell_system::NeighbourCellSystem;
    numthreads = 256,
)::Nothing
    numblocks = ceil(Int, particle_system.n_particles_ / numthreads)
    CUDA.@sync begin
        @cuda threads = numthreads blocks = numblocks cuCreateParticleNeighbourIndexList!(
            particle_system.n_particles_,
            particle_system.cu_is_alive_,
            particle_system.cu_cell_index_,
            particle_system.cu_position_,
            particle_system.cu_neighbour_count_,
            particle_system.cu_neighbour_index_list_,
            particle_system.cu_neighbour_position_list_,
            particle_system.cu_neighbour_distance_list_,
            neighbour_cell_system.n_x_,
            neighbour_cell_system.n_y_,
            neighbour_cell_system.radius_square_,
            neighbour_cell_system.cu_cell_contained_index_count_,
            neighbour_cell_system.cu_cell_contained_index_list_,
        )
    end
    return nothing
end

@inline function findNeighbours!(
    particle_system::ParticleSystem,
    neighbour_cell_system::NeighbourCellSystem;
    numthreads = 256,
)::Nothing
    resetNeighbour!(neighbour_cell_system)
    resetNeighbour!(particle_system)
    addParticleIndexIntoCells!(particle_system, neighbour_cell_system; numthreads = numthreads)
    createParticleNeighbourIndexList!(particle_system, neighbour_cell_system; numthreads = numthreads)
    return nothing
end
