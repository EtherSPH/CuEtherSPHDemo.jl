#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2024/08/29 17:04:40
  @ license: MIT
  @ description:
 =#

const kMaxCellContainedNumber::IntType = 100

mutable struct NeighbourCellSystem
    start_x_::RealType
    start_y_::RealType
    last_x_::RealType
    last_y_::RealType
    delta_x_::RealType
    delta_y_::RealType
    delta_x_inv_::RealType
    delta_y_inv_::RealType
    radius_::RealType
    radius_square_::RealType
    n_x_::IntType
    n_y_::IntType
    total_cell_number_::IntType
    cu_cell_contained_index_count_::CuArray{IntType, 1, CUDA.DeviceMemory}
    cu_cell_contained_index_list_::CuArray{IntType, 2, CUDA.DeviceMemory}
end

@inline function NeighbourCellSystem(
    start_x::RealType,
    start_y::RealType,
    last_x::RealType,
    last_y::RealType,
    radius::RealType,
    max_cell_contained_number::IntType = kMaxCellContainedNumber,
)::NeighbourCellSystem
    n_x = floor(IntType, (last_x - start_x) / radius)
    delta_x = (last_x - start_x) / n_x
    delta_x_inv = RealType(1.0 / delta_x)
    n_y = floor(IntType, (last_y - start_y) / radius)
    delta_y = (last_y - start_y) / n_y
    delta_y_inv = RealType(1.0 / delta_y)
    total_cell_number = n_x * n_y
    cu_cell_contained_index_count = CUDA.fill(IntType(0), total_cell_number)
    cu_cell_contained_index_list = CUDA.fill(IntType(0), total_cell_number, max_cell_contained_number)
    return NeighbourCellSystem(
        start_x,
        start_y,
        last_x,
        last_y,
        delta_x,
        delta_y,
        delta_x_inv,
        delta_y_inv,
        radius,
        radius * radius,
        n_x,
        n_y,
        total_cell_number,
        cu_cell_contained_index_count,
        cu_cell_contained_index_list,
    )
end

@inline function Base.show(io::IO, ncs::NeighbourCellSystem)
    println(io, "NeighbourCellSystem:")
    println(io, "  start from: $(ncs.start_x_), $(ncs.start_y_)")
    println(io, "  end at: $(ncs.last_x_), $(ncs.last_y_)")
    println(io, "  delta in x and y direction: $(ncs.delta_x_), $(ncs.delta_y_)")
    println(io, "  inverse delta in x and y direction: $(ncs.delta_x_inv_), $(ncs.delta_y_inv_)")
    println(io, "  radius: $(ncs.radius_)")
    println(io, "  number of cells in x and y direction: $(ncs.n_x_), $(ncs.n_y_)")
    return println(io, "  total cell number: $(ncs.total_cell_number_)")
end
