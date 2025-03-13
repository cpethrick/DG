
mutable struct CostTracking
   n_assemble_residual::Int64 
end

function init_CostTracking()
    return CostTracking(0)
end

function update_CostTracking(key::AbstractString, ct::CostTracking)
    if cmp(key, "assemble_residual") == 0
        ct.n_assemble_residual+=1
    end
end
