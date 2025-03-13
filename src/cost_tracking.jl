import Printf

mutable struct CostTracking
    is_active::Bool
    n_assemble_residual::Int64 
end

function init_CostTracking()
    return CostTracking(false, 0)
end

function update_CostTracking(key::AbstractString, ct::CostTracking)
    #Update the cost tracker.
    #Structured such that different parts of the code can update different tracking metrics.

    # Flag indicating whether the tracker has been used since being initialized
    ct.is_active = true

    if cmp(key, "assemble_residual") == 0
        ct.n_assemble_residual+=1
    end
end

function summary(ct::CostTracking)
    # This function prints a summary of the tracking in a human-readable format.

    # Check that the tracking has actually been used
    if ct.is_active
        
        Printf.@printf("Number of residual evaluations:    %d \n", ct.n_assemble_residual)
    end

end
