include("local_element.jl")

mutable struct DG
    # Category 1: inputs
    #P::Int #polynomial degree
    dim::Int #dimension - note that dim must be 1 for now!
    N_elem_per_dim::Int
    N_state::Int # Number of states in the PDE
    domain_x_limits::Vector{Float64} # x-limits of rectangular domain

    # Category 2:defined from Category 1.
    N_elem::Int
    EIDtoGroupID::Vector{Int} #NEW
    N_unique_GroupIDs::Int
    unique_GroupIDs::Vector{Int}
    max_N_soln::Int

    le::Dict{Int, LocalElement} # keys are groupID, accesses an arbitrary number of LocalElement objects

    #N_soln_per_dim::Int # Number of points per direction per cell; assumes x-direction
    #N_soln_y::Int # number of points in the y direction per cell
    #N_soln::Int # Total number of points per cell = N_soln^dim 
    #N_soln_dof::Int # Total number of DOFs per cell = N_soln*N_state
    N_soln_dof_global::Int # Global number of DOFs, i.e. length of the solution vector
    N_soln_global::Int # Global number of points
    #N_quad_per_dim::Int # Number of points per direction per cell; assumes y-direction
    #N_quad_y::Int # Number of points in y direction per cell
    #N_quad::Int # Total number of points per cell = N_quad^dim 
    #N_quad_dof::Int # Total number of DOFs per cell = N_quad*N_state #UNSURE IF THIS IS USED. 
    ##N_vol_per_dim::Int # WARNING: NOT FULLY IMPLEMENTED! Number of points per direction per cell. Used for unever # of basis and soln points.
    ##N_vol::Int # Total number of points per cell = N_quad^dim 
    N_faces::Int
    #N_face::Int # points on each face parallel to x-axis. I assume that the face nodes are 1D flux nodes.
    #N_face_y::Int # points on each face parallel to y-axis. I assume that the face nodes are 1D flux nodes.
    VX::Vector{Float64} # Array of points defining the extremes of each element along one dimension
    delta_x::Float64 # length of evenly-spaced Cartesian elements
    x::Vector{Float64} # physical x coords, index are global ID.
    y::Vector{Float64} # physical y coords, index are global ID.
    
    # Maps
    GIDtoLID::Vector{Int} #Index is global ID, values are local IDs
    EIDLIDtoGID_basis::AbstractMatrix{Int} # Index of first dimension is element ID, index of second
    #                         # dimension is element ID. values are global ID.
    EIDLIDtoGID_soln::AbstractMatrix{Int} # Index of first dimension is element ID, index of second
                             # dimension is element ID. values are global ID.
    ##LIDtoLFID::Vector{Int} # Index is local ID, value is local face ID
    ##                       # LFID = 1 is left face, LFID = 2 is right face. 0 is not a face.
    ##LFIDtoLID::AbstractMatrix{Int} # Index is local face ID, values are LID corresponding to that face
    LFIDtoNormal::AbstractMatrix{Int} # Normal of LFID,
                                      # first column is x, second column is y
    #EIDLFIDtoGIDofexterior::AbstractMatrix{Int} # Linker to exterior value at a face.
    #                                    # Index of first dim is element ID, index of second 
    #                                    # dimension is LFID of the edge.
    EIDLFIDtoEIDofexterior::AbstractMatrix{Int} # Linker to exterior ELEM at a face.
                                                # Index of first dim is element ID, index of second 
                                                # dimension is LFID of the edge.
                                                #
    #### Will update to mortar-element style interface handling
    LFIDtoLFIDofexterior::Vector{Int} # which LFID of the exterior cell matches to the index LFID.
    EIDtoTSID::Vector{Int} # maps element ID (index) to time-slab ID. 
                           # Elements with the same TSID are in the same "row"
                           # and cover the same t (y) vaues.
    TSIDtoEID::AbstractMatrix{Int} # TSID is the index, columns are EID.

    #StIDLIDtoLSID::AbstractMatrix{Int} # StID is the state ID, 1:Nstate
                                       # LID is the ID of the node
                                       # LSID ("local storage") indicates the index in the storage vector
    StIDGIDtoGSID::AbstractMatrix{Int}
    #LXIDLYIDtoLID::AbstractMatrix{Int} # local x, y (length N_soln_per_dim) to local ID (length N_soln)
    #LIDtoLXIDLYID::AbstractMatrix{Int} # local x, y (length N_soln_per_dim) to local ID (length N_soln)



    #Incomplete initializer - only assign Category 1 variables.
    DG(
       dim::Int, 
       N_elem_per_dim::Int,
       N_state::Int,
       domain_x_limits::Vector{Float64}) = new(
                                               dim::Int,
                                               N_elem_per_dim::Int,
                                               N_state::Int,
                                               domain_x_limits::Vector{Float64})

end



function build_coords_vectors(ref_vec_1D, dg::DG)

    if dg.dim==1
        return build_coords_vectors_1D(ref_vec_1D, dg::DG)
    elseif dg.dim==2
        return build_coords_vectors_2D(ref_vec_1D, ref_vec_1D, dg::DG)
    end
end

function build_local_coords_vectors_1D(ref_vec_1D, VX, delta_x)
    
    x_local = VX .+ 0.5* (ref_vec_1D .+1) * delta_x
    return x_local

end

function build_coords_vectors_1D(ref_vec_1D, dg::DG)

    x = zeros(dg.N_elem*(length(ref_vec_1D)^dg.dim))
    y = zeros(dg.N_elem*(length(ref_vec_1D)^dg.dim))
    Np=length(ref_vec_1D)^dg.dim
    Np_per_dim=length(ref_vec_1D)
    for ielem = 1:dg.N_elem
        x_local = build_local_coords_vectors_1D(ref_vec_1D, dg.VX[ielem], dg.delta_x)
        x[(ielem - 1) * Np+1:ielem*Np] .= x_local
    end
    return (x,y)
end

function build_local_coords_vectors_2D(ref_vec_x, ref_vec_y,VX, VY, delta_x)

    x_local_1D = VX .+ 0.5* (ref_vec_x .+1) * delta_x
    y_local_1D = VY .+ 0.5* (ref_vec_y .+1) * delta_x
    x_local = zeros(length(x_local_1D)*length(y_local_1D))
    y_local = zeros(size(x_local))
    Np_x_dim=length(ref_vec_x)
    Np_y_dim=length(ref_vec_y)
    for irow = 1:Np_y_dim
       #slightly gross indexing because we don't want to use LXIDLYIDtoLID for generality of ref_vec_1D.
       x_local[vec((1:Np_x_dim)' .+ (irow-1)*Np_x_dim)] .= x_local_1D
    end

    for icol = 1:Np_x_dim
       y_local[(1:Np_y_dim).*Np_x_dim.-(Np_x_dim-icol)] .= y_local_1D
    end
    return x_local,y_local
end

function build_coords_vectors_2D(ref_vec_x, ref_vec_y, dg::DG)
    
    x = zeros(dg.N_elem*length(ref_vec_x)*length(ref_vec_y))
    y = zeros(length(x))
    Np=length(ref_vec_x)*length(ref_vec_y)
    Np_x_dim=length(ref_vec_x)
    Np_y_dim=length(ref_vec_y)
    for ielem = 1:dg.N_elem
        x_index = mod(ielem-1,dg.N_elem_per_dim)+1
        VX = dg.VX[x_index]
        y_index = Int(ceil(ielem/dg.N_elem_per_dim))
        VY = dg.VX[y_index]

        x_local,y_local = build_local_coords_vectors_2D(ref_vec_x, ref_vec_y,VX, VY, dg.delta_x)

        x[(ielem - 1) * Np+1:ielem*Np] .= x_local
        y[(ielem - 1) * Np+1:ielem*Np] .= y_local
    end
    return (x,y)
end
# Outer constructor for DG object. Might be good to move to inner constructor at some point.
function init_DG(P::Int, dim::Int, N_elem_per_dim::Int, N_state::Int, domain_x_limits::Vector{Float64},
        solnnodes::String, basisnodes::String, quadnodes::String, quadnodes_overintegration::Int, fluxreconstructionC::Float64,
        usespacetime::Bool, y_dir_overintegration::Int)

    # Flag to set reference cell from 0 to 1, matching PHiLiP.
    reference_cell_01 = false
    if reference_cell_01 
        display("WARNING!! 2D will break because y-dim assumes (-1,1) reference cell!!")
    end

    #initialize incomplete DG struct
    dg = DG(dim, N_elem_per_dim, N_state, domain_x_limits)

    if dim == 1
        dg.N_faces = 2
        dg.N_elem = N_elem_per_dim
    elseif dim == 2
        dg.N_faces = 4
        dg.N_elem = N_elem_per_dim^dim
    end


    # Index is local ID, value is local face ID
    # LFID = 1 is left face, LFID = 2 is right face. 0 is not a face.
    if dim == 1
        dg.LFIDtoNormal = reshape([-1; 1], 2, 1) # normal of left face is 1, normal of right face is 1.
        #dg.LFIDtoLID = reshape([1,dg.N_soln_per_dim], 2,1)
    elseif dim == 2
        dg.LFIDtoNormal = [-1 0; 1 0; 0 -1; 0 1] #first col. is x, second col. is y
        #==
        dg.LFIDtoLID = [(0:dg.N_soln_per_dim-1)' *dg.N_soln_per_dim.+1 ;
                        (1:dg.N_soln_per_dim)' *dg.N_soln_per_dim;
                        (1:dg.N_soln_per_dim)';
                        (1:dg.N_soln_per_dim)' .+ (dg.N_soln-dg.N_soln_per_dim)
                       ]
                       ==#
    end

    # Connector mappings
    if dim == 1
        dg.EIDLFIDtoEIDofexterior = [circshift(1:dg.N_elem_per_dim,1)';circshift(1:dg.N_elem_per_dim,-1)']'
    elseif dim == 2
        dg.EIDLFIDtoEIDofexterior = zeros(Int, (dg.N_elem, dg.N_faces))
        for ielem = 1:dg.N_elem
            # face 1: left
            # regular joining (assume not on boundary)
            dg.EIDLFIDtoEIDofexterior[ielem,1] = ielem - 1
            if mod(ielem,dg.N_elem_per_dim) == 1
                #if on a periodic boundary
                dg.EIDLFIDtoEIDofexterior[ielem,1]+=dg.N_elem_per_dim
            end
            # hard-code the case where N_elem_per_dim=1
            if N_elem_per_dim == 1
                dg.EIDLFIDtoEIDofexterior[ielem,1] = 1
            end
            # face 2: right
            dg.EIDLFIDtoEIDofexterior[ielem,2] = ielem + 1
            if mod(ielem,dg.N_elem_per_dim) == 0
                dg.EIDLFIDtoEIDofexterior[ielem,2]-=dg.N_elem_per_dim
            end
            # face 3: bottom
            dg.EIDLFIDtoEIDofexterior[ielem,3] = ielem - dg.N_elem_per_dim
            if ielem/dg.N_elem_per_dim <= 1
                dg.EIDLFIDtoEIDofexterior[ielem,3] += dg.N_elem
                if usespacetime
                    # when using space-time, bottom is assigned dirichlet
                    # boundary
                    # This is set using an EIDofexterior = 0
                    dg.EIDLFIDtoEIDofexterior[ielem,3] = 0
                end
            end
            # face 4: top
            dg.EIDLFIDtoEIDofexterior[ielem,4] = ielem + dg.N_elem_per_dim
            if ielem/dg.N_elem_per_dim > dg.N_elem_per_dim - 1
                dg.EIDLFIDtoEIDofexterior[ielem,4] -= dg.N_elem
                if usespacetime
                    # when using space-time, top is outflow (transmissive)
                    # boundary
                    # This is set using an EIDofexterior = -1
                    dg.EIDLFIDtoEIDofexterior[ielem,4] = -1 
                end
            end
        end
    end
    dg.LFIDtoLFIDofexterior = [2, 1, 4, 3] #Hard-code for 2D. Also works fine for 1D. 
    #=== Not currently used anywhere (it messes with the generality of soln and basis nodes)
    if dim == 2
        dg.LXIDLYIDtoLID = zeros(Int, (N_soln_per_dim,N_soln_per_dim))
        dg.LIDtoLXIDLYID = zeros(Int, (N_soln_per_dim*N_soln_per_dim,2))
        counter=1
        for inodey = 1:N_soln_per_dim
            for inodex = 1:N_soln_per_dim
                dg.LXIDLYIDtoLID[inodex,inodey]=counter
                dg.LIDtoLXIDLYID[counter,:] .= [inodex,inodey]
                counter = counter+1
            end
        end
    end
    ===#
    if dim==2 #Only used for space-time, but the DG object has no knowledge of what is space-time
              # so we assemble for any 2D domain
        dg.EIDtoTSID = zeros(dg.N_elem)
        dg.TSIDtoEID = zeros(dg.N_elem_per_dim, dg.N_elem_per_dim)
        for ielem = 1:dg.N_elem
            iTS = convert(Int, floor((ielem-1)/dg.N_elem_per_dim+1))
            dg.EIDtoTSID[ielem] = iTS
            dg.TSIDtoEID[iTS, mod(ielem-1,dg.N_elem_per_dim)+1] = ielem
        end
    end


    dg.VX = range(domain_x_limits[1],domain_x_limits[2], dg.N_elem_per_dim+1) |> collect
    display("Elements per dim:")
    display(dg.N_elem_per_dim)
    dg.delta_x = dg.VX[2]-dg.VX[1]
    # constant jacobian on all elements as they are evenly spaced
    cell_length = 2.0
    if reference_cell_01
        cell_length = 1.0
    end
    jacobian = (dg.delta_x/cell_length)^dim #reference element is 2 units long



    ## Define group IDs
    # For now, all same group (ones)
    dg.EIDtoGroupID = ones(dg.N_elem)
    for ielem in 1:dg.N_elem_per_dim
        if dg.VX[ielem] < 0.5
            dg.EIDtoGroupID[ielem] = 1
        else
            dg.EIDtoGroupID[ielem] = 3
        end
    end

    dg.N_unique_GroupIDs = length(unique(dg.EIDtoGroupID))
    dg.unique_GroupIDs = unique(dg.EIDtoGroupID)

    #Initialize LocalElement structs for each unique groupID
    #For now, uses the (unique) inputs from the param file
    display("##### Operators for local element 1 #####")
    LE1 = init_LocalElement(P, dim, N_state,
        solnnodes, basisnodes, quadnodes, quadnodes_overintegration, fluxreconstructionC,
        usespacetime, y_dir_overintegration,
        jacobian, dg.N_faces, dg.delta_x, cell_length)
    display("##### Operators for local element 3 #####")
    LE2 = init_LocalElement(P, dim, N_state,
        solnnodes, basisnodes, quadnodes, quadnodes_overintegration, fluxreconstructionC,
        usespacetime, y_dir_overintegration,
        jacobian, dg.N_faces, dg.delta_x, cell_length)
    dg.le = Dict{Int, LocalElement}()
    dg.le[1] = LE1
    dg.le[3] = LE2
    display("##### Done local element allocations #####")

    # Count number of global DOFs for allocating arrasy
    if dg.N_unique_GroupIDs == 1
        dg.N_soln_global = dg.le[dg.unique_GroupIDs[1]].N_soln * dg.N_elem
        dg.max_N_soln = dg.le[dg.unique_GroupIDs[1]].N_soln
    else
        dg.max_N_soln = 0
        dg.N_soln_global = 0
        for igroup = dg.unique_GroupIDs
            occurences = count(==(igroup), dg.EIDtoGroupID)
            dg.N_soln_global += occurences * dg.le[igroup].N_soln
            dg.max_N_soln = maximum([dg.max_N_soln, dg.le[igroup].N_soln])
        end
    end
    dg.N_soln_dof_global = dg.N_soln_global * dg.N_state

    ##### Loop through all elements to build global mappings
    # Incomplete

    if dg.N_unique_GroupIDs == 1 
        # Index is global ID, values are local IDs
        N_soln = dg.le[dg.unique_GroupIDs[1]].N_soln
        dg.GIDtoLID = mod.(0:(N_soln*dg.N_elem.-1),N_soln).+1
        # Index of first dimension is element ID, index of second dimension is element ID
        # values are global ID
        #
        ##### Need to think about what to do with this
        dg.EIDLIDtoGID_basis = reshape(1:N_soln*dg.N_elem, (N_soln,dg.N_elem))' #note transpose
        dg.EIDLIDtoGID_soln = reshape(1:N_soln*dg.N_elem, (N_soln,dg.N_elem))' #note transpose
        
        dg.StIDGIDtoGSID = zeros(dg.N_state, N_soln*dg.N_elem)
        ctr = 1
        for ielem = 1:dg.N_elem
            for istate = 1:dg.N_state
                for ipoint = 1:N_soln
                    dg.StIDGIDtoGSID[istate,ipoint+N_soln*(ielem-1)]=ctr
                    ctr+=1
                end
            end
        end
    else
        dg.GIDtoLID = zeros(dg.N_soln_dof_global)
        dg.EIDLIDtoGID_basis = zeros(dg.N_elem, dg.max_N_soln) # size is based on largest number in all groups
        current_starting_ind = 1
        dg.StIDGIDtoGSID = zeros(dg.N_state, dg.N_soln_dof_global)
        ctr_StIDGIDtoGSID=1
        for ielem in 1:dg.N_elem
            N_soln_local = dg.le[dg.EIDtoGroupID[ielem]].N_soln
            dg.GIDtoLID[current_starting_ind:current_starting_ind+N_soln_local-1] = 1:N_soln_local

            dg.EIDLIDtoGID_basis[ielem, :] = [Array(current_starting_ind:current_starting_ind+N_soln_local-1); zeros(dg.max_N_soln-N_soln_local)]

            for istate = 1:dg.N_state
                for ipoint = 1:N_soln_local
                    dg.StIDGIDtoGSID[istate,ipoint+current_starting_ind-1]=ctr_StIDGIDtoGSID
                    ctr_StIDGIDtoGSID+=1
                end
            end
            current_starting_ind+=N_soln_local


        end
        dg.EIDLIDtoGID_soln = dg.EIDLIDtoGID_basis
    end

    if dg.N_unique_GroupIDs == 1
        # Uniform elements
        if dim==1
            (dg.x, dg.y) = build_coords_vectors_1D(dg.le[dg.unique_GroupIDs[1]].r_soln, dg) 
        elseif dim==2
            (dg.x, dg.y) = build_coords_vectors_2D(dg.le[dg.unique_GroupIDs[1]].r_soln, dg.le[dg.unique_GroupIDs[1]].r_soln_y, dg) 
        end
    else
        # Loop through all elements
        dg.x=[]
        dg.y=[]
        for  elemID in 1:dg.N_elem
        
            x_index = mod(elemID-1,dg.N_elem_per_dim)+1
            VX = dg.VX[x_index]
            y_index = Int(ceil(elemID/dg.N_elem_per_dim))
            VY = dg.VX[y_index]

            local_x, local_y = build_local_coords_vectors_2D(dg.le[dg.EIDtoGroupID[elemID]].r_soln, dg.le[dg.EIDtoGroupID[elemID]].r_soln_y, VX, VY, dg.delta_x)
            dg.x = [dg.x; local_x]
            dg.y = [dg.y; local_y]
        end
    end



    return dg
end


function test_initialization()

    P = 3
    dim = 2
    N_elem_per_dim = 2
    N_state=1
    domain_x_limits = [0.0,2.0]
    solnnodes = "GLL"
    basisnodes = "GLL"
    quadnodes = "GLL"
    quadnodes_overintegration = 0
    fluxreconstructionC = 0.0
    usespacetime = true
    y_dir_overintegration = 1


init_DG(P::Int, dim::Int, N_elem_per_dim::Int, N_state::Int, domain_x_limits::Vector{Float64},
        solnnodes::String, basisnodes::String, quadnodes::String, quadnodes_overintegration::Int, fluxreconstructionC::Float64,
        usespacetime::Bool, y_dir_overintegration::Int)


end

#test_initialization()
