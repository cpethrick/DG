#==============================================================================
# Functions which define the various mappings needed to select elements.
# Mainly copied from Hesthaven textbook.
==============================================================================#

function Normals1D(Nfp::Int64, Nfaces::Int64, K::Int64)
    nx = zeros(Nfp*Nfaces, K)
    nx[1,:] .= -1.0
    nx[2,:] .= 1.0
    return nx
end

function Connect1D(EtoV, Nfaces, K)
    
    totalfaces = Nfaces * K
    Nv = K+1
    
    #vn = [1,2]

    SpFtoV = SparseArrays.spzeros(totalfaces, Nv)
    sk = 1
    for k = 1:K
        for face = 1:Nfaces
            SpFtoV[sk, EtoV[k,face]] = 1
            sk += 1
        end
    end

    # Face-to-face connections and remove self-reference
    SpFtoF = SpFtoV * SpFtoV' - LinearAlgebra.I(totalfaces)

    faces = Tuple.(findall(==(1), SpFtoF))
    faces1 = first.(faces)
    faces2 = last.(faces)
    
    element1 = Int.(floor.((faces1.-1)/Nfaces) .+ 1) # which element number the face belongs to
    face1 = mod.(faces1.-1,Nfaces) .+ 1 # which element face the (global) face is (1, left, or 2, right)
    element2 = floor.((faces2.-1)/Nfaces) .+ 1
    face2 = mod.(faces2.-1,Nfaces) .+ 1
    
    ind = vec(LinearIndices(zeros(K,Nfaces))[CartesianIndex.(element1,face1)])

    EtoE::Matrix{Int64} = (1:K)*ones(1,Nfaces)
    EtoE[ind] .= element2

    EtoF::Matrix{Int64} = ones(K,1)*(1:Nfaces)'
    EtoF[ind] = face2

    return (EtoE, EtoF)
end

function BuildMaps1D(EtoE, EtoF, K, Np, Nfp, Nfaces, Fmask,x)
    
    nodeids = reshape(collect(1:K*Np),Np,K)
    vmapM_arr::Array{Int64,3} = zeros(Nfp, Nfaces, K)
    vmapP_arr::Array{Int64,3} = zeros(Nfp, Nfaces, K)

    for k1 = 1:K
        for f1=1:Nfaces
            vmapM_arr[:,f1, k1] = nodeids[Fmask[:,f1],k1]
        end
    end
    
    for k1 = 1:K
        for f1 = 1:Nfaces
            k2 = EtoE[k1, f1]
            f2 = EtoF[k1,f1]

            vidM = vmapM_arr[1,f1,k1]
            vidP = vmapM_arr[1,f2,k2]
            x1 = x[vidM]
            x2 = x[vidP]

            D = (x1-x2).^2
            if (D < 1E-10)
                vmapP_arr[1,f1,k1]=vidP
            end
        end
    end
    
    vmapP = vec(vmapP_arr)

    vmapM = vec(vmapM_arr)

    mapB = findall(==(0), vmapP-vmapM)
    vmapB = vmapM[mapB]

    mapI = 1
    mapO = K*Nfaces
    vmapI = 1
    vmapO = K * Np

    #Periodic boundaries
    vmapP[mapB[1]] = vmapM[mapB[2]]
    vmapP[mapB[2]] = vmapM[mapB[1]]

    return (vmapM, vmapP,vmapB,mapB,mapI,mapO,vmapI,vmapO)
end
