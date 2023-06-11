using LinearAlgebra
using Plots
using QuadGK

function get_TRG_initial_tensor(beta)

    # 4-tensor where each index has dimension 2
    # This defines the basic building block of the two dimensional tensor network which corresponds to the Ising model partition function
    # Given the definition is A_{t,r,b,l} = exp(-(1/T) * (tt*rr + rr*bb + bb*ll + ll*tt)) where pp = 2*p-3 for p in [t,r,b,l] this is just because the indices t,r,b,l 
    # go from 1 to 2 and we want the spin variables to take value -1, 1
    A_initial = zeros(Float64, 2, 2, 2, 2)

    function Hamiltonian_pattern(t, r, b, l)

        """

        Since the Hamiltonian is sigma_i * sigma_j where i and j are nearest neighbours there exists for every i index 4 neighbours.
        The A tensor to be used by the TRG lies in the middle of 4 spins as the spins live on the tensor's edges.

        """

        return ((2*t-3)*(2*r-3) + (2*r-3)*(2*b-3) + (2*b-3)*(2*l-3) + (2*l-3)*(2*t-3))
    end

    # Return an iterator over the product of several iterators. Each generated element is a tuple whose ith element comes from the ith argument iterator. The first iterator changes the fastest.
    values = [1, 2]
    combinations = collect(Iterators.product(values, values, values, values))
    for combination in combinations
        t, r, b, l = combination
        A_initial[t, r, b, l] = exp(-beta*Hamiltonian_pattern(t, r, b, l))
    end

    return A_initial

end

function contraction(A, c_A::Tuple, B, c_B::Tuple)::Array{ComplexF64}

    """
    The contraction function takes 2 tensors A, B and 2 tuples c_A, c_B and returns
    another tensor after contracting A and B

    A: first tensor
    c_A: indices of A to contract (Tuple of Int64)
    B: second tensor
    c_B: indices of B to contract (Tuple of Int64)

    Note 1: c_A and c_B should be the same length and the first index from c_A should
    have the same dimension as the first index of c_B, the second index from c_A
    should have the same dimension as the second index of c_B and so on.

    Note 2: It is assumed that the first index in c_A is to be contracted with the
    first index in c_B and so on.

    Note 3: If we were instead to use vectors for c_A and c_B, the memory allocation 
    sky rockets and the run time is 10 times slower. Vectors require more memory than
    tuples and run time since tuples are immutable and only store a certain type each time etc.

    Example: If A is a 4-tensor, B is a 3-tensor and I want to contract the first
    index of A with the second index of B and the fourth index of A with the first
    index of B, then the input to the contraction function should be:

    contraction(A, (1, 4), B, (2, 1))

    This will result in a 3-tensor since we have 3 open indices left after the
    contraction, namely second and third indices of A and third index of B

    Code Example:
    # @time begin
    # A = cat([1 2; 3 4], [5 6; 7 8], dims = 3)
    # B = cat([9 11; 11 12], [13 14; 15 16], dims = 3)
    # c_A = (1, 2)
    # c_B = (2, 1)
    # display(contraction(A, c_A, B, c_B))
    # end
    """

    # Get the dimensions of each index in tuple form for A and B

    A_indices_dimensions = size(A) # returns tuple(dimension of index 1 of A, ...)
    B_indices_dimensions = size(B)

    # Get the uncontracted indices of A and B named u_A and u_B. The setdiff
    # returns the elements which are in the first argument and which are not
    # in the second argument.

    u_A = setdiff(1:ndims(A), c_A)
    u_B = setdiff(1:ndims(B), c_B)

    # Check that c_A and c_B agree in length and in each of their entry they
    # have the same index dimension using the macro @assert. Below we also find
    # the dimensions of each index of the uncontracted indices as well as for the
    # contracted ones.

    dimensions_c_A = A_indices_dimensions[collect(c_A)]
    dimensions_u_A = A_indices_dimensions[collect(u_A)]
    dimensions_c_B = B_indices_dimensions[collect(c_B)]
    dimensions_u_B = B_indices_dimensions[collect(u_B)]

    @assert(dimensions_c_A == dimensions_c_B, "Note 1 in the function
    contraction docstring is not satisfied: indices of tensors to be contracted
    should have the same dimensions. Input received: indices of first tensor A
    to be contracted have dimensions $(dimensions_c_A) and indices of second
    tensor B to be contracted have dimensions $(dimensions_c_B).")

    # Permute the indices of A and B so that A has all the contracted indices
    # to the right and B has all the contracted indices to the left.

    # NOTE: The order in which we give the uncontracted indices (in this case
    # they are in increasing order) affects the result of the final tensor. The
    # final tensor will have indices starting from A's indices in increasing
    # ordera and then B's indices in increasing order. In addition c_A and c_B
    # are expected to be given in such a way so that the first index of c_A is
    # to be contracted with the first index of c_B and so on. This assumption is
    # crucial for below, since we need the aforementioned specific order for
    # c_A, c_B in order for the vectorisation below to work.

    A = permutedims(A, (u_A..., c_A...)) # Splat (...) unpacks a tuple in the argument of a function
    B = permutedims(B, (c_B..., u_B...))

    # Reshape tensors A and B so that for A the u_A are merged into 1 index and
    # the c_A are merged into a second index, making A essentially a matrix.
    # The same goes with B, so that A*B will be a vectorised implementation of
    # a contraction. Remember that c_A will form the columns of A and c_B will
    # form the rows of B and since in A*B we are looping over the columns of A
    # with the rows of B it is seen from this fact why the vectorisation works.

    # To see the index dimension of the merged u_A for example you have to think
    # how many different combinations I can have of the individual indices in
    # u_A. For example if u_A = (2, 4) this means the uncontracted indices of A
    # are its second and fourth index. Let us name them alpha and beta
    # respectively and assume that alpha ranges from 1 to 2 and beta from
    # 1 to 3. The possible combinations are 1,1 and 1,2 and 1,3 and 2,1 and 2,2
    # and 2,3 making 6 in total. In general the total dimension of u_A will be
    # the product of the dimensions of its indivual indices (in the above
    # example the individual indices are alpha and beta with dimensions 2 and
    # 3 respectively so the total dimension of the merged index for u_A will
    # be 2x3=6).

    A = reshape(A, (prod(dimensions_u_A), prod(dimensions_c_A)))
    B = reshape(B, (prod(dimensions_c_B), prod(dimensions_u_B)))

    # Perform the vectorised contraction of the indices

    C = A*B

    # Reshape the resulting tensor back to the individual indices in u_A and u_B
    # which we previously merged. This is the unmerging step.

    C = reshape(C, (dimensions_u_A..., dimensions_u_B...))

    return C

end

function coarse_grain(A, nsv, norms)

    """

    The index convention for tensor A is:

         1
         |
    4 -- A --- 2 
         |
         3

    where 1 = t, 2 = l, 3 = b, 4 = r

    Inputs:

    A = the 4-tensor to be coarse grained

    nsv = max_dim = the maximum bond dimension allowed on the A

    norms = the list tracking the norms we accumulate during the TRG, we normalize A for stable numerics by the highest value in A i.e. divide all its elements by that value
    
    Outputs:

    A_final = the coarse grained 4-tensor

    """

    # Normalize by the highest value in the A_final tensor
    norm = argmax(abs2, A)
    append!(norms, norm)
    A = A/norm

    # Prepare two tensors into matrices such that svd can be performed on them
    A_1 = permutedims(A, (1, 4, 3, 2)) # t, l, b, r (this comment and subsequent ones keep track of the indices that the final tensor ends up with i.e. in this case A_1)
    t, l, b, r = size(A_1) # t is the dimension of the index t and similarly for l, b, r
    A_1 = reshape(A_1, (t*l, b*r)) # (t, l), (b, r)
    A_2 = reshape(A, (t*r, b*l)) # (t, r), (b, l)

    # Perform the svd
    svd_1 = svd(A_1)
    svd_2 = svd(A_2)

    # These tensors from svd_1 will produce tensors F_1, F_3
    diag_S_1 = svd_1.S # s1
    s_1 = min(nsv, length(diag_S_1))
    U_1 = svd_1.U[:, 1:s_1] # (t, l), s1 
    sqrt_S_1 = Diagonal(sqrt.(diag_S_1[1:s_1])) # s1, s1
    Vt_1 = svd_1.Vt[1:s_1, :] # s1, (b, r)

    # These tensors from svd_1 will produce tensors F_2, F_4
    diag_S_2 = svd_2.S # s2
    s_2 = min(nsv, length(diag_S_2))
    U_2 = svd_2.U[:, 1:s_2] # (t, r), s2
    sqrt_S_2 = Diagonal(sqrt.(diag_S_2[1:s_2])) # s2, s2
    Vt_2 = svd_2.Vt[1:s_2, :] # s2, (b, l)

    # Multiply the square root of the singular value matrix on U and V respectively to get F_1, F_2, F_3, F_4
    F_1 = U_1*sqrt_S_1 # (t, l), s1
    F_3 = sqrt_S_1*Vt_1 # s1, (b, r)
    F_2 = U_2*sqrt_S_2 # (t, r), s2
    F_4 = sqrt_S_2*Vt_2 # s2, (b, l)

    # Reshape F_1, F_2, F_3, F_4, to open up the original indices merged together to perform svd
    F_1 = reshape(F_1, (t, l, s_1)) # t, l, s1
    F_3 = reshape(F_3, (s_1, b, r)) # s1', b, r
    F_2 = reshape(F_2, (t, r, s_2)) # t, r, s2
    F_4 = reshape(F_4, (s_2, b, l)) # s2', b, l

    # Contract the F_1, F_2, F_3, F_4 to obtain the new coarse grained A tensor
    tmp_1 = contraction(F_1, (2,), F_4, (3,)) # t, s1, s2', b 
    tmp_2 = contraction(F_2, (2,), F_3, (3,)) # t, s2, s1', b
    tmp_3 = contraction(tmp_1, (1, 4), tmp_2, (1, 4)) # s1, s2', s2, s1' : this is the biggest cost of the algorithm's subroutine (which is performed log(N_tensors) times) and has cost order(bond dimension ^ 6) - see page 26 of https://libstore.ugent.be/fulltxt/RUG01/002/836/279/RUG01-002836279_2020_0001_AC.pdf

    # Permute the indices into the order of the convention
    A_final = permutedims(tmp_3, (2, 1, 3, 4)) # s2', s1, s2, s1' (here we are effectively tilting the lattice back into its original orientation but the tilting rotation direction does not matter i.e. if its anti-clockwise or clockwise)

    return A_final

end

function TRG(A, N, max_dim; verbose = false)

    """

    The TRG wants to calculate the partition function of the 2D Ising model in this particular script example by contracting a 2 dimension network of tensors.
    Given the fact that all tensors in the network are the same, we can just play with 1 tensor and any action we do on it will be the same for all others so 
    we don't have to actually do it for all others and only keep in memory a single A tensor. For the description of the algorithm on which this script example
    is based on see https://tensornetwork.org/trg/ which describes plain TRG. 

    Inputs:

    A = initial 4-tensor to be input in TRG, it can be non-normalized because the coarse grain function normalizes before anything else

    N = the side length of the square spin lattice

    max_dim = the maximum bond dimension allowed in the TRG

    verbose = boolean to check whether we want the dimensions of the tensor we are doing TRG on to be printed out

    Ouputs:

    Z, norms = partition function given by the trace of the last A/A_norm tensor we end up with and
               a list of floats used to normalize the A tensor in the coarse graining iterations to keep stable numerics, 
               the length of this vector is number_of_TRG_iterations-1 where the -1 comes from the last norm being in the Z accounted for

    """

    N_spins = N*N # Total number of spins
    N_tensors = N_spins // 2 # Each tensor has 2 unique legs and since spins live on the legs of the tensors for every tensor we have 2 spins
    norms = [] # Initialize the norm list and put in it the norm of the initial tensor
    for i in 1:Int(log2(N_tensors)) # This amount of iterations will effectively allow us to end up with 1 single A tensor which we finally trace its legs over
        if verbose 
            println("Iteration number: ", i, ", Size of A tensor: ", size(A))
        end
        A = coarse_grain(A, max_dim, norms) 
    end

    t, l = size(A)[1:2]
    tmp_1 = contraction(A, (1, 3), I(t), (1, 2)) # l, r (this comment keeps track of the result indices i.e. the indices of tmp_1)
    Z = real(contraction(tmp_1, (1, 2), I(l), (1, 2))[1]) # contract the legs of the final A/A_norm tensor where the A_norm is attached in the coarse grain function

    return Z, norms

end

function get_free_energy_density(Z, beta, norms, N_total)

    """

    Inputs:

    Z = The value of the partition function which is given by tr(A_last/A_last_norm), float
    
    beta = 1/T (inverse temperature, float)

    norms = a list of floats used to normalize the A tensor in the coarse graining iterations to keep stable numerics, 
            the length of this vector is number_of_TRG_iterations-1 where the -1 comes from the last norm being in the Z accounted for

    N_total = total number of spins which is equal to N*N where N is the side length of the square lattice

    Output:

    The free energy density f based on the numerical results of TRG where f is defined overall as f = -(T/Vol)*ln(Z) = -(1/beta*Vol)*ln(Z) where Vol = volume = number of spins = N*N = N_total = 2*N_tensors

    """

    N_tensors = N_total // 2 # Each tensor has 2 legs which can be considered unique which implies for every tensor we have two spins, remember spins live on the legs of the tensors
    f = 0
    for i in 1:Int(log2(N_tensors)-1)
        f += -((1)/(beta*(2^i)))*log(real(norms[i])) # As the TRG is being carried out in coarse graining iterations, we normalize the A tensor for stable numerics and these norm factors end up in the formula for f
    end
    f += -((1)/(2*N_tensors*beta))*log(Z) # This takes care of the last tensor we trace out into Z

    return f

end

function get_free_energy_density_exact(beta)

    """

    Sources for the exact Onsager's result for the 2D Ising model: 

    https://gandhiviswanathan.wordpress.com/2015/01/09/onsagers-solution-of-the-2-d-ising-model-the-combinatorial-method/ or https://itensor.org/docs.cgi?page=book/trg

    Inputs:

    beta = 1/T (inverse temperature, float)

    Output:

    Onsager's analytical solution to the 2D Ising model at a given beta

    """
    
    inner1(theta1, theta2) = log(cosh(2 * beta)^2 - sinh(2 * beta) * cos(theta1) - sinh(2 * beta) * cos(theta2))

    inner2(theta2) = quadgk(theta1 -> inner1(theta1, theta2), 0, 2 * π)[1]

    I = quadgk(inner2, 0, 2*π)[1]
    
    return -(log(2) + I / (8 * pi^2)) / beta

end

# Define the length of the side of the square spin lattice
pow = 16
N = 2^(pow)

# Define the maximum bond dimension allowed for the A tensor to be coarse grained
max_dim_list = [15, 16, 17]

# Define the temperature = 1/beta
temp_list = LinRange(2, 3, 5)

# Some plotting stuff
available_markers = [:+, :x, :diamond, :hexagon, :square, :circle, :star4]
plot_1 = plot()
plot_2 = plot()

# Do TRG for different bond dimensions and temperatures and plot results
for (max_dim_idx, max_dim) in enumerate(max_dim_list)
    
    # These local lists keep trace of the free energy density f and its exact solution
    local f_list = []
    local f_exact_list = []

    println("-----------------------------------------")
    for temp in temp_list
        
        println("Starting temperature: ", temp)
        
        beta = 1/temp
        A_initial = get_TRG_initial_tensor(beta) # Remeber this A_initial can be unnormalized since coarse grain will normalize it
        Z, norms = TRG(A_initial, N, max_dim, verbose = false) # Perform the actual algorithm to get the partition function Z and the norms used for stable numerics
        f = get_free_energy_density(Z, beta, norms, N*N)
        f_exact = get_free_energy_density_exact(beta)
        append!(f_list, f)
        append!(f_exact_list, f_exact)
        
        println("Temperature: ", temp, ", Partition function: ", Z, ", Free energy density: ", f, ", Exact free energy density: ", f_exact)
        println("-----------------------------------------")
    end
    

    scatter!(plot_1, temp_list, f_list, marker = rand(available_markers), label = "Max bond dim = $(max_dim)")
   
    delta_f_list = []
    for i in 1:length(temp_list)
        append!(delta_f_list, abs(f_list[i] - f_exact_list[i])/abs(f_exact_list[i])) # Fractional difference between f and f_exact for all temperatures at fixed maximum bond dimension simulation
    end
    display(delta_f_list)
    scatter!(plot_2, temp_list, delta_f_list, marker = rand(available_markers), label = "Max bond dim = $(max_dim)")

end

# Plot free energy density f vs temperature for different maximum bond dimensions
title!(plot_1, "Lattice size $(N)x$(N)")
ylabel!(plot_1, "Free Energy Density")
xlabel!(plot_1, "Temperature")
plot!(plot_1, legend=:outerbottom, legendcolumns = length(max_dim_list))
savefig(plot_1, "f_vs_t.png")

# Plot the fractional error of f with respect to f_exact for different maximum bond dimensions
ylabel!(plot_2, "Fractional f Error")
xlabel!(plot_2, "Temperature T")
title!(plot_2, "Lattice size $(N)x$(N)")
plot!(plot_2, legend=:outerbottom, legendcolumns = length(max_dim_list))
savefig(plot_2, "fractional_error.png")
