using LinearAlgebra

# Based on https://tensornetwork.org/trg/ - Plain TRG 

# TODO: Normalize the coarse-grained tensor after each iteration
# TODO: Check contraction order in coarse_grain function is optimal
# TODO: Generalize to any Hamiltonian (classical or quantum) and boundary conditions

# Define a temperature T
temp = 1
beta = 1/temp

# 4-tensor where each index has dimension 2
# This defines the basic building block of the two dimensional tensor network which corresponds to the Ising model partition function
# Given the definition is A_{t,r,b,l} = exp(-(1/T) * (tt*rr + rr*bb + bb*ll + ll*tt)) where pp = 2*p-3 for p in [t,r,b,l] this is just because the indices t,r,b,l 
# go from 1 to 2 and we want the spin variables to take value -1, 1
A_initial = zeros(Float64, 2, 2, 2, 2)

function f(t, r, b, l)
    return ((2*t-3)*(2*r-3) + (2*r-3)*(2*b-3) + (2*b-3)*(2*l-3) + (2*l-3)*(2*t-3))
end

# Return an iterator over the product of several iterators. Each generated element is a tuple whose ith element comes from the ith argument iterator. The first iterator changes the fastest.
values = [1, 2]
combinations = collect(Iterators.product(values, values, values, values))
for combination in combinations
    t, r, b, l = combination
    A_initial[t, r, b, l] = exp(-beta*f(t, r, b, l))
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

function coarse_grain(T, nsv)

    T_1 = permutedims(T, (1, 4, 3, 2)) # t, l, b, r
    t, l, b, r = size(T_1)
    T_1 = reshape(T_1, (t*l, b*r)) # (t, l), (b, r)
    T_2 = reshape(T, (t*r, b*l)) # (t, r), (b, l)

    svd_1 = svd(T_1)
    svd_2 = svd(T_2)

    diag_S_1 = svd_1.S # s1
    s_1 = min(nsv, length(diag_S_1))
    U_1 = svd_1.U[:, 1:s_1] # (t, l), s1 
    sqrt_S_1 = Diagonal(sqrt.(diag_S_1[1:s_1])) # s1, s1
    Vt_1 = svd_1.Vt[1:s_1, :] # s1, (b, r)

    diag_S_2 = svd_2.S # s2
    s_2 = min(nsv, length(diag_S_2))
    U_2 = svd_2.U[:, 1:s_2] # (t, r), s2
    sqrt_S_2 = Diagonal(sqrt.(diag_S_2[1:s_2])) # s2, s2
    Vt_2 = svd_2.Vt[1:s_2, :] # s2, (b, l)

    F_1 = U_1*sqrt_S_1 # (t, l), s1
    F_3 = sqrt_S_1*Vt_1 # s1, (b, r)

    F_2 = U_2*sqrt_S_2 # (t, r), s2
    F_4 = sqrt_S_2*Vt_2 # s2, (b, l)

    F_1 = reshape(F_1, (t, l, s_1)) # t, l, s1
    F_3 = reshape(F_3, (s_1, b, r)) # s1', b, r

    F_2 = reshape(F_2, (t, r, s_2)) # t, r, s2
    F_4 = reshape(F_4, (s_2, b, l)) # s2', b, l

    tmp_1 = contraction(F_1, (2,), F_4, (3,)) # t, s1, s2', b 
    tmp_2 = contraction(F_2, (2,), F_3, (3,)) # t, s2, s1', b
    tmp_3 = contraction(tmp_1, (1, 4), tmp_2, (1, 4)) # s1, s2', s2, s1'
    
    T_final = permutedims(tmp_3, (2, 1, 3, 4)) # s2', s1, s2, s1'

    return T_final

end

function TRG(T, N, max_dim)

    N_spins = N*N
    N_tensors = N_spins // 2
    for i in 1:log2(N_tensors)
        T = coarse_grain(T, max_dim) 
    end

    t, l = size(T)[1:2]
    tmp_1 = contraction(T, (1, 3), I(t), (1, 2)) # l, r 
    Z = real(contraction(tmp_1, (1, 2), I(l), (1, 2))[1])

    return Z

end

N = 4
max_dim = 20
result = TRG(A_initial, N, max_dim)
println(result)
