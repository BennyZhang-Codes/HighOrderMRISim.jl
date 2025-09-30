# Use the dev_jyzhang branch at https://github.com/BennyZhang-Codes/HighOrderMRI.jl/tree/dev_jinyuan

# HighOrderMRI.jl and HighOrderMRISim.jl should be run in separate environments
using HighOrderMRI # add this package to your env
using MRIGeometry  # add this package to your env (located at the directory of HighOrderMRI.jl)

using MRIReco   # v0.9.0 required by HighOrderMRI.jl
                # v0.8.0 required by KomaMRI@0.8.0 and HighOrderMRISim.jl

using RegularizedLeastSquares
using MAT
using PyPlot
pygui(true)

using CUDA
CUDA.device!(7)

T = Float32;

path = "demo3D";

name = "Spiral_2p0x2p0x2p0_100x100x35_1x4_int1_9p5ms_tr500_te5p0_1111_wB0";
data = matread("$(path)/$(name).mat")


kdata    = data["signal"];
kspha    = -data["kspha"];
csm      = data["csm"];
b0       = -data["b0"] ./ 2π;
datatime = data["t_adc"];
xyz      = data["xyz_grid"]

dt_adc   = data["dt_adc"];

FOV        = data["target_fov"]; # [m]
MatrixSize = data["matrixsize"]; # [px]

nX, nY, nZ = MatrixSize;
Resolution = FOV ./ MatrixSize; # [m/px]
Δx, Δy, Δz = Resolution


kdata    = reshape(kdata, :, 32);
datatime = reshape(datatime, :);

# norm = sqrt.(sum(abs.(csm) .^ 2, dims=4)) .+ 1e-8;
# csm = csm./ norm;
# csm[isnan.(csm)] .= 0 + 0im;

# xyz = generate_rps_grid(FOV, MatrixSize);
xyz = T.(xyz);
y = vec(xyz[:, :, :, 1]);
x = vec(xyz[:, :, :, 2]);
z = vec(xyz[:, :, :, 3]);

gridding = Grid(nX=nX, nY=nY, nZ=nZ, Δx=Δx, Δy=Δy, Δz=Δz, x=x, y=y, z=z)

weight  = SampleDensity(kspha[:, 2:4]', (nX, nY, nZ));



use_gpu = true;
verbose = true;

solver = CGNR; reg = L2Regularization(1.e-9); iter = 10;
recParams = Dict{Symbol,Any}()
recParams[:reconSize]      = (nX, nY, nZ)
recParams[:reg] = reg  
recParams[:iterations]     = iter
recParams[:solver]         = solver


recon_terms = "1111";
HOOp = HighOrderOp3D_Kernel(gridding, T.(kspha'), T.(datatime); recon_terms=recon_terms, csm=Complex{T}.(permutedims(csm, [2,1,3,4])), fieldmap=T.(permutedims(b0.*0, [2,1,3]) .- 0), use_gpu=use_gpu, verbose=verbose);  # reverse(csm, dims=(2,3))

@time x = recon_HOOp(HOOp, Complex{T}.(kdata), Complex{T}.(weight), recParams);

img = reverse(x, dims=(1,2,3))
fig = plt_image(mosaic(abs.(img); dim=3, nRow=2, nCol=5); vminp=0, vmaxp=99.9, width=21, height=15)
fig.savefig("$(path)/$(name)_recon1111_wB0.png", dpi=300, transparent=true, bbox_inches="tight", pad_inches=0.0)

################################################
# NUFFT 
################################################
nSample = data["nSample_adc"]
C = maximum(2*abs.(kspha[:,2:4][:]));  #Normalize k-space to -.5 to .5 for NUFFT

tr = Trajectory("3dspiral", T.(kspha[:,2:4]' ./ C),  T.(datatime), T.(5e-3), T.(nSample*dt_adc), 1, nSample, 1, false, false);
dat         = Array{Array{Complex{T},2},3}(undef,1,1,1);
dat[1,1,1]  = kdata;
acqData     = AcquisitionData(tr, dat, encodingSize=(nX, nY, nZ));

solver = CGNR; reg = L2Regularization(0.e-9); iter = 20;
params = Dict{Symbol, Any}()
params[:reco]        = "multiCoil"
params[:reconSize]   = (nX, nY, nZ)
params[:reg]         = reg  
params[:iterations]  = iter
params[:solver]      = solver
params[:senseMaps]   = Complex{T}.(permutedims(reverse(csm, dims=(1,2,3)), [1,2,3,4]));

@time Ireco = reconstruction(acqData, params);
img = Ireco[:,:,:,1,1,1];
img = mapslices(rotr90, img; dims=(1,2))
fig = plt_image(mosaic(abs.(img); dim=3, nRow=2, nCol=5); vminp=0, vmaxp=99.9, width=21, height=15)

fig.savefig("$(path)/$(name)_reconNUFFT.png", dpi=300, transparent=true, bbox_inches="tight", pad_inches=0.0)
