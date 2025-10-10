using HighOrderMRISim
using PyPlot; PyPlot.pygui(true)
using MAT

T = Float32

# ------------------------------------------------------------
# 1. Load the base sequence and initialize parameters
# ------------------------------------------------------------
path = "demo3D";
name = "Spiral_2p0x2p0x2p0_100x100x35_1x4_int1_9p5ms_tr500_te5p0";

seq = read_seq("$(path)/$(name).seq")
seq.GR[1,:] = -seq.GR[1,:]  # invert Gx

hoseq = HO_Sequence(seq);

params = matread("$(path)/$(name).mat")
dt_adc         = params["skope"]["dt_adc"];
nSample_adc    = Int64(params["skope"]["nSample_adc"]);
dur_adc        = nSample_adc*dt_adc;
gradRasterTime = 10e-6;
nGrad_adc      = Int64(round(dur_adc/gradRasterTime));

# ------------------------------------------------------------
# 2. Load measured trajectories (by Skope)
# ------------------------------------------------------------
skope   = matread("$(path)/$(name)_skope.mat")
g_skope = T.(skope["g_skope"] * 1e-3); # T
t_g     = T.(vec(skope["t_g"])); # [s]

# ------------------------------------------------------------
# 3. Replace nominal gradients with measured gradients
# ------------------------------------------------------------
nBlock = size(seq.GR, 2)
iadc = 1;
for iblock = 1:nBlock
    if seq.GR[1, iblock].delay > 0 && is_ADC_on(seq[iblock])
        g = InterpTrajTime(g_skope[iadc, :, :], gradRasterTime, -0.9*gradRasterTime, t_g)

        nGrad_Gz_pre  = 34;
        nGrad_Spoiler = 300;
        nGrad_adc     = Int64(round(dur_adc/gradRasterTime));
        nGrad_delay   = Int64(round(seq.GR[1, iblock].delay/gradRasterTime));
    
        for i = 1:16
            if i == 4
                hoseq.GR_dfc[i, iblock-1].T    = (nGrad_Gz_pre-1)*gradRasterTime;
                hoseq.GR_dfc[i, iblock-1].A    = g[1:nGrad_Gz_pre, i];
                hoseq.GR_dfc[i, iblock-1].rise = gradRasterTime/2;
                hoseq.GR_dfc[i, iblock-1].fall = gradRasterTime/2;
            end

            hoseq.GR_dfc[i, iblock].T     = (nGrad_adc-1)*gradRasterTime;
            hoseq.GR_dfc[i, iblock].delay = nGrad_delay*gradRasterTime;
            hoseq.GR_dfc[i, iblock].A     = g[nGrad_Gz_pre+nGrad_delay+1:end, i];
            hoseq.GR_dfc[i, iblock].rise  = gradRasterTime/2;
            hoseq.GR_dfc[i, iblock].fall  = gradRasterTime/2;
            if i == 2 || i == 3
                hoseq.GR_dfc[i, iblock].T = (nGrad_adc+nGrad_Spoiler-1)*gradRasterTime;
                hoseq.GR_dfc[i, iblock].A = [g[nGrad_Gz_pre+nGrad_delay+1:end, i]; seq.GR[i-1, iblock].A[end-nGrad_Spoiler+1:end]]
            end
        end
        iadc += 1
    else
        hoseq.GR_dfc[2:4, iblock]  = deepcopy(hoseq.SEQ.GR[1:3, iblock]); 
    end
end

# ------------------------------------------------------------
# 4. Invert ADC phase for RF phase compensation
# ------------------------------------------------------------
nBlock = size(seq.GR, 2)
iadc = 1;
for iblock = 1:nBlock
    if is_ADC_on(seq[iblock])
        seq[iblock].ADC[1].ϕ = -seq[iblock].ADC[1].ϕ
    end
end

# ------------------------------------------------------------
# 5. Plot the final sequence
# ------------------------------------------------------------
plt_seq(hoseq)
plot_seq(hoseq.SEQ)
