import argparse
import numpy as onp
import jax.numpy as np
from jax import random, vmap
from tqdm import trange
import matplotlib.pyplot as plt
import pickle
import os
import datetime
import pandas as pd
from utils.utils import *
import shutil
import sys
import yaml

from matplotlib.ticker import ScalarFormatter

# --- helpers for scientific notation and real iteration axis ---
def sci(x, prec=3) -> str:
    """Format a number in scientific notation for legend labels."""
    return f"{x:.{prec}e}"

def set_sci_x(ax):
    """Force scientific notation on the x-axis."""
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((0, 0))  # always use scientific notation
    ax.xaxis.set_major_formatter(fmt)

# ----------------------
# argparse for parameters
# ----------------------
parser = argparse.ArgumentParser(description="PINO Training Parameters")

parser.add_argument('--power_dbm', type=float, default=5, help='Launch power in dBm')
parser.add_argument('--sps', type=int, default=8, help='Samples per symbol')
parser.add_argument('--nsym', type=int, default=32, help='Number of symbols')
parser.add_argument('--rolloff', type=float, default=0.1, help='RRC rolloff factor')
parser.add_argument('--rrc_delay', type=int, default=10, help='RRC filter delay in symbols')
parser.add_argument('--M', type=int, default=16, help='QAM modulation order')
parser.add_argument('--N_data', type=int, default=220, help='Number of data samples')
parser.add_argument('--Rs', type=float, default=14e9, help='Symbol rate in Hz')
parser.add_argument('--L', type=float, default=80e3, help='Length of the fiber in meters')
parser.add_argument('--step_num', type=int, default=200, help='Number of steps')
parser.add_argument('--step_num_dt', type=int, default=10, help='Number of time steps for digital_twin')
parser.add_argument('--Nitr', type=int, default=50000, help='Training iterations')
parser.add_argument('--lamio', type=float, default=20.0, help='Scaling factor for lam_io')
parser.add_argument('--loss', type=str, default='l2', help='Loss function (L1 or L2)')
parser.add_argument('--noise_db', type=float, default=40.0, help='SNR in dB for adding noise to labeled data')
parser.add_argument('--pde_flag', type=int, default=1, help='PDE flag (1: use PDE residuals, 0: do not use)')


args = parser.parse_args()

# ----------------------
# [Directories]
# ----------------------
os.makedirs('results', exist_ok=True)
now = datetime.datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
result_dir = f"results/{timestamp}"
os.makedirs(result_dir, exist_ok=True)

# Save a copy of the current script into result directory
script_path = sys.argv[0]  # the script being executed
script_name = os.path.basename(script_path)
backup_path = os.path.join(result_dir, script_name)
try:
    shutil.copyfile(script_path, backup_path)
    print(f"Saved a copy of the script to {backup_path}")
except Exception as e:
    print(f"Failed to save script copy: {e}")

with open(f"{result_dir}/args.yaml", "w") as f:
    yaml.dump(vars(args), f)
# ----------------------
# [Physical constants & parameters]
# ----------------------
power_dbm = args.power_dbm
sps = args.sps
nsym = args.nsym
rolloff = args.rolloff
rrc_delay = args.rrc_delay
M = args.M
N_data = args.N_data
Nitr = args.Nitr
L = args.L
step_num = args.step_num
step_num_dt = args.step_num_dt

pde_flag = args.pde_flag # pde_flag=1 means training with PDE residuals

P0 = 10 ** (power_dbm / 10) * 10 ** (-3)
dB_conv = 4.342944819032518
alpha_db = 0.2
alpha = alpha_db * 1e-3 / dB_conv
c = 299_792_458.0
rf = 193.1e12
Rs = args.Rs
Ts = 1/Rs
Twin = nsym * Ts
Aeff = 80e-12
D = 16e-6
n2 = 2.6e-20
rw = c / rf
delta_beta1 = 0.0
beta2 = -D * rw**2 / (2 * onp.pi * c)
LD = Ts**2 / np.abs(beta2)
gamma = 2 * onp.pi * n2 * rf / c / Aeff
LNL = 1.0 / (gamma * P0)

nt = nsym * sps
dt = Twin / nt

T = onp.linspace(0, Twin, nt)
z = onp.linspace(0.0, L, step_num+1)
k1 = z.max() / LD / 1
k2 = T.max() / Ts / 1
l = z / LD / k1
deltaz = l[1] - l[0]
t = T / Ts / k2
deltat = t[1] - t[0]

A2 = 0.0
A3 = np.sign(beta2) * k1 / k2**2
A4 = k1 * LD / LNL

lamio = args.lamio

ord_a3 = np.floor(np.log10(np.abs(A3)))
ord_a4 = np.floor(np.log10(np.abs(A4)))

print(f"A3 order: {ord_a3}, A4 order: {ord_a4}")

ref_ord = 0

diff_a3 = ord_a3 - ref_ord
diff_a4 = ord_a4 - ref_ord

A3_factor = 10 ** diff_a3
A4_factor = 10 ** diff_a4

print(f"A3 factor: {A3_factor}, A4 factor: {A4_factor}")

# Display parameters like nvidia-smi
param_table = {
    "Parameter": ["power_dbm", "P0", "alpha", "beta2", "LD", "gamma", "LNL", "A3", "A4", "dz(m)", "norm_l", "norm_t", "k1", "k2", "step_num_dt", "lamio","loss","noise_db", "pde_flag"],
    "Value": [power_dbm, P0, alpha, beta2, LD, gamma, LNL, A3, A4, z[1] - z[0], l.max(), t.max(), k1, k2, step_num_dt, lamio, args.loss, args.noise_db, pde_flag],
}
print("\n========== Parameter Summary ==========")
print(pd.DataFrame(param_table).to_string(index=False))
print("=======================================\n")


# ----------------------
# [QAM signal generation]
# ----------------------
s_ref_batch = []
s_in_batch = []
n_up = sps * nsym

try:
    with open(f'PIDT_NORM_{N_data}_{sps}_{power_dbm}_{nsym}_{z.max()}.pkl', 'rb') as f:
        dataset = pickle.load(f)
    print("Dataset already exists, loading from file.")
    inS_array   = dataset['transmitted']
    s_ref_array = dataset['received']
except FileNotFoundError:
    print("Dataset not found, creating a new one.")
    pbar = trange(N_data)
    onp.random.seed(1024)
    const = QAM(M)

    rrc_filter = rrcosine(rolloff, rrc_delay, sps)
    filter_length = 2 * sps * rrc_delay + 1
    rrc_delay_samples = rrc_delay * sps
    rrc_filter_up = onp.concatenate([rrc_filter, onp.zeros(n_up - filter_length)])
    rrc_filter_up = onp.roll(rrc_filter_up, -rrc_delay_samples)
    rrc_filter_freq = onp.fft.fft(rrc_filter_up, n_up)

    for _ in pbar:
        symbols = const[onp.random.randint(const.shape[0], size=[1, nsym])]
        upsampled_symbols = onp.zeros((1, n_up), dtype=onp.complex64)
        upsampled_symbols[:, ::sps] = symbols * onp.sqrt(sps)
        inS = onp.fft.ifft(onp.fft.fft(upsampled_symbols) * rrc_filter_freq)

        s_ref, S_xt_gt = ssfm(A4, inS, deltat, deltaz, step_num, A3, A2)

        s_ref_batch.append(s_ref.astype(onp.complex64))
        s_in_batch.append(inS.astype(onp.complex64))

    inS_array = np.array(s_in_batch)
    s_ref_array = np.array(s_ref_batch)

    dataset = {
        'transmitted':    onp.stack(inS_array, axis=0),
        'received':       onp.stack(s_ref_array, axis=0),
    }

    with open(f'PIDT_NORM_{N_data}_{sps}_{power_dbm}_{nsym}_{z.max()}.pkl', 'wb') as f:
        pickle.dump(dataset, f)

# Plot signal example
plt.figure(figsize=(8, 3))
plt.subplot(1, 2, 2)
plt.plot(onp.real(s_ref_array[0, 0, :]), label='Received/Reference Signal')
plt.legend()
plt.subplot(1, 2, 1)
plt.plot(onp.real(inS_array[0, 0, :]), label='Transmitted/Initial condition Signal')
plt.legend()
plt.title('Collected data for training')
plt.savefig(f'{result_dir}/dataset_sample.png', dpi=600)
plt.close()

# ----------------------
# [Prepare training data]
# ----------------------
N_train = int(N_data * 0.9)
px = 2
pt = n_up
pres = 5000
inter = 1

key = random.PRNGKey(0)
keys = random.split(key, N_train)

y_res_train = vmap(generate_one_pde_res_training_data, in_axes=(0, None, None, None))(keys,pres, l.max(), t.max())
print(y_res_train.shape)
y_res_train = y_res_train.reshape(N_train * pres, -1)
print(f'y_res_train: {y_res_train.shape}') # 180 * 2000 = 360000

# res_dataset = DataGenerator_unlabeled(y_res_train, 50)
res_dataset = PDEPointGenerator(L=l.max(), Tw=t.max(), pres=pres,
                                rng_key=random.PRNGKey(5678))
io_dataset = DataGenerator_labeled(inS_array[:N_train], s_ref_array[:N_train], 100) # 10x1xn_up 20x512x2=20480

model = PIDT(A2, A3, A4, params=None, lr=1e-3, L=l.max(), dz=deltaz, dt=deltat, beta2_factor=A3_factor, gamma_factor=A4_factor, num_steps=step_num_dt, T=t.max(), lamio = lamio, loss_type=args.loss, noise_db=args.noise_db, pde_flag=pde_flag)

# Train the model
model.train(res_dataset, io_dataset, nIter=Nitr)
params = model.get_params(model.opt_state)


print(f"A3: {A3} | Estimated A3: {params[0][0]}; A4: {A4} | Estimated A4: {params[0][1]}")

# ----------------------
# [Visualization]
# ----------------------
b = np.sign(beta2) / k2**2
n_a3 = len(model.beta2_iter)
n_a4 = len(model.gamma_iter)

def build_iter_axis(n_points: int, total_iters: int, start_iter: int = 0):
    if n_points <= 1:
        return np.arange(n_points)
    step = max(1, int(round((total_iters - start_iter) / (n_points - 1))))
    return start_iter + step * np.arange(n_points)

x_a3 = build_iter_axis(n_a3, Nitr, start_iter=0)  # set start_iter to 0 or to your first logged iter if different
x_a4 = build_iter_axis(n_a4, Nitr, start_iter=0)

# Series (same as before)
A3_series = np.array(model.beta2_iter) * A3_factor        # estimator trace for A3
A4_series = np.array(model.gamma_iter) * A4_factor        # estimator trace for A4

# Physical param estimates (same as before)
estimated_beta2 = ((np.array(model.beta2_iter) * A3_factor) * (Ts**2) / b / z.max()) * np.sign(beta2)
estimated_gamma = np.array(model.gamma_iter) * A4_factor / (k1 * LD * P0)

# Final values for legends
A3_est_final = A3_series[-1] if len(A3_series) else np.nan
A4_est_final = A4_series[-1] if len(A4_series) else np.nan
b2_est_final = estimated_beta2[-1] if len(estimated_beta2) else np.nan
g_est_final  = estimated_gamma[-1] if len(estimated_gamma) else np.nan

plt.figure(figsize=(8, 3))

# (1) beta2 estimate
ax3 = plt.subplot(1, 2, 1)
ax3.plot(x_a3, estimated_beta2, lw=3, label=f"Estimated beta2 (final={sci(b2_est_final)})")
ax3.hlines(beta2, x_a3[0], x_a3[-1], colors='r', linestyles='--', linewidth=2,
           label=f"beta2 true ({sci(beta2)})")
ax3.set_xlabel('Iteration')
ax3.set_ylabel('beta2')
ax3.legend()
set_sci_x(ax3)

# (2) gamma estimate
ax4 = plt.subplot(1, 2, 2)
ax4.plot(x_a4, estimated_gamma, lw=3, label=f"Estimated gamma (final={sci(g_est_final)})")
ax4.hlines(gamma, x_a4[0], x_a4[-1], colors='r', linestyles='--', linewidth=2,
           label=f"gamma true ({sci(gamma)})")
ax4.set_xlabel('Iteration')
ax4.set_ylabel('gamma')
ax4.legend()
set_sci_x(ax4)

plt.tight_layout()
plt.savefig(f'{result_dir}/parameter_estimation.png', dpi=600)
plt.close()



# Plot multiple snapshots over z, compare to true (normalized)
snap_iters = model.param_dt_history["iters"]
snap_betas = model.param_dt_history["beta2_profiles"]  # list of (num_steps,)
snap_gamms = model.param_dt_history["gamma_profiles"]  # list of (num_steps,)

snap_beta2 = ((np.array(snap_betas) * A3_factor) * (Ts**2) / b / z.max()) * np.sign(beta2)
snap_gamma = np.array(snap_gamms) * A4_factor / (k1 * LD * P0)

plt.figure(figsize=(8, 3))
plt.subplot(1, 2, 1)
plt.plot(snap_beta2[-1], linestyle='-',  marker='*',label='Estimated beta2 profiles')
plt.hlines(beta2, 0, model.num_steps - 1, colors='r', linestyles='--', label='True beta2 profile')
plt.xlabel('Step index')
plt.ylabel('Normalized beta2')
plt.title('Estimated vs True beta2 Profiles Over Steps')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(snap_gamma[-1], linestyle='-',  marker='*',label='Estimated gamma profiles')
plt.hlines(gamma, 0, model.num_steps - 1, colors='r', linestyles='--', label='True gamma profile')
plt.title('Estimated vs True gamma Profiles Over Steps')
plt.legend()
plt.xlabel('Step index')
plt.ylabel('Normalized gamma')
plt.tight_layout()
plt.savefig(f'{result_dir}/param_profiles_over_steps.png', dpi=600)
plt.close()

# Test on new data, for forward prediction
inS_test = inS_array[-1]

predicted_s, _ = model.ssfm(params[1], inS_test, model.dt, model.dz, model.num_steps, A2)
real_s = ssfm(A4, inS_test, deltat, deltaz, step_num, A3, A2)[0]

error=np.linalg.norm(predicted_s - real_s) / np.linalg.norm(real_s)
print(f"Relative L2 error on new data: {error}")

plt.figure(figsize=(7, 3))
plt.subplot(1, 2, 1)
plt.plot(np.real(predicted_s[0, :]), label='Predicted', lw=2)
plt.plot(np.real(real_s[0, :]), label='Ground Truth', linestyle='dashed', lw=2)
plt.title('Real Part')
plt.xlabel('Time Step')
plt.ylabel('State Value')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(np.imag(predicted_s[0, :]), label='Predicted', lw=2)
plt.plot(np.imag(real_s[0, :]), label='Ground Truth', linestyle='dashed', lw=2)
plt.title('Imaginary Part')
plt.xlabel('Time Step')
plt.ylabel('State Value')
plt.legend()
plt.tight_layout()
plt.savefig(f'{result_dir}/fw_pred_test.png', dpi=600)
plt.close()

print("Training complete.")
