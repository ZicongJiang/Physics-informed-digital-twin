import numpy as onp
import jax.numpy as np
from jax import random, grad, vmap, jit, jacfwd
from jax.example_libraries import optimizers
import itertools
from functools import partial
from torch.utils import data
from tqdm import trange
import jax.tree_util as jtu
import interpax

def tree_l2_norm(grad_tree):
    leaves, _ = jtu.tree_flatten(grad_tree)
    return np.sqrt(sum([np.sum(g**2) for g in leaves]))

def add_noise(x, snr_db):
    P_s = np.mean(np.abs(x)**2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = P_s / snr_linear
    noise_std = np.sqrt(noise_power/2)
    noise = noise_std * (onp.random.randn(*x.shape) + 1j * onp.random.randn(*x.shape))
    x = noise + x
    return x

class PDEPointGenerator(data.Dataset):
    def __init__(self, L, Tw, pres, rng_key=random.PRNGKey(1234)):
        self.L = float(L)
        self.Tw = float(Tw)
        self.pres = int(pres)
        self.key = rng_key

    def __len__(self):
        return 10**9

    def __getitem__(self, index):
        self.key, kx, kt = random.split(self.key, 3)
        x_res = random.uniform(kx, (self.pres, 1),
                               minval=0.01*self.L, maxval=0.99*self.L)
        t_res = random.uniform(kt, (self.pres, 1),
                               minval=0.01*self.Tw, maxval=0.99*self.Tw)
        y = np.hstack([x_res, t_res])  # (pres, 2)
        return y

# Data generator for labeled data - with paired inputs and outputs
class DataGenerator_labeled(data.Dataset):
    def __init__(self, s_in, s_ref,
                 batch_size=64, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.s_in = np.array(s_in)
        self.s_u = np.array(s_ref.real)
        self.s_v = np.array(s_ref.imag)
        self.key = rng_key
        self.batch_size = batch_size
        self.N = s_ref.shape[0]

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        inputs, outputs = self.__data_generation(subkey)
        return inputs, outputs

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        'Generates data containing batch_size samples'
        idx = random.choice(key, self.N, (self.batch_size,), replace=False)
        s_u = self.s_u[idx,:,:]
        s_v = self.s_v[idx,:,:]
        s_in = self.s_in[idx,:,:]
        # Construct batch
        inputs = s_in
        outputs = (s_u, s_v)
        return inputs, outputs
    
# Data generator for unlabeled data - with only inputs
class DataGenerator_unlabeled(data.Dataset):
    def __init__(self, y,
                 batch_size=64, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.y = y
        self.batch_size = batch_size
        self.key = rng_key
        self.N = self.y.shape[0]
        print(self.y.shape)

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        inputs = self.__data_generation(subkey)
        return inputs

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        'Generates data containing batch_size samples'
        idx = random.choice(key, self.N, (self.batch_size,), replace=True)
        y = self.y[idx,:]
        inputs = (y)
        return inputs

    
class PIDT:
    def __init__(self, alpha, beta2, gamma, params=None, lr=2e-3, L=None, dz=None, dt=None, beta2_factor=None, gamma_factor=None, num_steps=None, T=None, lamio=10.0, loss_type='l1', noise_db=40, pde_flag=1):
        # NLS basic coefficients
        self.A2 = alpha
        self.A3 = beta2
        self.A4 = gamma

        self.ema_alpha = 0.5
        self.lam_pde = 1.0 if pde_flag==1 else 0.0
        self.lam_io = lamio

        assert beta2_factor is not None; assert gamma_factor is not None ; assert L is not None
        assert dz is not None ; assert dt is not None; assert num_steps is not None; assert T is not None

        self.beta_factor = beta2_factor
        self.gamma_factor = gamma_factor
        self.pde_nsignals = 1  # how many signals to use for PDE residual each iteration

        # Initialize
        if params is None:
            params_dt = (random.normal(random.PRNGKey(1234), (num_steps,)),
                        random.normal(random.PRNGKey(1234), (num_steps,)))

            param_esti = np.array([0.0, 0.0])

            params = (param_esti, params_dt)
        else:
            params = params

        # Use optimizers to set optimizer initialization and update functions
        self.opt_init, \
        self.opt_update, \
        self.get_params = optimizers.adamax(optimizers.exponential_decay(lr,
                                                                          decay_steps=5000,
                                                                          decay_rate=0.9))
        
        self.opt_state = self.opt_init(params)

        # Logger
        self.itercount = itertools.count()
        self.loss_log = []
        self.loss_pde_res_log = []
        self.loss_io_log = []
        self.gamma_iter = []
        self.alpha_iter = []
        self.beta2_iter = []

        self.dz = L/num_steps
        self.L = L
        self.dt = dt
        self.num_steps = num_steps
        self.T = T
        self.pde_flag = pde_flag
        self.loss_type = loss_type
        self.noise_db = noise_db


        self.param_dt_history = {
            "iters": [],                # list[int]
            "beta2_profiles": [],       # list[np.ndarray] shape (num_steps,)
            "gamma_profiles": []        # list[np.ndarray] shape (num_steps,)
        }

    
    def ssfm(self, params_dt, x, dt, dz, num_steps, alpha):
        '''OUTPUT: x, S_xt : x - [1,n_up], S_xt - [num_steps, n_up] without signal at z=0'''
        beta2_array = params_dt[0]
        gamma_array = params_dt[1]
        
        f = np.fft.fftfreq(x.shape[1], dt)
        omega = (2 * np.pi * f)
        # key = random.PRNGKey(0)
        S_X = []
        S_X.append(x[0,:])
        for step in range(num_steps):
            beta2 = beta2_array[step] * self.beta_factor
            gamma = gamma_array[step] * self.gamma_factor
            # beta2 = self.A3
            # gamma = self.A4
            dispersion = np.exp((-alpha / 2.0 + 1j * (beta2) / 2 * (omega ** 2)) * dz/2)
            x = np.fft.ifft(np.fft.fft(x) * dispersion)
            x = x * np.exp(1j * (gamma) * dz * (np.abs(x) ** 2))
            x = np.fft.ifft(np.fft.fft(x) * dispersion)
            S_X.append(x[0,:])
        return x, np.array(S_X)


    def pde_residual_net(self, params, s_esti_xt, x, t):
        params_esti, _ = params
        [A3, A4] = params_esti
        num_steps, n_up = s_esti_xt.shape
        grid_x = np.linspace(0, self.L, num_steps)
        grid_y = np.linspace(0, self.T, n_up)

        x = np.mod(x, self.L)
        t = np.mod(t, self.T)

        def field(x, t):
            u = interpax.interp2d(x, t, grid_x, grid_y, np.real(s_esti_xt), method='cubic2', derivative=0, period=(None, n_up))
            v = interpax.interp2d(x, t, grid_x, grid_y, np.imag(s_esti_xt), method='cubic2', derivative=0, period=(None, n_up))
            return u, v
        
        du_dx = jacfwd(field, argnums=0)(x, t)[0]
        dv_dx = jacfwd(field, argnums=0)(x, t)[1]
        # second derivative wrt t
        d2u_dt2 = jacfwd(jacfwd(field, argnums=1), argnums=1)(x, t)[0]
        d2v_dt2 = jacfwd(jacfwd(field, argnums=1), argnums=1)(x, t)[1]
        # Compensate field
        # endregion

        u = field(x, t)[0]
        v = field(x, t)[1]

        res_1 = du_dx + self.A2 * u / 2.0 - (A3 * self.beta_factor) * d2v_dt2 / 2.0 + (A4 * self.gamma_factor) * (u**2 + v**2) * v
        res_2 = dv_dx + self.A2 * v / 2.0 + (A3 * self.beta_factor) * d2u_dt2 / 2.0 - (A4 * self.gamma_factor) * (u**2 + v**2) * u

        # leverage proxy: Frobenius norm of the 2x2 block
        return res_1, res_2
    
    def loss_pde_res(self, params, y_batch, io_batch):
        """
        Compute PDE residual loss using the SAME regenerated coordinates y_batch
        for MULTIPLE input signals (first self.pde_nsignals in the IO batch).
        """
        _, params_dt = params
        inputs, _ = io_batch           # inputs shape: [B_io, 1, Nt] (as in your DataGenerator_labeled)

        # Select how many signals to use for PDE residual this iteration
        K = int(self.pde_nsignals)
        inputs_sel = inputs[:K, ...]

        # 1) Propagate EACH selected signal with current params_dt (vectorized)
        def propagate_one(x_in):
            _, s_esti_xt = self.ssfm(params_dt, x_in, self.dt, self.dz, self.num_steps, self.A2)
            return s_esti_xt  # [Nz, Nt]
        S_xt = vmap(propagate_one)(inputs_sel)  # [Bsig, Nz, Nt]

        # 2) Evaluate PDE residuals at the SAME (x,t) points for EACH signal
        xq = y_batch[:, 0]  # [P]
        tq = y_batch[:, 1]  # [P]

        def residuals_one(s_esti_xt):
            # For one signal's space-time field, evaluate residual at all (xq,tq)
            r1, r2 = vmap(lambda x, t: self.pde_residual_net(params, s_esti_xt, x, t))(xq, tq)  # each [P]
            return r1, r2

        R1, R2 = vmap(residuals_one)(S_xt)  # each shape [Bsig, P]

        # 3) Aggregate across signals and points
        if self.loss_type == 'l2':
            loss_res = np.mean(R1**2) + np.mean(R2**2)   # mean over both signal and point dims
        elif self.loss_type == 'l1':
            loss_res = np.mean(np.abs(R1)) + np.mean(np.abs(R2))
        else:
            raise ValueError("Unknown loss type. Use 'l1' or 'l2'.")
        return loss_res
    
    def operator_twin(self, params, inputs):
        _, params_dt = params
        s_esti, _ = self.ssfm(params_dt, inputs, self.dt, self.dz, self.num_steps, self.A2)
        u_pred = s_esti[0].real
        v_pred = s_esti[0].imag
        return u_pred, v_pred
    

    # Define MSE loss compare estimated output and groundtruth output
    def loss_io(self, params, batch):
        inputs, outputs = batch
        u_label, v_label = outputs

        # add noise
        x = u_label + 1j * v_label

        x_noisy = add_noise(x, snr_db=self.noise_db)

        u_label = x_noisy.real
        v_label = x_noisy.imag
        # end

        u_pred, v_pred = vmap(self.operator_twin, (None, 0))(params, inputs)

        loss_u = np.mean((u_label.flatten() - u_pred.flatten()) ** 2)
        loss_v = np.mean((v_label.flatten() - v_pred.flatten()) ** 2)

        loss_io = loss_u + loss_v
        return loss_io


    # AFTER:
    def loss(self, params, res_batch, io_batch, lam_pde, lam_io):
        """Compute total loss with runtime weights (no constant capture in JIT)."""
        loss_io = self.loss_io(params, io_batch)
        loss_pde = self.loss_pde_res(params, res_batch, io_batch)
        total_loss = lam_pde * loss_pde + lam_io * loss_io
        return total_loss

    # AFTER:
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, res_batch, io_batch, lam_pde, lam_io):
        """One optimizer step that uses runtime weights."""
        params = self.get_params(opt_state)
        g = grad(self.loss)(params, res_batch, io_batch, lam_pde, lam_io)
        g_params_esti, g_params_dt = g
        masked_grads = (g_params_esti, g_params_dt)
        return self.opt_update(i, masked_grads, opt_state)

    # Optimize parameters in a loop
    def train(self, res_dataset, io_dataset, nIter=10000):
        res_data = iter(res_dataset)
        io_data = iter(io_dataset)
        pbar = trange(nIter)
        for self.it in pbar:
            res_batch = next(res_data)
            io_batch = next(io_data)
            # self.opt_state = self.step(next(self.itercount), self.opt_state, res_batch, io_batch)
            lam_pde_val = float(self.lam_pde)
            lam_io_val  = float(self.lam_io)

            self.opt_state = self.step(next(self.itercount),
                                    self.opt_state,
                                    res_batch, io_batch,
                                    lam_pde_val, lam_io_val)
            
            if self.it % 5000 == 0:
                params = self.get_params(self.opt_state)
                # [A2,A3,A4] = params_esti
                A3 = (params[0][0])
                A4 = (params[0][1])
                # loss_value = self.loss(params, res_batch, io_batch, lam_pde_val, lam_io_val)
                loss_io_value = self.loss_io(params, io_batch)
                loss_pde_value = self.loss_pde_res(params, res_batch, io_batch)
                loss_value = lam_pde_val * loss_pde_value + lam_io_val * loss_io_value

                self.loss_log.append(loss_value)
                self.loss_io_log.append(loss_io_value)
                self.loss_pde_res_log.append(loss_pde_value)
                self.gamma_iter.append(A4)
                self.beta2_iter.append(A3)

            if (self.it % 1000 == 0): #  and (self.it > 0)
                if self.pde_flag == 1:
                    grad_pde = grad(lambda p: self.loss_pde_res(p, res_batch, io_batch))(params)  # ∇L_pde
                    grad_io  = grad(lambda p: self.loss_io(p, io_batch))(params)        # ∇L_io

                    pde_norm = tree_l2_norm(grad_pde)
                    io_norm  = tree_l2_norm(grad_io)

                    sum_norm = pde_norm + io_norm + 1e-12
                    hat_lam_pde = sum_norm / (pde_norm + 1e-12)  # (2.12) / (2.13)
                    hat_lam_io  = sum_norm / (io_norm  + 1e-12)  # (2.13) / (2.14)

                    alpha = self.ema_alpha
                    self.lam_pde = alpha * self.lam_pde + (1 - alpha) * hat_lam_pde
                    self.lam_io  = alpha * self.lam_io  + (1 - alpha) * hat_lam_io
                else:
                    pass
                # # ----- NEW: snapshot params_dt every 1000 iters -----
                _, params_dt = params
                beta2_arr = np.array(params_dt[0])  # shape (num_steps,)
                gamma_arr = np.array(params_dt[1])  # shape (num_steps,)

                # store the *normalized* arrays as used in ssfm() before scaling factors
                self.param_dt_history["iters"].append(int(self.it))
                self.param_dt_history["beta2_profiles"].append(beta2_arr.copy())
                self.param_dt_history["gamma_profiles"].append(gamma_arr.copy())

            pbar.set_postfix({'Tot': loss_value, 'IO': loss_io_value, 'PDE': loss_pde_value, 'gam': (A4), 'gtgam': self.A4/self.gamma_factor, 'bet': (A3), 'gtbet': self.A3/self.beta_factor, 'lam_io': self.lam_io, 'lam_pde': self.lam_pde})
    
    # Evaluates predictions at test points
    @partial(jit, static_argnums=(0,))
    def predict_s(self, params, inputs):
        s_pred = vmap(self.operator_twin, (None, 0))(params, inputs)
        return s_pred

# Geneate unlabeled pde res evaluations
def generate_one_pde_res_training_data(key,pres, L,Tw):
    # guard_syms = 12  # >= rrc_delay (10), add margin
    # valid_syms = 128 - 2*guard_syms
    # assert valid_syms > 0, "Increase nsym or reduce guard"
    # t_min = guard_syms / 128
    # t_max = 1.0 - guard_syms / 128
    subkeys = random.split(key, 2)

    x_res = random.uniform(subkeys[0], (pres, 1), minval=0.01*L, maxval=0.9*L)
    t_res = random.uniform(subkeys[1], (pres, 1), minval=0.01*Tw, maxval=0.9*Tw)

    y =  np.hstack([x_res, t_res])

    return y

def ordered_direct_product(A, B):
    p = A.shape[0]
    q = B.shape[0]
    n = A.shape[1]
    m = B.shape[1]
    
    C = onp.zeros([p * q, n + m])
    C[:, :n] = onp.repeat(A, q, axis=0)
    C[:, n:] = onp.tile(B, (p, 1))
    return C

def QAM(M):
    Msqrt = int(onp.sqrt(M))
    if Msqrt ** 2 != M:
        raise ValueError('M must be a power of 4')
    
    x_pam = onp.expand_dims(onp.linspace(-Msqrt + 1, Msqrt - 1, Msqrt), axis=1)
    x_qam = ordered_direct_product(x_pam, x_pam)
    const = x_qam[:, 0] + 1j * x_qam[:, 1]
    return const / onp.sqrt(onp.mean(onp.abs(const)**2))  # Normalize based on average power

def rrcosine(rolloff, delay, OS):
    rrcos = onp.zeros(2 * delay * OS + 1)
    rrcos[delay * OS] = 1 + rolloff * (4 / onp.pi - 1)
    for i in range(1, delay * OS + 1):
        t = i / OS
        if t == 1 / (4 * rolloff):
            val = rolloff / onp.sqrt(2) * ((1 + 2 / onp.pi) * onp.sin(onp.pi / (4 * rolloff)) +
                                          (1 - 2 / onp.pi) * onp.cos(onp.pi / (4 * rolloff)))
        else:
            val = (onp.sin(onp.pi * t * (1 - rolloff)) +
                   4 * rolloff * t * onp.cos(onp.pi * t * (1 + rolloff))) / (onp.pi * t * (1 - (4 * rolloff * t) ** 2))
        rrcos[delay * OS + i] = val
        rrcos[delay * OS - i] = val
    return rrcos / onp.sqrt(onp.sum(rrcos ** 2))

def ssfm(gamma, x, dt, dz, num_steps, beta2, alpha):
    '''OUTPUT: x, S_xt : x - [1,n_up], S_xt - [num_steps, n_up] without signal at z=0'''
    f = np.fft.fftfreq(x.shape[1], dt)
    omega = (2 * np.pi * f)
    dispersion = onp.exp((-alpha / 2.0 + 1j * beta2 /2 * (omega ** 2)) * dz/2)
    # key = random.PRNGKey(0)
    S_X = []
    S_X.append(x[0,:])
    for _ in range(num_steps):
        x = np.fft.ifft(np.fft.fft(x) * dispersion)
        x = x * np.exp(1j * gamma * dz * (np.abs(x) ** 2))
        x = np.fft.ifft(np.fft.fft(x) * dispersion)
        S_X.append(x[0,:])
    return x, np.array(S_X)
