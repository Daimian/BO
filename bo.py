import os, shutil
import traceback
import math
import numpy as np
from dataclasses import dataclass
from dataclasses import field
import torch
import torch.nn.functional as F
from botorch.acquisition import qExpectedImprovement
from botorch.acquisition import qUpperConfidenceBound
from botorch.acquisition import UpperConfidenceBound
from botorch.fit import fit_gpytorch_model
from botorch.generation import MaxPosteriorSampling
from botorch.models import FixedNoiseGP, SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize
from botorch.utils.transforms import unnormalize
from torch.quasirandom import SobolEngine
import gpytorch
from gpytorch.constraints import Interval
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import HorseshoePrior
from typing import List, Optional, Tuple
from botorch.test_functions.base import BaseTestProblem
from torch import Tensor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def get_initial_points(dim, n_pts):
    #sobol = SobolEngine(dimension=dim, scramble=True, seed=0)
    sobol = SobolEngine(dimension=dim, scramble=True)
    X_init = sobol.draw(n=n_pts).to(dtype=dtype, device=device)
    return X_init
#test = get_initial_points(3, 5)
#print(test)
#fun_V = VASP(dim=3, negate=False).to(dtype=dtype, device=device)
#fun_V.bounds[0, :] = torch.tensor([-0.002,-0.002,-0.002])
#fun_V.bounds[1, :] = torch.tensor([0.002, 0.002, 0.002])
#dum = unnormalize(test, fun_V.bounds)
#print(dum)

def read_amp(root_path):

    with open(root_path+'dict_candi') as f:
        candi_lines = f.readlines()

    _X = []
    _Y = []
    for i in candi_lines:
        dum = []
        info = i.split()
        file_name = info[0]
        x = np.array([float(num) for num in info[1].split('_') if num])
        x_term = convert_amp(x)
        for sublist in x_term:
            for item in sublist:
                dum.append(item)
        y = float(info[2])
        _X.append(dum)
        _Y.append(y)
        #candi_dict[file_name] = [x_term, y]

    X = np.array(_X); Y = np.array(_Y)
    return [X, Y]

def convert_amp(x):

    ux = x[0]
    uy = x[1]
    uz = x[2]
#    ux = 0
#    uy = 0
#    uz = 0

    n1 = x[3]
    n2 = x[4]
    n3 = x[5]
#    n4 = 0
#    n5 = 0
#    n6 = 0

    n4 = x[6]*2
    n5 = x[7]*2
    n6 = x[8]*2

    #soft = [ux**2+uy**2+uz**2, (ux**2+uy**2+uz**2)**2, ux**2*uy**2+uy**2*uz**2+uz**2*ux**2, (ux**2+uy**2+uz**2)**3, ux**4*(uy**2+uz**2) + uy**4*(ux**2+uz**2) + uz**4*(ux**2+uy**2), ux**2*uy**2*uz**2, (ux**2+uy**2+uz**2)**4]
    soft = [ux**2+uy**2+uz**2, (ux**2+uy**2+uz**2)**2, ux**2*uy**2+uy**2*uz**2+uz**2*ux**2]
    elas = [0.5*(n1**2+n2**2+n3**2), n1*n2+n2*n3+n3*n1, 0.5*(n4**2+n5**2+n6**2)]
    #elas = [0.5*(n1**2+n2**2+n3**2)*127, (n1*n2+n2*n3+n3*n1)*40, 0.5*(n4**2+n5**2+n6**2)*50]

    inter = [0.5*(n1*ux**2+n2*uy**2+n3*uz**2), 0.5*(n1*(uy**2+uz**2)+n2*(uz**2+ux**2)+n3*(ux**2+uy**2)), n4*uy*uz+n5*uz*ux+n6*ux*uy]
#    dummy = [n1+n2+n3]
    #dummy = [n1+n2+n3, n1*ux+n2*uy+n3*uz, n1*(uy+uz)+n2*(uz+ux)+n3*(ux+uy)]
    #dummy = [n1+n2+n3, ux*uy+uy*uz+uz*ux]
    #dummy = [n1+n2+n3, n1*ux+n2*uy+n3*uz, n1*(uy+uz)+n2*(uz+ux)+n3*(ux+uy), n4*(uy*uz)**2+n5*(uz*ux)**2+n6*(ux*uy)**2]
    #x_term = torch.tensor([soft[0], soft[1], soft[2], elas[0], elas[1], elas[2], inter[0], inter[1], inter[2]], dtype=dtype, device=device)
    return [soft, elas, inter]

def read_cof(root_path):
    with open(root_path+'results.txt') as f:
        cof_lines = f.readlines()
    info = cof_lines[-1].split()
    cof = np.array([float(num) for num in info[0].split('_') if num])
    return cof

def Linear_search(init_number):

    if restart is True:

        L_points = read_amp(root_path)
        #print("L_points[0]", L_points[0])
        lin = LinearRegression()
        reg = lin.fit(L_points[0], L_points[1])
        #info = Linear_fitting(torch.tensor(L_points[0], dtype=dtype, device=device), torch.tensor(L_points[1], dtype=dtype, device=device), lr=0.1, epochs_num=20000, method = 'SGD')
        #info = Linear_fitting(L_points[0], L_points[1], lr, epochs_num, method = 'Adam')

    if restart is not True:

        print('Intialization')

        X_turbo_V = get_initial_points(dim_V, init_number)
        Y_turbo_V = calculate_Vasp(unnormalize(X_turbo_V, fun_V.bounds)).unsqueeze(-1)

        L_points = read_amp(root_path)
        lin = LinearRegression()
        reg = lin.fit(L_points[0], L_points[1])
        #info = Linear_fitting(L_points[0], L_points[1], lr, epochs_num, method = 'Adam')
        #info = Linear_fitting(torch.tensor(L_points[0], dtype=dtype, device=device), torch.tensor(L_points[1], dtype=dtype, device=device), lr=0.1, epochs_num=20000, method = 'SGD')

    #return {'X_best': info['Parameters'], 'Best_value': info['Loss']}
    return {'Coefficient': reg.coef_, 'Intercept': reg.intercept_}


def BO_search_Vasp(acqf = 'qucb', hpar = 0.5):
    batch_size = 2
    n_init = 8  # 2*dim, which corresponds to 5 batches of 4

    ###### get Vasp parameters######

    turbo_info = restarts_BO_V(root_path)
    X_turbo_V = turbo_info[0]
    Value_true_turbo_V = turbo_info[1]
    Value_fit_turbo_V = eval_objective_Vasp(X_turbo_V).unsqueeze(-1)
    Y_turbo_V = torch.abs(torch.sub(Value_true_turbo_V, Value_fit_turbo_V))

    state = TurboState(dim_V, batch_size=batch_size, length=10, length_min=9, length_max=40, success_tolerance=2)

    NUM_RESTARTS = 10 if not SMOKE_TEST else 2
    RAW_SAMPLES = 4096 if not SMOKE_TEST else 4
    N_CANDIDATES = min(5000, max(2000, 200 * dim_V)) if not SMOKE_TEST else 4
    #N_CANDIDATES = max(5000, max(2000, 200 * dim)) if not SMOKE_TEST else 4

    try:

        # Fit a GP model
        train_Y_V = (Y_turbo_V - Y_turbo_V.mean()) / Y_turbo_V.std()
        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-7, 1e-2))
        model = SingleTaskGP(X_turbo_V, train_Y_V, likelihood=likelihood)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)


        if acqf == 'ucb':
            batch_size = 1


        # Create a batch
        X_info = generate_batch(
            state=state,
            model=model,
            X=X_turbo_V,
            Y=train_Y_V,
            batch_size=batch_size,
            n_candidates=N_CANDIDATES,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            acqf=acqf,
            hpar=hpar
        )

        X_next = X_info['X_next']
        acq = X_info['acq'].numpy()
        #print(acq)
        print('\n--------------------------------------------\n')

        print(f"{len(X_turbo_V)}) X_best_V: {unnormalize(X_next, fun_V.bounds)}")

        return {'X_best': unnormalize(X_next, fun_V.bounds),'acq':acq}

    except Exception as e:
        print('Error returned')
        traceback.print_exc()
    finally:
        return {'X_best': unnormalize(X_next, fun_V.bounds),'acq':acq}

fun_V = VASP(dim=9, negate=False).to(dtype=dtype, device=device)
fun_V.bounds[0, :] = torch.tensor([0.000, 0.000, 0.000, -0.015, -0.015, -0.015, -0.008, -0.008, -0.008])
fun_V.bounds[1, :] = torch.tensor([0.175, 0.175, 0.175,  0.015,  0.015,  0.015,  0.008,  0.008,  0.008])

print(fun_V.bounds)
dim_V = fun_V.dim

def eval_objective_Vasp(x):
    """This is a helper function we use to unnormalize and evalaute a point"""
    return fun_V(unnormalize(x, fun_V.bounds))


@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 10
    length_min: float = 0.1 ** 7
    length_max: float = 100
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 3  # Note: The original paper uses 3
    best_value: float = -float("inf")
    #best_n_value: List[float]=field(default_factory=list)
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )

def update_state(state, Y_next):
    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0


    state.best_value = max(state.best_value, max(Y_next).item())


    #Y_top = torch.topk(Y_next.squeeze(-1), k=math.ceil(state.batch_size/4), dim=0, largest=True)[0]
    #print(state.best_n_value)
    #state.best_n_value = torch.topk(torch.cat((torch.tensor(state.best_n_value, dtype=dtype, device=device), Y_top),0), k=math.ceil(state.batch_size/4), dim=0, largest=True)[0].cpu().tolist()
    #numpy.maximum(state.best_n_value, torch.topk(Y_next.squeeze(-1), k=math.ceil(state.batch_size/4), dim=0, largest=True)[0].cpu())
    #print(state.best_value, state.best_n_value)


    if state.length < state.length_min:
        state.restart_triggered = True
    return state


def restarts_BO_V(root_path, file='dict_candi'):

    with open(root_path+file, 'r') as f:
        candi_lines = f.readlines()
    candi_dict = {}
    for i in candi_lines:
        info = i.split()
        file_name = info[0]
        x = [float(num) for num in info[1].split('_') if num]
        y = float(info[2])
        candi_dict[file_name] = [x, y]

    folders = os.popen('ls '+path).read()

    #print(folders)

    tensor_x = []
    tensor_y = []
    for i in candi_dict:
        x = candi_dict[i][0]
        y = candi_dict[i][1]

        tensor_x.append(x)
        tensor_y.append(-y)

    X = normalize(torch.tensor(tensor_x, dtype=dtype, device=device), fun_V.bounds)
    Y = torch.tensor(tensor_y, dtype=dtype, device=device).unsqueeze(-1)

    return [X, Y]


def generate_batch(
    state,
    model,  # GP model
    X,  # Evaluated points on the domain [0, 1]^d
    Y,  # Function values
    batch_size,
    n_candidates=None,  # Number of candidates for Thompson sampling
    num_restarts=2,
    raw_samples=512,
    acqf="qucb",  # "ei" or "ts"
    hpar=10000,
):
    assert acqf in ("ts", "ei", "ucb","qucb")
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

    # Scale the TR to be proportional to the lengthscales
    x_center = X[Y.argmax(), :].clone()
    weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

    if acqf == "ts":
        dim = X.shape[-1]
        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = (
            torch.rand(n_candidates, dim, dtype=dtype, device=device)
            <= prob_perturb
        )
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

        # Create candidate points from the perturbations and the mask
        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand[mask] = pert[mask]

        # Sample on the candidate points
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        X_next = thompson_sampling(X_cand, num_samples=batch_size)


    elif acqf == "ei":
        ei = qExpectedImprovement(model, train_Y.max(), maximize=True)
        X_next, acq_value = optimize_acqf(
            ei,
            bounds=torch.stack([tr_lb, tr_ub]),
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )

    elif acqf == "qucb":
        qucb = qUpperConfidenceBound(model, hpar)
        X_next, acq_value = optimize_acqf(
            qucb,
            bounds=torch.stack([tr_lb, tr_ub]),
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )

    elif acqf == "ucb":
        ucb = UpperConfidenceBound(model, beta = hpar)
        #ucb = UpperConfidenceBound_spec(model, beta = hpar)
        X_next, acq_value = optimize_acqf(
            ucb,
            bounds=torch.stack([tr_lb, tr_ub]),
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )

    return {'X_next': X_next, 'acq':acq_value}

if __name == "__main__":
    print('BO test')
