from bo import *
from vasp import *
import torch
import os

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    SMOKE_TEST = os.environ.get("SMOKE_TEST")
    print(device)

    Compound = 'Loss_function'
    root_path = '/work/scratch/md88vyxi/calculation/Loss_b_300/'
    os.makedirs(root_path, exist_ok=True)
    path = '/work/scratch/md88vyxi/calculation/Loss_b_300/calculation_fit/'
    os.makedirs(path, exist_ok=True)
    ssh_root = '/work/scratch/md88vyxi/calculation/Loss_b_300/Botorch_calculation/calculation_BTO/BTO_fit/'
    os.makedirs(ssh_root, exist_ok=True)

    state_V = TurboState(dim=dim_V, batch_size=2)

    if os.path.isfile(root_path + 'dict_candi'):
        restart = True
    else:
        restart = False

    if restart is not True:
        os.chdir(root_path)
        result = Linear_search(10)
        #with open('results.txt', 'a') as f:
        #    f.write('_'.join([str(i.item()) for i in result['Coefficient'][0:-1]])+'   '+str(result['Intercept'])+'\n')
        restart = True

    n = 0
    while n < 100:
        os.chdir(root_path)
        result = Linear_search(10)
        with open('results.txt', 'a') as f:
            f.write('_'.join([str(i.item()) for i in result['Coefficient']])+'   '+str(result['Intercept'])+'\n')
        BO_search = BO_search_Vasp(acqf = 'ucb', hpar=1)
        VASP_coeff = BO_search['X_best'].squeeze(0)
        acq = BO_search['acq']
        #VASP_coeff = BO_search_Vasp(acqf = 'ucb', hpar=1)['X_best'].squeeze(0)
        print('acq: ', str(acq))
        #print(type(acq))
        Y_vasp = calculate_Vasp(VASP_coeff)
        with open(root_path + "/acq.dat", 'a') as f:
            f.write('_'.join([str(i) for i in VASP_coeff.numpy()]) + '  ' + str(acq) + '\n')
        n += 1

    os.chdir(root_path)
    result = Linear_search(10)
    with open('results.txt', 'a') as f:
        f.write('_'.join([str(i.item()) for i in result['Coefficient']])+'   '+str(result['Intercept'])+'\n')
