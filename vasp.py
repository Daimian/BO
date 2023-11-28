import numpy as np
import os
from typing import List, Optional, Tuple
from botorch.test_functions.base import BaseTestProblem
from torch import Tensor
from pymatgen.io.vasp.outputs import Vasprun, Oszicar

def modify_POSCAR(X, fold):
    #print(X[-3], X[-2], X[-1])
    ### X in form of [u1, u2, u3, n1, n2, n3, n4, n5, n6]
    #X[0]=0;X[1]=0;X[2]=0
    xx = round((X[3]+1.0)*3.9850223763701771, 12)
    yy = round((X[4]+1.0)*3.9850223763701771, 12)
    zz = round((X[5]+1.0)*3.9850223763701771, 12)

    yz = round(X[6]*3.9850223763701771, 12)
    zx = round(X[7]*3.9850223763701771, 12)
    xy = round(X[8]*3.9850223763701771, 12)

    #strain = np.array([xx,yy,zz,yz,zx,xy])
    direct_lattice = np.array([[xx,xy,zx],[xy,yy,yz],[zx,yz,zz]])
    #print(direct_lattice)
    direct_pos = np.array([[0.0,0.0,0.0],[0.5,0.5,0.5],[0.5,0.5,0.0],[0.5,0.0,0.5],[0.0,0.5,0.5]])
    #pos = numpy.array([[0.0, 0.0, 0.0], [1.991569996, 1.991569996, 1.991569996], [1.991569996, 1.991569996, 0.0], [1.991569996, 0.0, 1.991569996], [0.0, 1.991569996, 1.991569996]])
    pos = np.dot(direct_pos, direct_lattice)
    amp = np.array([[0.15297701, 0.77821986, -0.18479087, -0.18479087, -0.55016059],
                    [0.15297701, 0.77821986, -0.18479087, -0.55016059, -0.18479087],
                    [0.15297701, 0.77821986, -0.55016059, -0.18479087, -0.18479087]]).transpose()
    _dis = np.array([[X[0], 0.0, 0.0],[0.0, X[1], 0.0],[0.0, 0.0, X[2]]])

    #print(pos)
    #print(numpy.dot(amp, _dis))
    newpos = pos+np.dot(amp, _dis)
    with open(path+fold+'/POSCAR', 'w') as f:
        f.write('BTO\n1.000000000\n')
        for i in direct_lattice:
            f.write('   '+'   '.join([str(j) for j in i])+'\n')
        f.write('   Ba   Ti    O\n    1   1    3\nCartesian\n')
        for i in newpos:
            f.write('   '+'   '.join([str(j) for j in i])+'\n')

class SyntheticTestFunction(BaseTestProblem):
    r"""Base class for synthetic test functions."""

    _optimizers: List[Tuple[float, ...]]
    _optimal_value: float
    num_objectives: int = 1

    def __init__(self, noise_std: Optional[float] = None, negate: bool = False) -> None:
        r"""Base constructor for synthetic test functions.

        Args:
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
        """
        super().__init__(noise_std=noise_std, negate=negate)
        if self._optimizers is not None:
            self.register_buffer(
                "optimizers", torch.tensor(self._optimizers, dtype=torch.float)
            )

    @property
    def optimal_value(self) -> float:
        r"""The global minimum (maximum if negate=True) of the function."""
        return -self._optimal_value if self.negate else self._optimal_value


class VASP(SyntheticTestFunction):

    _optimal_value = 0.0
    _check_grad_at_opt: bool = False

    def __init__(self, dim: int = 3, noise_std: Optional[float] = None, negate: bool = False) -> None:
        self.dim = dim
        self._bounds = [(-0.5, 0.5) for _ in range(self.dim)]
        self._optimizers = [tuple(0.0 for _ in range(self.dim))]
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self,X,noise=0.0):
        #print(noise)
        os.chdir(root_path)
        Y = []

        if X.ndim <= 1:
            X = [X]

        cof = read_cof(root_path)
        for x in X:
            _X = []
            x_term = convert_amp(x)
            for sublist in x_term:
                for item in sublist:
                    _X.append(item)
            _x = torch.tensor(_X, dtype=dtype, device=device)
            E_Landau = 0.0
            for i in range(len(cof)):
                E_Landau += cof[i]*_x[i]
            ### to de done
            #E_Landau = cof[0]*_x[0]+cof[1]*_x[1]+cof[2]*_x[2]+cof[3]*_x[3]+cof[4]*_x[4]+cof[5]*_x[5]+cof[6]*_x[6]+cof[7]*_x[7]+cof[8]*_x[8]

            Y.append(E_Landau)

        return torch.tensor(Y, dtype=dtype, device=device)


def calculate_Vasp(X, noise=0.0):
    #print(noise)
    os.chdir(root_path)
    Y = []
    if X.ndim <= 1:
        X = [X]
    folder_dict = {}
    folder_init = int(os.popen('cat '+root_path+'dict_candi | wc -l').read())
    #print(X)
    for folder_num, x in enumerate(X):
        #folder = '_'.join(str(round(i.item(), 8)) for i in x)
        folder = str(folder_num+folder_init)
        print(folder)

        Num_x=[i.item() for i in x]

        folder_dict[folder] = [Num_x]


        if not os.path.isdir(path+folder):
            os.mkdir(path+folder)
        #os.chdir(path+folder)
        cal_judge = False
        #print(ssh_root+folder, '\n', path+folder)
        is_file = os.popen("if [ -f '"+ssh_root+folder+"/out' ]; then echo 'True'; else echo 'False'; fi").read().strip()
        if is_file == 'True':
            out_message=os.popen('cat '+ssh_root+folder+'/out').read()
            if 'writing wavefunctions' in out_message:
                print(folder+' has been calculated')
                cal_judge = True
        if cal_judge == False:
            for root, dirs, files in os.walk(root_path+'/cal_files/'):
                for file in files:
                    src_file = os.path.join(root, file)
                    shutil.copy(src_file, path+folder)

            modify_POSCAR(Num_x, folder)
            #print(Num_x)
            #ssh_transport_dir(ssh_root+folder, path+folder, 'upload')
            if os.path.exists(ssh_root+folder):
                shutil.rmtree(ssh_root+folder)
            shutil.copytree(path+folder, ssh_root+folder)
            os.popen('cd '+ssh_root+folder+'; sbatch job-vasp.sh')
            os.system('sleep 1s')

    finish_judge = True
    while finish_judge:
        squeue_info = os.popen("squeue | grep 'b50'").read().strip()
        #print(squeue_info, len(squeue_info))
        if 'RUNNING' not in squeue_info and 'PENDING' not in squeue_info and len(squeue_info)==0:
            judge_info = []
            for folder in folder_dict:
                is_file = os.popen("if [ -f '"+ssh_root+folder+"/out' ]; then echo 'True'; else echo 'False'; fi").read().strip()
                print(is_file)
                if is_file == 'True':
                    out_message=os.popen('cat '+ssh_root+folder+'/out').read()
                    if 'writing wavefunctions' not in out_message:
                        os.popen('cd '+ssh_root+folder+'; sbatch job-vasp.sh')
                        os.system('sleep 1s')
                        judge_info.append(0)
                    else:
                        judge_info.append(1)
                else:
                    judge_info.append(0)
                    os.popen('cd '+ssh_root+folder+'; sbatch job-vasp.sh')
                    os.system('sleep 1s')
            if 0 not in judge_info:
                print('VASP finished')
                break
        else:
            #print('Not yet')
            os.system('sleep 20s')

    for folder_x in folder_dict:
        folder = folder_x
        #ssh_transport_file(ssh_root+folder+'/vasprun.xml', path+folder+'/vasprun.xml', 'download')
        shutil.copy(ssh_root+folder+'/OSZICAR', path+folder+'/OSZICAR')
        try:
            #energy=float(str(Vasprun(path+folder+'/vasprun.xml').final_energy).split()[0])
            energy = float(Oszicar(path+folder+'/OSZICAR').final_energy)
            print('Y:', energy)
        except FileNotFoundError:
            print(folder)
        Y.append(energy)
        os.chdir(root_path)
        folder_dict[folder_x].append(energy)
        with open(path+folder+'/amplitude', 'w+') as inf:
            inf.write('_'.join([str(num) for num in folder_dict[folder][0]])+'   '+str(energy)+'\n')
        with open(root_path+'dict_candi','a+') as f:
            f.write(folder+'   '+'_'.join([str(num) for num in folder_dict[folder][0]])+'   '+str(folder_dict[folder][1])+'\n')


    #Y = torch.tensor(Y)
    return torch.tensor(Y, dtype=dtype, device=device)


if __name__ == "__main__":

    #path = '/work/scratch/md88vyxi/calculation/Loss_b_300/calculation_fit/'
    path = os.getcwd()
    fold = 'test'
    os.makedirs(path+fold, exist_ok=True)
    X = [0.1, 0.1, 0.1, 0.1, 0.003, 0.1, 0.1, 0.1, 0.1]
    modify_POSCAR(X, fold)
