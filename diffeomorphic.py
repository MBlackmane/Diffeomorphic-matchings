import math as m
import re


import click
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import torch




class Mesure:

    def __init__(self, point, poid):
        self.point = point  
        self.poid = poid

def ker_mat_torch(x, sigma):
  return(torch.exp(-x/(sigma**2))) # 

def fv_exp(x, y, sigma):
    return(torch.exp(-torch.norm(((x-y)/sigma))**2)) # Compute the norm in the hilbert space using RKHS

def test_j1_torch(alpha, mu, sigmaV):
  return(torch.sum(torch.mul(torch.mm(alpha, torch.t(alpha)), ker_mat_torch(torch.sum((torch.transpose(mu.point[np.newaxis, :, :], 1, 0)-mu.point)**2, dim = 2), sigmaV))))

def test_j2_torch(alpha, mu, nu, p, sigmaI):
  A = torch.sum(torch.mul(torch.mm(mu.poid[:, np.newaxis], torch.t(mu.poid[:, np.newaxis])), ker_mat_torch(torch.sum((torch.transpose(p[np.newaxis, :, :], 1, 0)- p)**2, dim = 2), sigmaI)))
  B = torch.sum(torch.mul(-2 * torch.mm(mu.poid[:, np.newaxis], torch.t(nu.poid[:, np.newaxis])), ker_mat_torch(torch.sum((torch.transpose(p[np.newaxis, :, :], 1, 0) - nu.point)**2, dim = 2), sigmaI)))
  C = torch.sum(torch.mul(torch.mm(nu.poid[:, np.newaxis], torch.t(nu.poid[:, np.newaxis])), ker_mat_torch(torch.sum((torch.transpose(nu.point[np.newaxis, :, :], 1, 0) - nu.point)**2, dim = 2), sigmaI)))
  return(A + B + C)

def grad_descent2(mu, nu, delta, n, sigmaI, sigmaV, sigmaR, alpha):
    r = 0
    while r <= n:     
        test_j1_torch(alpha, mu, sigmaV).backward()
        gradient_1 = alpha.grad
        p = mu.point + construction_v(mu, alpha, sigmaV)
        test_j2_torch(alpha, mu, nu, p, sigmaI).backward()
        gradient_2 = alpha.grad
        alpha.data = alpha.data - delta*(gradient_1 + (1/sigmaR**2)*gradient_2)
        alpha.grad.zero_()
        r += 1
    return(alpha)

def construction_v(mu, alpha1, sigmaV):
  v_opt = torch.mm(torch.t(ker_mat_torch(torch.sum((torch.transpose(mu.point[np.newaxis, :, :], 1, 0)-mu.point)**2, dim = 2), sigmaV)), alpha1)
  return(v_opt)

def import_matrices(origin_name):
    # Import check for filename extension
    if not re.search(r'.*?\.mat$', origin_name):
        raise Exception('Wrong extension name {}, should be .mat'.format(origin_name))
        return()
    else:
        return(loadmat(origin_name))


@click.command()
@click.option('--origin', help='origin matrix file, (x,y)')
@click.option('--sigma_i', help='Range of the norm, default behavior assume data in [0, 1]', default=0.4)
@click.option('--sigma_v', help='Range of the norm, pratically same as sigma_i', default=0.4)
@click.option('--sigma_r', help='Trade-off parameter between regularity of deformation and precision of the method', default=0.1)
@click.option('--precision', is_flag=True, help="If set to true, will execute an additional step for precision improvment. default:True", default=True)

def main(*args, **kwargs):
    x_4 = import_matrices(origin_name=kwargs['origin'])
    x4, y4 = [v for (k, v) in x_4.items() if not k.startswith('__')]
    sigmaI = kwargs['sigma_i']
    sigmaV = kwargs['sigma_v']
    sigmaR = kwargs['sigma_r']
    FLAG_PRECISION = kwargs['precision']
    mu = Mesure(torch.tensor(x4, dtype = torch.float32), 1*torch.rand(x4.shape[0])) # Origin points with random weights given to data to illustrate
    nu = Mesure(torch.tensor(y4, dtype = torch.float32), 10*torch.rand(y4.shape[0])) # Target points again with randomized weights

    alpha_0 = torch.zeros(np.shape(mu.point), requires_grad = True)
    alpha_test = grad_descent2(mu, nu, 0.0000000001, 20, sigmaI, sigmaV, sigmaR, alpha_0 )
    v_test = construction_v(mu, alpha_test, sigmaV)

    if FLAG_PRECISION == True:
        for j in range(4):
            sigmaI = sigmaI/2
            alpha_test = grad_descent2(mu, nu, 0.0000000001, 100, sigmaI, sigmaV, sigmaR, alpha_test)

    v_test = construction_v(mu, alpha_test, sigmaV).detach().numpy()
    plt.title("Appariemment de points :")
    plt.scatter(mu.point[:, 0], mu.point[:, 1], c='r')
    plt.scatter(nu.point[:, 0], nu.point[:, 1], c='g')
    plt.legend(("Points de base", "Points cibles"))
    m, d = np.shape(mu.point)
    for ligne in range(0, m):
        plt.arrow(mu.point[ligne, 0], mu.point[ligne, 1], v_test[ligne, 0], v_test[ligne, 1])
    plt.show()


if __name__ == '__main__':
    main()