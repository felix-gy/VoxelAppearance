import math
import numpy as np
import torch

def generalized_binomial_coeff(a, k):
    return np.prod(a - np.arange(k)) / math.factorial(k)

def assoc_legendre_coeff(l, m, k):
    return ((-1)**m * 2**l * math.factorial(l) / math.factorial(k) /
            math.factorial(l - k - m) *
            generalized_binomial_coeff(0.5 * (l + k + m - 1.0), l))

def sph_harm_coeff(l, m, k):
    return (np.sqrt((2.0 * l + 1.0) * math.factorial(l - m) /
            (4.0 * np.pi * math.factorial(l + m))) * assoc_legendre_coeff(l, m, k))

def get_ml_array(deg_view):
    ml_list = []
    for i in range(deg_view):
        l = 2**i
        for m in range(l + 1):
            ml_list.append((m, l))
    return np.array(ml_list).T

def generate_ide_fn_pytorch(deg_view):
    if (
        isinstance(deg_view, bool)
        or not isinstance(deg_view, int)
        or not 1 <= deg_view <= 5
    ):
        raise ValueError("deg_view debe ser un entero entre 1 y 5.")

    ml_array = get_ml_array(deg_view)
    l_max = 2 ** (deg_view - 1)

    mat = np.zeros(
        (l_max + 1, ml_array.shape[1]),
        dtype=np.float64,
    )

    for i, (m, l) in enumerate(ml_array.T):
        for k in range(l - m + 1):
            mat[k, i] = sph_harm_coeff(l, m, k)

    # Constantes: no requieren gradientes.
    mat_tensor = torch.from_numpy(mat)
    m_values = [int(m) for m in ml_array[0]]
    l_values = torch.from_numpy(ml_array[1].astype(np.float64))
    sigma = 0.5 * l_values * (l_values + 1)

    def integrated_dir_enc_fn(xyz, kappa_inv):
        """
        xyz: [..., 3], direcciones unitarias.
        kappa_inv: [..., 1], parámetro no negativo de suavizado.

        Ambas entradas deben tener dimensiones compatibles
        mediante broadcasting.
        """
        if xyz.dtype not in (torch.float32, torch.float64):
            raise TypeError("xyz debe ser float32 o float64.")

        if xyz.shape[-1] != 3:
            raise ValueError("xyz debe tener forma [..., 3].")

        # .to() mantiene la conexión con autograd.
        kappa_inv = kappa_inv.to(
            device=xyz.device, dtype=xyz.dtype
        )

        mat_dev = mat_tensor.to(
            device=xyz.device, dtype=xyz.dtype
        )
        sigma_dev = sigma.to(
            device=xyz.device, dtype=xyz.dtype
        )

        x = xyz[..., 0:1]
        y = xyz[..., 1:2]
        z = xyz[..., 2:3]

        vmz = torch.cat(
            [z ** k for k in range(l_max + 1)],
            dim=-1,
        )

        xy = torch.complex(x, y)
        vmxy = torch.cat(
            [xy ** m for m in m_values],
            dim=-1,
        )

        sph_harms = vmxy * (vmz @ mat_dev)
        ide = sph_harms * torch.exp(-sigma_dev * kappa_inv)

        return torch.cat([ide.real, ide.imag], dim=-1)

    return integrated_dir_enc_fn