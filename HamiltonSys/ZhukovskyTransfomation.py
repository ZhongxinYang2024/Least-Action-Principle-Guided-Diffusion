import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class Joukowski_airfoil():
    # z --> zetabb -->(zetabb - mu = )zetab -->(zetab - c =) zeta (where circle's center is in origin)
    def __init__(self, params):
        self.half_chord_length = params['half_chord_length']     # control the length of chord
        self.tau = params['degree_of_foreaft_asymmetry'] #e.g. tau=0 for ellipses
        self.D  = params['degree_of_smoothness_trailing'] # e.g. D=0 for sharp trailing edge D=1 for identity map 
        self.b_J = params['transform_param']
        self.center = params['center_cylinder']
        self.radius = params['radius_cylinder']
        self.U_inf  = params['free_flow']
        self.Aoa    = params['angle_of_attack']
        self.gamma  = params['gamma']
        self.IF_modified = params['IF_modified']
        
    def Get_Zetab_cylinder(self, num_points=200):
        """
        generate points on the circle
        
        Arg:
        center: complex, the position of center
        radius: real number, the radius
        num_points: int, how many points on the circle
        
        Return:
        zeta: complex number array, one the zeta-plane
        """
        theta = np.linspace(0, 2*np.pi, num_points)
        self.zetab_cylinder_pnts = self.center + self.radius * np.exp(1j * theta)
    
    
    def _Joukowski_transform(self, zetab_pnts=None):
        """
        apply: z = ζbb + b²/ζbb
        
        Arg:
        zeta: Complex number of complex number array, the points before transformation
        b: real number, control the transformation
        
        return: 
        z: Complex number of complex number array, the points aftern transformation
        """
        if zetab_pnts is None:
            b_squared = self.b_J**2
            zetabb_pnts = self.zetab_cylinder_pnts + 0. # here mu = 0
            self.z_foil_pnts = zetabb_pnts + b_squared / zetabb_pnts
        else:
            b_squared = self.b_J**2
            zetabb_pnts = zetab_pnts + 0. # here mu = 0
            return zetabb_pnts + b_squared / zetabb_pnts
    
    
    def _Joukowski_transform_modified(self, zetab_pnts=None):
        '''
        Arg: b is the half chord length
        '''
        tau = self.tau
        D_ = self.D
        b = self.half_chord_length 
        eps = (4. * tau / 3. / np.sqrt(3) )
        C_ = b * (1. + D_) * (1. + 2. * eps) / 2. / (1. + 2.*eps + eps**2 + D_*eps*(1. + eps) )
        mu = - eps * C_ 
        
        b_squared = (1. - D_)/(1. + D_) * C_**2
           
        
        if zetab_pnts is None:
            zetabb_pnts = self.zetab_cylinder_pnts + mu
            self.z_foil_pnts = zetabb_pnts + b_squared / zetabb_pnts
        else:
            zetabb_pnts = zetab_pnts + mu
            return zetabb_pnts + b_squared / zetabb_pnts
        
    def Joukowski_transform(self, zetab_pnts=None):
        # if zetabb_pnts is None, then use the stored circle points
        # else use the input points for transformation usually for flow field grid
        if self.IF_modified:
            return self._Joukowski_transform_modified(zetab_pnts)
        else:
            return self._Joukowski_transform(zetab_pnts)
    

    def Inverse_joukowski(self, z_pnts):
        """
        稳健的茹科夫斯基逆变换实现
        
        参数:
        z_points: 复数或复数数组，需要变换的点
        b: 实数，茹科夫斯基变换参数
        modified: 布尔值，是否使用修改版本
        tau, D, half_chord_length: 修改版本所需的参数
        
        返回:
        变换后的复数或复数数组
        """
        if self.IF_modified:
            tau = self.tau
            D = self.D
            b = self.half_chord_length
            
            # 计算修改版本的参数
            eps = (4. * tau / 3. / np.sqrt(3))
            C_ = b * (1. + D) * (1. + 2. * eps) / 2. / (1. + 2.*eps + eps**2 + D*eps*(1. + eps))
            b_squared = (1. - D)/(1. + D) * C_**2
            mu = -eps * C_ + 0.0j
        else:
            b_squared = self.b_J**2
            mu = 0.0 + 0.0j  # 确保是复数类型

        root = np.sqrt(z_pnts**2 - 4*b_squared)
        
        # initial results
        result = np.zeros_like(z_pnts, dtype=complex)
        
        zetab1 = (z_pnts + root) / 2.0;
        zetab2 = b_squared / zetab1;    

        result[np.abs(zetab1) > b_squared**0.5] =  zetab1[np.abs(zetab1) > b_squared**0.5]
        result[np.abs(zetab1) <= b_squared**0.5] = zetab2[np.abs(zetab1) <= b_squared**0.5]
        
        return result - mu

    # generate flow field grid
    def Get_Z_flowfield(self, x_range, y_range, nx=100, ny=100):
        x = np.linspace(x_range[0], x_range[1], nx)
        y = np.linspace(y_range[0], y_range[1], ny)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j*Y
        return Z

    def Get_Zetab_flowfield_Mesh(self, R_far, n_r, n_theta, ratio=1.05):
        """
        Generate a fine mesh grid in the z-plane for flow field calculation
        
        Arg:
        R_far: real number, the far field radius
        n_radial: int, number of radial points
        n_angular: int, number of angular points
        
        Return:
        Z: complex number array, the mesh grid in z-plane
        """
        # 创建极坐标网格
        Dr = R_far - self.radius
        h = Dr * (ratio - 1.)/(ratio**(n_r - 1) - 1.)
        radii = h * (ratio**np.arange(n_r) - 1.) / (ratio - 1.) + self.radius
        angles = np.linspace(0, 2*np.pi, n_theta)  # angle grid
    
        # transform to Cartesian coordinates
        r, theta = np.meshgrid(radii, angles, indexing='ij')
        x = r * np.cos(theta)
        y = r * np.sin(theta)
    
        # move to zetab-plane
        zetab  = x + 1j*y + self.center
    
        return zetab
    
    # generate potential flow over clinder（uniform flow + dipole + vortex）
    def Get_F_Inzetab(self, zetab_pnts):
        zeta_pnts = zetab_pnts - self.center
        U_inf  = self.U_inf
        radius = self.radius
        Aoa = self.Aoa 
        gamma = self.gamma
        # Uniform flow + dipole（clinder） + vortex（generate lift）
        F = U_inf * (zeta_pnts * np.exp(- 1.j * Aoa)  +  radius**2 / zeta_pnts * np.exp(+ 1.j * Aoa) ) + 1j*gamma/(2*np.pi)*np.log(zeta_pnts)
        return F


    # Compute reverse transform
    def Get_F_Inz(self, z_pnts):
        zetab_points = self.Inverse_joukowski(z_pnts)
        F = self.Get_F_Inzetab(zetab_points)
        return F


    def Get_W_Inz(self, z_pnts):
        if self.IF_modified:
            tau = self.tau
            D_ = self.D
            b = self.half_chord_length
            eps = (4. * tau / 3. / np.sqrt(3) )
            C_ = b * (1. + D_) * (1. + 2. * eps) / 2. / (1. + 2.*eps + eps**2 + D_*eps*(1. + eps) )
            b_squared = (1. - D_)/(1. + D_) * C_**2
            
            mu = - eps * C_
        else:
            b_squared = self.b_J**2
            
            mu = 0. + 0.j
        
        U_inf  = self.U_inf
        Aoa = self.Aoa
        radius  = self.radius
        gamma = self.gamma
        
        zetab_pnts = self.Inverse_joukowski(z_pnts)
        
        # Compute complex velcity
        zeta_pnts = zetab_pnts - self.center
        dF_dzeta = U_inf * (np.exp(- 1.j * Aoa)  - np.exp(+ 1.j * Aoa) * radius**2/zeta_pnts**2) + 1j*gamma/(2*np.pi*zeta_pnts)
        zetabb_pnts = zetab_pnts + mu
        dz_dzeta = 1 - b_squared/(zetabb_pnts)**2
        
        # W = dF/dz W = u - iv
        dF_dz = dF_dzeta / dz_dzeta
        return dF_dz

    def Get_W_Inzetab(self, zetab_pnts):
        zeta_pnts = zetab_pnts - self.center
        
        U_inf = self.U_inf
        Aoa = self.Aoa
        radius = self.radius
        gamma = self.gamma
        
        dF_dzetab = U_inf * (np.exp(- 1.j * Aoa)  - np.exp(+ 1.j * Aoa) * radius**2/zeta_pnts**2) + 1j*gamma/(2*np.pi*zeta_pnts)
        return dF_dzetab
    
    def Get_dWdzetab_Inzetab(self, zetab_pnts):
        zeta_pnts = zetab_pnts - self.center
        
        U_inf = self.U_inf
        Aoa = self.Aoa
        radius = self.radius
        gamma = self.gamma
        
        dW_dzetab = U_inf * (  2. * np.exp(+ 1.j * Aoa) * radius**2/zeta_pnts**3) - 1j*gamma/(2*np.pi*zeta_pnts**2)
        return dW_dzetab
        
    def integral_QoI_in_zeta_np(self, num_pnts, R_max=100.0):
        if self.IF_modified:
            tau = self.tau
            D_ = self.D
            b = self.half_chord_length
            eps = (4. * tau / 3. / np.sqrt(3) )
            C_ = b * (1. + D_) * (1. + 2. * eps) / 2. / (1. + 2.*eps + eps**2 + D_*eps*(1. + eps) )
            b_squared = (1. - D_)/(1. + D_) * C_**2
            
            mu = - eps * C_
        else:
            b_squared = self.b_J**2
            
            mu = 0. + 0.j
        
        radius = self.radius
        
        n_r = num_pnts[0]; n_theta = num_pnts[1]
        
        # rr_ = np.linspace(radius, R_max, n_r)
        rr_ = np.geomspace(radius, R_max, n_r)
        rr =  (rr_[1:] + rr_[:-1]) / 2.
        dr = rr_[1:] - rr_[:-1]
        
        theta = np.linspace(0, 2 * np.pi, n_theta)
        dtheta = theta[1] - theta[0]
        
        ## create integral grid
        RR_Inzeta, Theta_Inzeta = np.meshgrid(rr, theta, indexing='ij')
        
        zeta_pnts = RR_Inzeta * ( np.cos(Theta_Inzeta) + 1.j * np.sin(Theta_Inzeta) )
        zetab_pnts = zeta_pnts + self.center
        zetabb_pnts = zetab_pnts + mu
        
        dz_dzeta = 1 - b_squared/zetabb_pnts**2
        G_pnts = 1. / (dz_dzeta)
        G_squared_pnts = np.abs(G_pnts)**2
        G_pnts_conj = np.conjugate(G_pnts)
        dG_dzeta_conj = np.conjugate(- 2. * b_squared  * zetabb_pnts/ (zetabb_pnts**2 - b_squared)**2 )
        
        W_pnts = self.Get_W_Inzetab(zetab_pnts)
        dW_dzeta_conj = np.conjugate(self.Get_dWdzetab_Inzetab(zetab_pnts))
        a_pnts = W_pnts * dW_dzeta_conj
        
        Weigth = G_squared_pnts * np.abs(G_pnts_conj * a_pnts + np.abs(W_pnts)**2 * dG_dzeta_conj)**2
    
        return  np.sum(RR_Inzeta * Weigth * dr[:, np.newaxis] ) / 2. * dtheta 


    def integral_QoI_in_zeta_torch(self, num_pnts, R_max=100.0):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.float64
        
        if self.IF_modified:
            tau = self.tau
            D_ = self.D
            b = self.half_chord_length
            eps = (4. * tau / 3. / torch.sqrt(torch.tensor(3.0, device=device)))
            C_ = b * (1. + D_) * (1. + 2. * eps) / 2. / (1. + 2.*eps + eps**2 + D_*eps*(1. + eps))
            b_squared = (1. - D_)/(1. + D_) * C_**2
            mu = - eps * C_
        else:
            b_squared = self.b_J**2
            mu = torch.tensor(0.0, device=device)
        
        radius = self.radius
        n_r = num_pnts[0]; n_theta = num_pnts[1]
        
        # 
        #rr_ = torch.linspace(radius, R_max, n_r, device=device, dtype=dtype) 
        ratio = (R_max/radius) ** (1.0 / (n_r - 1))
        rr_ = radius * ratio ** torch.arange(n_r, device=device, dtype=dtype)
        
        rr = (rr_[1:] + rr_[:-1]) / 2.
        dr = rr_[1:] - rr_[:-1]  # 每个区间的实际宽度
        
        # 角度方向
        theta = torch.linspace(0, 2 * torch.pi, n_theta, device=device, dtype=dtype)
        dtheta = theta[1] - theta[0]
        
        # 创建网格
        RR_Inzeta, Theta_Inzeta = torch.meshgrid(rr, theta, indexing='ij')
        
        # 复数点计算
        center = self.center if isinstance(self.center, torch.Tensor) else torch.tensor(self.center, device=device)
        zeta_pnts = RR_Inzeta * (torch.cos(Theta_Inzeta) + 1.j * torch.sin(Theta_Inzeta))
        zetab_pnts = zeta_pnts + center
        zetabb_pnts = zetab_pnts + mu
        
        # 复数导数计算
        dz_dzeta = 1 - b_squared / zetabb_pnts**2
        G_pnts = 1. / dz_dzeta
        G_squared_pnts = torch.abs(G_pnts)**2
        G_pnts_conj = torch.conj(G_pnts)
        dG_dzeta_conj = torch.conj(-2. * b_squared * zetabb_pnts / (zetabb_pnts**2 - b_squared)**2)
        
        # 获取 W 函数及其导数
        W_pnts = self.Get_W_Inzetab(zetab_pnts)
        dW_dzeta_conj = torch.conj(self.Get_dWdzetab_Inzetab(zetab_pnts))
        a_pnts = W_pnts * dW_dzeta_conj
        
        # 计算权重
        Weigth = G_squared_pnts * torch.abs(G_pnts_conj * a_pnts + torch.abs(W_pnts)**2 * dG_dzeta_conj)**2
        
        # 积分求和
        return torch.sum(RR_Inzeta * Weigth * dr.unsqueeze(1)) / 2. * dtheta 

def GetAFoil(aoa, creal, cimag, radius):
    rho = 1.0; U_inf = 1.0; Aoa = aoa / 180 * np.pi; gamma= 0.00
    
    # get geom
    center = creal + cimag * 1j; # real controls left right symmetry, imag control upper and lower symetry, origin should in circle
    radius = radius;  # thickness, must bigger than 
    
    # standard transformation parameters
    b_J = 1.00
    # flat plate = 0 + 0j; radius = 1.0; b_J = 1.0 # standard 
    
    # Taha modified transformation parameters 
    tau = 0.0; D = 0.0   # only for modified
    
    # params setting 
    params = {'degree_of_foreaft_asymmetry':   tau, #  e.g. tau=0 for ellipses # only for modified
            'degree_of_smoothness_trailing':   D,# e.g. D=0 for sharp trailing edge D=1 for identity map  # only for modified
            'half_chord_length':         1.0, # only for modified
            'transform_param':        b_J, # b
            'center_cylinder':       center,
            'radius_cylinder': radius,
            'free_flow': U_inf,
            'angle_of_attack': Aoa,
            'gamma': gamma,
            'IF_modified': False,   
            } 


    foil_std = Joukowski_airfoil(params)
    
    foil_std.Get_Zetab_cylinder(num_points=200)
    foil_std.Joukowski_transform()

    
    # get flow field
    Zetab_Mesh = foil_std.Get_Zetab_flowfield_Mesh(100.0, 96, 128, ratio=1.05)
    Z_Mesh = foil_std.Joukowski_transform(Zetab_Mesh)

    return foil_std, Z_Mesh
if __name__ == '__main__':
    foil_std, Z_Mesh = GetAFoil(0.0, -0.15, 0.10, 1.20)

    # 计算变换后的流动
    F_grid = foil_std.Get_F_Inz(Z_Mesh)
    phi = F_grid.real  # 势函数
    psi = F_grid.imag  # 流函数


    W = foil_std.Get_W_Inz(Z_Mesh)
    velocity_magnitude = np.abs(W)


    
    circle_std = foil_std.zetab_cylinder_pnts
    foilpnt_std = foil_std.z_foil_pnts
    S = foil_std.integral_QoI_in_zeta_np([96,128], R_max=100.0)
    print(S)
#%% plot
    fig = plt.figure()

    # plot airfoil on z-plane
    ax1 = fig.add_subplot(3,3,1)
    ax1.plot(circle_std.real, circle_std.imag,'b', label='Circle')
    ax1.plot(foilpnt_std.real, foilpnt_std.imag,'r-', label='Joukovski airfoil')
    ax1.set_xlabel('Re(ζ)')
    ax1.set_ylabel('Im(ζ)')
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = fig.add_subplot(3, 3, 2)
    ax2.contour(Z_Mesh.real, Z_Mesh.imag, psi, levels=50, colors='Blue', linestyles='solid', alpha=0.5)
    ax2.plot(foilpnt_std.real, foilpnt_std.imag, 'r-', linewidth=2)
    ax2.fill(foilpnt_std.real, foilpnt_std.imag, 'black', alpha=1.0)
    ax2.set_xlabel('Re(z)')
    ax2.set_ylabel('Im(z)')
    ax2.grid(True, alpha=0.3)



    ax4 = fig.add_subplot(3, 3, 4)
    contourf_plot = ax4.contourf(Z_Mesh.real, Z_Mesh.imag, W.real, levels=20)
    cbar = plt.colorbar(contourf_plot, ax=ax4)
    cbar.set_label('Velocity Magnitude')
    cl = ax4.contour(Z_Mesh.real, Z_Mesh.imag, W.real, levels=10, colors='black', alpha=0.3)
    ax4.clabel(cl, inline=True, fontsize=8, fmt='%.2f')
    ax4.plot(foilpnt_std.real, foilpnt_std.imag, 'r-', linewidth=2)
    ax4.fill(foilpnt_std.real, foilpnt_std.imag, 'black', alpha=1.0)

    
    ax5 = fig.add_subplot(3, 3, 5)
    contourf_plot = ax5.contourf(Z_Mesh.real, Z_Mesh.imag, -W.imag, levels=20)
    cbar = plt.colorbar(contourf_plot, ax=ax5)
    cbar.set_label('Velocity Magnitude')
    cl = ax5.contour(Z_Mesh.real, Z_Mesh.imag, -W.imag, levels=10, colors='black', alpha=0.3)
    ax5.clabel(cl, inline=True, fontsize=8, fmt='%.2f')
    ax5.plot(foilpnt_std.real, foilpnt_std.imag, 'r-', linewidth=2)
    ax5.fill(foilpnt_std.real, foilpnt_std.imag, 'black', alpha=1.0)
    
    
    ax6 = fig.add_subplot(3, 3, 6)
    contourf_plot = ax6.contourf(Z_Mesh.real, Z_Mesh.imag, velocity_magnitude, levels=20)
    cbar = plt.colorbar(contourf_plot, ax=ax6)
    cbar.set_label('Velocity Magnitude')
    cl = ax6.contour(Z_Mesh.real, Z_Mesh.imag, velocity_magnitude, levels=10, colors='black', alpha=0.3)
    ax6.clabel(cl, inline=True, fontsize=8, fmt='%.2f')
    ax6.plot(foilpnt_std.real, foilpnt_std.imag, 'r-', linewidth=2)
    ax6.fill(foilpnt_std.real, foilpnt_std.imag, 'black', alpha=1.0)

    


    plt.tight_layout()
    plt.show()

    # 打印翼型参数
    chord_length = max(foilpnt_std.real) - min(foilpnt_std.real)
    thickness = max(foilpnt_std.imag) - min(foilpnt_std.imag)
    print(f"where is the leading edge: {min(foilpnt_std.real):.3f}")
    print(f"chord: {chord_length:.3f}")
    print(f"thickness: {thickness:.3f}")
    print(f"ratio: {thickness/chord_length*100:.2f}%")
    
    from Mesh import DrawNonuniformGrid
    DrawNonuniformGrid(Z_Mesh.real, Z_Mesh.imag)
    
