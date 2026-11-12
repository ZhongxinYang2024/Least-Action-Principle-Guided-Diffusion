import numpy as np
import os, sys
sys.path.append(f'{os.getcwd()}')

from HamiltonSys.ZhukovskyTransfomation import Joukowski_airfoil
from HamiltonSys.Mesh import StructuredGrid

def FoilAnditsMesh_Generation(center, radius, nr, ntheta, U_inf=1.0, Aoa=0.0, gamma=0.0):
    
    params = {'degree_of_foreaft_asymmetry':   0.0, #  e.g. tau=0 for ellipses
        'degree_of_smoothness_trailing':   0.0,# e.g. D=0 for sharp trailing edge D=1 for identity map 
        'transform_param':              1.0, # b
        'half_chord_length':         1.0,
        'center_cylinder':       center,
        'radius_cylinder': radius,
        'free_flow': U_inf,
        'angle_of_attack': Aoa,
        'gamma': gamma,
        'IF_modified': False,   
        }  
    
    foil_std = Joukowski_airfoil(params)
    foil_std.Get_Zetab_cylinder(num_points=256)
    foil_std.Joukowski_transform()
    
    Zetab_Mesh = foil_std.Get_Zetab_flowfield_Mesh(100.0, nr, ntheta, ratio=1.08)
    Z_Mesh = foil_std.Joukowski_transform(Zetab_Mesh)
    
    return foil_std, Z_Mesh


def WriteBc2d2File(grid: StructuredGrid,  filename: str, info: str = ""):
    """
    write 2D boundary condition data to a file
    
    参数:
    bc_data: List of tuples, 每个元组包含 (block_no, side_no, bc_type)
    filename: 输出文件名
    """
    with open(filename, 'w') as f:
        f.write(f" {info}\n")  # 可选信息行
        f.write(" # Blocks\n")
        f.write(f"          {grid.num_blocks}\n")
        for i, block in enumerate(grid.blocks):
            nx, ny = block['nx'], block['ny']
            f.write(f" Block {i+1}\n")
            f.write(f" Subface\n")
            f.write(f"          4\n")  
            f.write("  f_no, face, istart, iend, jstart, jend,  neighb, subface orient\n")  # 假设每个块都有4个边界面
            f.write(f"  {1}  {1}  {1}  {1}   {1}  {ny}   {-30}  {0}   {0}\n") # i-
            f.write(f"  {2}  {2}  {1}  {nx}  {1}  {1}   {1}   {4}    {2}\n") # j-
            f.write(f"  {3}  {3}  {nx}  {nx}  {1}  {ny} {-201}  {0}   {0}\n") # i+
            f.write(f"  {4}  {4}  {1}  {nx}  {ny}  {ny}  {1}  {2}   {2}\n")  #  j+
    
def WriteControl2File(filename: str, aoa: float):
    with open(filename, 'w') as f:
        f.write("Ma      Re      gamma      AoA     Pr     t_end   Kstep_save Iflag_viscous Turbulence_model Iflag_init\n") 
        f.write(f"0.10d0  6.d10    1.4d0     {aoa:.2f}d0   0.77d0  2.d0    20000       0               0         0\n")      
        f.write("Iflag_local_dt, dt_global, CFL, dtmax,    dtmin   Time_Method   P_outlet        T_inf   Twall vt_inf  Kt_inf   Wt_inf\n")
        f.write("1               2.d-5      10.    0.1     1.d-7     0       -1.d0       288.15      -1.0      3.0      1.d-8   1.d-6\n")
        f.write("Iflag_scheme Iflag_flux  IFlag_Reconstruction   Kstep_show\n") 
        f.write("2          4          1                      50\n")
        f.write("Num_Mesh  Num_THREADS  Nstep_Inner_Limit    Res_Inner_Limit\n")
        f.write("1         1               20                  1.d-6\n")
        f.write("Pre_Step_Mesh(1)   Pre_Step_Mesh(2)   Pre_Step_Mesh(3)\n")
        f.write("0           0              0\n")
        f.write("If_agptbl\n")
        f.write("0\n")
'''
  integer,parameter:: Scheme_UD1=0,Scheme_NND2=1, Scheme_UD3=2,Scheme_WENO3=3,Scheme_MUSCL2=4,Scheme_MUSCL3=5 ,Scheme_OMUSCL2=6 
  integer,parameter:: Flux_Steger_Warming=1, Flux_HLL=2, Flux_HLLC=3,Flux_Roe=4,Flux_VanLeer=5,Flux_Ausm=6
  integer,parameter:: Reconst_Original=0,Reconst_Conservative=1,Reconst_Characteristic=2
'''


def WriteKeyInfo2File(filename: str, Creal, Cimag, aoa, grid_x, grid_y):
    np.savez(filename, 
             Creal=Creal, 
             Cimag=Cimag, 
             Aoa=aoa, 
             grid_x=grid_x,
             grid_y=grid_y)

def LoadKeyInfoFromFile(filename: str):
    loaded = np.load(filename, allow_pickle=True)

    Creal = loaded['Creal']
    Cimag = loaded['Cimag']
    aoa = loaded['Aoa']
    grid_x = loaded['grid_x']
    grid_y = loaded['grid_y']
    
    loaded.close()
    
    return Creal, Cimag, aoa, grid_x, grid_y
        
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from HamiltonSys.Mesh import DrawNonuniformGrid
    
    def PrepareFolder(casefolder,
                    center, radius,
                    nr, ntheta,
                    aoa,
                    If_plot=False):

        os.makedirs(casefolder, exist_ok=True)
        
        foil, Z_Mesh = FoilAnditsMesh_Generation(center=center, radius=radius, nr=96, ntheta=128)
        # save foil data#################
        np.savetxt(os.path.join(casefolder, 'foilpnt.dat'), np.c_[foil.z_foil_pnts.real, foil.z_foil_pnts.imag])
        #########################
        
        grid = StructuredGrid(num_blocks=1)
        grid.add_block(block_no=1, nx=nr, ny=ntheta, x=Z_Mesh.real, y=Z_Mesh.imag)
        
        # save foil data#################
        grid.write_grid_to_Fortranfile(os.path.join(casefolder, 'Mesh2d.dat'))
        #########################
        
        # save bc data#################
        WriteBc2d2File(grid, os.path.join(casefolder, 'bc2d_1.in'), info=f"2D mesh around Joukowski airfoil centered at {center}, radius {radius}")
        #########################
  
        # save control data#################
        WriteControl2File(os.path.join(casefolder, 'control.in'), aoa=aoa)
        ################################
        
        WriteKeyInfo2File(os.path.join(casefolder, 'keyInfo2d.npz'), center.real, center.imag, aoa, Z_Mesh.real, Z_Mesh.imag)
        LoadKeyInfoFromFile(os.path.join(casefolder, 'KeyInfo2d.npz'))
        
        if If_plot:
            DrawNonuniformGrid(Z_Mesh.real, Z_Mesh.imag)
            
        print(f"Finish prepare case folder: {casefolder}")
#%%     

    # Aoa = np.linspace(0.0, 10.0, 11); Creal = np.linspace(-1.0, 0.0, 30); Cimag = [0.0]; Radius = np.linspace(1.0 , 2.0 , 30)  + 2e-2; #training set
    Aoa = np.linspace(20.0, 30.0, 11); Creal = [-0.1724137921]; Cimag = [0.0]; Radius = [1.2268965517]  # T2
    # Aoa = np.linspace(20.0, 30.0, 11); Creal = [-0.1500]; Cimag = [0.10]; Radius = [1.20] # T3
                                                                                                     
    for i_aoa in Aoa:
        for i_creal in Creal:
            for i_cimag in Cimag:
                for i_radius in Radius:
                    if i_radius < np.sqrt(i_creal**2 + i_cimag**2) + 1.0:
                        print(f"Skip invalid case: Creal={i_creal}, Cimag={i_cimag}, radius={i_radius}")
                        continue
                    
                    center = i_creal + i_cimag * 1j; radius = i_radius;
                    nr=96; ntheta=128
                    casefolder = f'./data/Cases/C2/C_CrealNeg{str(int(-i_creal*1e5)).zfill(5)}_Cimag{str(int(i_cimag*1e5)).zfill(5)}_R{str(int(i_radius*1e4)).zfill(5)}_Aoa{str(int(i_aoa*1e1)).zfill(3)}'
                    PrepareFolder(casefolder, center, radius, nr, ntheta, i_aoa, If_plot=False)
    
