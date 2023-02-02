import cuda.utils as utils
import torch
import numpy as np
import torch.nn.functional as F


from cuda.defs import *
from tqdm import tqdm
from dataclasses import dataclass
from warnings import warn
from functools import reduce
from torch import nn, autograd
from typing import Union, List, NamedTuple, Optional, Tuple


_C = utils.get_extension()

@dataclass
class RenderOptions():
    
    backend: str = "cuvol"
    
    background_brightness: float = 1.0

    setp_size: float = 0.5

    sigma_thresh: float = 1e-10

    stop_thresh: float = ( 1e-7 )

    last_sample_opaque: bool = False  # Make the last sample opaque (for forward-facing)
    
    near_clip: float = 0.0
    use_spheric_clip: float = False

    random_sigma_std:float = 1.0
    random_sigma_std_background: float = 1.0
    
    depth_sigma: float = 0.01  
    """Uncertainty around depth values in meters (defaults to 1cm)."""

    is_euclidean_depth: bool = False
    """Whether input depth maps are Euclidean distances (or z-distances)."""

    should_decay_sigma: bool = False
    """Whether to exponentially decay sigma."""

    sigma_decay_rate: float = 0.99985
    """Rate of exponential decay."""

    starting_depth_sigma: float = 0.2
    """Starting uncertainty around depth values in meters (defaults to 0.2m)."""


    def _to_cpp(self, randomsize:bool = False):
        """
        Generate object to pass to C++
        """

        if self.should_decay_sigma:
            self.sigma = self.starting_depth_sigma
            self.sigma = max(self.sigma_decay_rate * self.sigma , self.depth_sigma)
        else:
            self.sigma = self.depth_sigma

        opt = _C.RenderOptions()
        opt.background_brightness = self.background_brightness
        opt.step_size = self.setp_size
        opt.sigma_thresh = self.sigma_thresh
        opt.stop_thresh = self.stop_thresh
        opt.near_clip = self.near_clip
        opt.use_spheric_clip = self.use_spheric_clip

        opt.last_sample_opaque = self.last_sample_opaque

        print(self.sigma)
        opt.sigma = self.sigma
        
        return opt

@dataclass
class Rays:

    origins: torch.Tensor

    dirs: torch.Tensor

    def _to_cpp(self):

        rspec = _C.RaysSpec()

        rspec.origins = self.origins

        rspec.dirs = self.dirs

        return rspec

    def __get_item__(self, i):
        ray_sample = Rays(self.origins[i], self.dirs[i])
        return ray_sample
    
    @property
    def is_cuda(self) -> bool:
        return self.origins.is_cuda and self.dirs.is_cuda

@dataclass
class Camera:
    c2w: torch.Tensor
    fx: Optional[float]
    fy: Optional[float]   
    cx: Optional[float]
    cy: Optional[float]
    width: int
    height: int

    ndc_coeffs: Union[Tuple[float, float], List[float]] = (-1.0, 1.0)

    @property
    def fx_val(self):
        return self.fx
    @property
    def fy_val(self):
        return self.fy if self.fy is not None else self.fx
    @property
    def cx_val(self):
        return self.cx if self.cx is not None else 0.5 * self.width
    @property
    def cy_val(self):
        return self.cy if self.cy is not None else 0.5 * self.height

    @property
    def using_ndc(self):
        return self.ndc_coeffs[0] > 0.0

    def _to_cpp(self):
        camspce = _C.CameraSpec()
        camspce.c2w = self.c2W
        camspce.fx = self.fx_val
        camspce.fy = self.fy_val
        camspce.cx = self.cx_val
        camspce.cy = self.cy_val
        camspce.width = self.width
        camspce.height = self.height
        camspce.ndc_coeffx = self.ndc_coeffs[0]
        camspce.ndc_coeffy = self.ndc_coeffs[1]
        return camspce

    @property
    def is_cuda(self) -> bool:
        return self.c2w.is_cuda
    
    def gen_rays(self) -> Rays:
        origins = self.c2w[None, :3, 3].expand(self.height * self.width, -1).contiguous()
        dirs = None
        return Rays(origins=origins, dirs=dirs)


class _SampleGridAutogradFunction(autograd.Function):
    @staticmethod
    def forward(ctx, 
                data_density: torch.Tensor, 
                data_sh: torch.Tensor, 
                grid, 
                points: torch.Tensor, 
                want_colors: bool) -> None:
        assert not points.requires_grad, "Point gradient not supported"
        out_density, out_sh = _C.sample_grid(grid, points, want_colors)
        ctx.save_for_backward(points)
        ctx.grid = grid
        ctx.want_colors = want_colors
        return out_density, out_sh
            
class SparseGrid(nn.Module):

    def __init__(self,
        reso: Union[int, List[int], Tuple[int, int, int]] = 128,
        radius: Union[float, List[float]] = 1.0,
        center: Union[float, List[float]] = [0.0, 0.0, 0.0],
        device: Union[torch.device, str] = 'cpu',
        use_z_order : bool=False,
        use_sphere_bound: bool = True,
        basis_dim: int = 9,
        basis_type: int = BASIS_TYPE_SH,
        background_nlayers: int = 0
        ) -> None:
        
        super().__init__()
        
        # initial hyperparameter
        self.basis_dim = basis_dim
        self.basis_type = basis_type
        self.device = device

        self.background_nlayers = background_nlayers
        if isinstance(reso, int):
            reso = [reso] * 3
        else:
            assert(
                len(reso) == 3
            ), "reso must be an interger or indexable object of 3 ints"
            
        if isinstance(radius, float) or isinstance(radius, int):
            radius = [radius] * 3
        if isinstance(radius, torch.Tensor):
            radius = radius.to(device="cpu", dtype=torch.float32)
        else:
            radius = torch.tensor(radius, dtype=torch.float32, device="cpu")
        if isinstance(center, torch.Tensor):
            center = center.to(device="cpu", dtype=torch.float32)
        else:
            center = torch.tensor(center, dtype=torch.float32, device="cpu")

        self.radius: torch.Tensor = radius
        self.center: torch.Tensor = center
        self._offset = 0.5 * (1 - self.center / self.radius)
        self._scaling = 0.5 / radius

        n3: int = reduce(lambda x, y: x * y, reso)

        init_links = torch.arange(n3, device=device, dtype=torch.int32)

        if use_sphere_bound:
            X = torch.arange(reso[0], dtype=torch.float32, device=device) - 0.5
            Y = torch.arange(reso[1], dtype=torch.float32, device=device) - 0.5
            Z = torch.arange(reso[2], dtype=torch.float32, device=device) - 0.5
            
            X, Y, Z = torch.meshgrid(X, Y, Z)
            points = torch.stack((X, Y, Z), dim=-1).view(-1, 3)
            gsz = torch.tensor(reso)
            roffset = 1.0 / gsz - 1.0
            rscaling = 2.0 / gsz

            points = torch.addcmul(
                roffset.to(device=device),
                points,
                rscaling.to(device=device)
            )

            norms = torch.norm(points, dim=-1)
            mask = norms <= 1 + (3 ** 0.5) / gsz.max()
            self.capacity: int = mask.sum()

            data_mask = torch.zeros(n3, dtype=torch.int32, device=device)
            idxs = init_links[mask].long()
            data_mask[idxs] = 1
            data_mask = torch.cumsum(data_mask, dim=0) - 1
            
            init_links[mask] = data_mask[idxs].int()
            init_links[~mask] = -1

        else:
            self.capacity = n3


        self.density_data = nn.Parameter(torch.zeros(self.capacity, 1, dtype=torch.float32, device=device))
        
        # spherical basis functions 
        self.sh_data = nn.Parameter(torch.zeros(self.capacity, self.basis_dim * 3, dtype=torch.float32, device=device))

        self.basis_data = nn.Parameter(torch.empty(0, 0, 0, 0, dtype=torch.float32, device=device), requires_grad=False)

        self.background_links: Optional[torch.Tensor]
        self.background_data: Optional[torch.Tensor]

        self.background_data = nn.Parameter(
                torch.empty(
                    0, 0, 0,
                    dtype=torch.float32, device=device
                ),
                requires_grad=False
            )
        
        self.register_buffer("links", init_links.view(reso))  # [reso, reso, reso]
        
       
        self.links: torch.Tensor
        self.opt = RenderOptions()
        self.sparse_grad_indexer: Optional[torch.Tensor] = None
        self.sparse_sh_grad_indexer: Optional[torch.Tensor] = None
        self.sparse_background_indexer:Optional[torch.Tensor] = None
        self.density_rms:Optional[torch.Tensor] = None
        self.sh_rms: Optional[torch.Tensor] = None
        self.background_rms: Optional[torch.Tensor] = None
        self.basis_rms: Optional[torch.Tensor] = None

        if self.links.is_cuda and use_sphere_bound:
            self.accelerate()

    @property
    def use_background(self):
        return self.background_nlayers > 0

    def sample(self, points: torch.Tensor,
               use_kernel:bool,
               grid_coords: bool = False,
               want_colors: bool = True):
            if use_kernel and self.links.is_cuda and _C is not None:
                assert points.is_cuda
                return _SampleGridAutogradFunction.apply(
                    self.density_data, 
                    self.sh_data, 
                    self._to_cpp(grid_coords=grid_coords), 
                    points,
                    want_colors
                )

            pass

    # SELF FORWORD FUNCTION
    def forward(self, points: torch.Tensor, use_kernel:bool = True):
        return self.sample(points, use_kernel)

    def accelerate(self):
        """
        Accelerate
        """
        assert (
            _C is not None and self.links.is_cuda
        ), "CUDA extension is currently required for accelerate"
        _C.test_grid(self.links)


    # gain sparse grid resos
    def _grid_reso(self):
        
        return torch.tensor(self.links.shape, device='cpu', dtype=torch.float32)

    # get grad
    def _get_data_grads(self):
        ret = []
        
        for subitem in ["sh_data", "density_data", "basis_data", "background_data"]:
            param = self.__getattr__(subitem)

            if not param.requires_grad:
                ret.append(torch.zeros_like(param.data))
            else:
                if (not hasattr(param, "grad") 
                    or param.grad is None 
                    or param.grad.shape != param.data.shape
                ):
                    if hasattr(param, "grad"):
                        param.grad = torch.zeros_like(param.data)
                ret.append(param.grad)
        return ret
    
    # change python to c
    def _to_cpp(self, grid_coords:bool = False, replace_basis_data:Optional[torch.Tensor] = None):
        
        gspec = _C.SparseGridSpec()
        gspec.density_data = self.density_data
        gspec.sh_data = self.sh_data
        gspec.links = self.links
        
        if grid_coords:       
            gspec._offset = torch.zeros_like(self._offset)
            gspec._scaling = torch.ones_like(self._scaling) 
        else:
            gsz = self._grid_reso()
            gspec._offset = self._offset * gsz - 0.5
            gspec._scaling = self._scaling * gsz

        gspec.basis_dim = self.basis_dim
        gspec.basis_type = self.basis_type
        
        if replace_basis_data:
            gspec.basis_data = replace_basis_data
        elif self.basis_type == BASIS_TYPE_3D_TEXTURE:
            gspec.basis_data = self.basis_data
        
        if self.use_background:
            gspec.background_links = self.background_links
            gspec.background_data = self.background_data

       
        return gspec

    # use cuda c rendering function
    def training_render_fused(
        self, 
        rays: Rays, 
        rgb_gt: torch.Tensor, 
        randomize:bool = False, 
        beta_loss: float = 0.0,
        sparsity_loss: float = 0.0):

        grad_sh, grad_density, grad_basis, grad_bg = self._get_data_grads()
        rgb_out = torch.zeros_like(rgb_gt)
        basis_data : Optional[torch.Tensor] = None

        self.sparse_grad_indexer = torch.zeros((self.density_data.size(0)), dtype=torch.bool, device=self.density_data.device)
        grad_holder = _C.GridOutputGrads()
        grad_holder.grad_density_out = grad_density
        grad_holder.grad_sh_out = grad_sh  
        grad_holder.mask_out = self.sparse_grad_indexer

        if self.use_background:
            grad_holder.grad_background_out = grad_bg
            self.sparse_background_indexer = torch.zeros(list(self.background_data.shape[:-1]),
                    dtype=torch.bool, device=self.background_data.device)
            grad_holder.mask_background_out = self.sparse_background_indexer
        
        cu_fn = _C.__dict__["volume_render_cuvol_fused"]

        cu_fn(
            self._to_cpp(replace_basis_data=basis_data),
            rays._to_cpp(),
            self.opt._to_cpp(randomsize=randomize),
            rgb_gt,
            beta_loss,
            sparsity_loss,
            rgb_out,
            grad_holder
        )
        self.sparse_sh_grad_indexer = self.sparse_grad_indexer.clone()

        return rgb_out


    # use cuda c rendering function with depth supervion
    def training_opt_render_fused(
        self, 
        rays: Rays, 
        rgb_gt: torch.Tensor, 
        depth_gt: torch.Tensor,
        randomize:bool = False, 
        beta_loss: float = 0.0,
        sparsity_loss: float = 0.0,
        density_loss: float = 0.0,
        penalize_loss: float = 0.0):

        grad_sh, grad_density, grad_basis, grad_bg = self._get_data_grads()
        rgb_out = torch.zeros_like(rgb_gt)

        depth_out = torch.zeros_like(depth_gt)
        basis_data : Optional[torch.Tensor] = None

        self.sparse_grad_indexer = torch.zeros((self.density_data.size(0)), dtype=torch.bool, device=self.density_data.device)
        grad_holder = _C.GridOutputGrads()
        grad_holder.grad_density_out = grad_density
        grad_holder.grad_sh_out = grad_sh  
        grad_holder.mask_out = self.sparse_grad_indexer

        if self.use_background:
            grad_holder.grad_background_out = grad_bg
            self.sparse_background_indexer = torch.zeros(list(self.background_data.shape[:-1]),
                    dtype=torch.bool, device=self.background_data.device)
            grad_holder.mask_background_out = self.sparse_background_indexer
        
        cu_fn = _C.__dict__["volume_render_optimize_fused"]

        cu_fn(
            self._to_cpp(replace_basis_data=basis_data),
            rays._to_cpp(),
            self.opt._to_cpp(randomsize=randomize),
            rgb_gt,
            rgb_out,
            depth_gt,
            depth_out,
            beta_loss,
            sparsity_loss,
            density_loss,
            penalize_loss,
            grad_holder
        )
        # self.sparse_sh_grad_indexer = self.sparse_grad_indexer.clone()

        return rgb_out