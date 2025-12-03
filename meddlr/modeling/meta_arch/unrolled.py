from numbers import Number
from typing import Any, Dict, Sequence, Tuple, Union
import torch
import torchvision.utils as tv_utils
from torch import nn
import h5py
import numpy as np
from PIL import Image
import meddlr.ops.complex as cplx
from meddlr.config import CfgNode
from meddlr.config.config import configurable
from meddlr.forward.mri import SenseModel
from meddlr.modeling.meta_arch.resnet import ResNetModel
from meddlr.ops.opt import conjgrad
from meddlr.utils.events import get_event_storage
from meddlr.utils.general import move_to_device
import os
from .build import META_ARCH_REGISTRY, build_model
from matplotlib import pyplot as plt
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from sklearn.linear_model import LinearRegression
import time
from torch.func import vmap, jvp
from scipy.stats import wilcoxon, ttest_1samp, shapiro

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

__all__ = ["GeneralizedUnrolledCNN"]

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os


def measure_similarity(std_theoretical_np, std_empirical_np,  dir_path, slice_, variance_type='theoretical', method='', acceleration=1):
    # Flatten the arrays for comparison
    std_theoretical_flat = std_theoretical_np.flatten()
    std_empirical_flat = std_empirical_np.flatten()

    # Calculate the Pearson correlation coefficient using NumPy
    pearson_correlation = np.corrcoef(std_theoretical_flat, std_empirical_flat)[0, 1]

    # Calculate NRMSE, normalized root mean squared error
    nrmse = np.sqrt(np.mean(((std_theoretical_flat - std_empirical_flat))**2))/np.max(std_empirical_flat)

    # Return the specified metrics
    metrics = {
        'pearson_correlation': pearson_correlation,
        'nrmse': nrmse,
    }

    return metrics

def normalize_image_by_top_magnitude(image, percentile=0.05):
    """
    Normalizes the image by a given percentile of the magnitude values.

    Parameters:
    - image: np.ndarray, complex image data.
    - percentile: float, the percentile value to determine the scale factor (default 0.95).

    Returns:
    - normalized_image: np.ndarray, normalized image data.
    - mean: float, mean value used for normalization (always 0.0).
    - std: float, standard deviation value used for normalization.
    """
    # Calculate the magnitude values of the complex image
    magnitude_vals = np.abs(image).flatten()
    
    # Determine the scale based on the specified percentile
    k = int(round((1 - percentile) * magnitude_vals.size))
    scale = np.partition(magnitude_vals, k)[k]
    
    # Normalize the image
    normalized_image = image / scale
    
    # Set the mean to 0 and the std to the computed scale
    mean = 0.0
    std = scale
    
    return normalized_image, mean, std


@META_ARCH_REGISTRY.register()
class GeneralizedUnrolledCNN(nn.Module):
    """Unrolled compressed sensing model.

    This implementation is adapted from:
    https://github.com/MRSRL/dl-cs

    Reference:
        CM Sandino, JY Cheng, et al. "Compressed Sensing: From Research to
        Clinical Practice with Deep Neural Networks" IEEE Signal Processing
        Magazine, 2020.
    """

    @configurable
    def __init__(
        self,
        blocks: Union[nn.Module, Sequence[nn.Module]],
        step_sizes: Union[float, Sequence[float]] = -2.0,
        fix_step_size: bool = False,
        num_emaps: int = 1,
        vis_period: int = -1,
        noise_calculation: bool = False,
        calculate_variance: bool = False,
        variances_list: Sequence[str] = None,
        acc_rate: Sequence[int] = None,
        num_grad_steps: int = None,
        order: Tuple[str] = ( "dc","reg"),
        noise_level: float = 0.01,
        input_noise: float = 0.01,
        model_meta_arch: str = 'GeneralizedUnrolledCNN',
        variance_calculation_method: str = "monte_carlo",
        method_name: str = "E2E_VARNET",
        dataset_name: str = "knee_data",
        mask_name: str = "PoissonDiskMaskFunc",
    ):
        """
        Args:
            blocks: A sequence of blocks
            step_sizes: Step size for data consistency prior to each block.
                If a single float is given, the same step size is used for all blocks.
            fix_step_size: Whether to fix the step size to a given value --
                i.e. set to `True to make the step size non-trainable.
            num_emaps: Number of sensitivity maps used to estimate the image.
            vis_period: Number of steps between logging visualizations.
            num_grad_steps: Number of unrolled steps in the network.
                This is deprecated - the number of steps will be determined
                from the length of `blocks.
            order: The order to apply the data consistency (dc) and model-based
                regularization (reg) blocks. One of `('dc', 'reg') or
                `('reg', 'dc').
        """
        super().__init__()

        self.resnets = blocks
        if num_grad_steps is None:
            if isinstance(blocks, Sequence) and not isinstance(blocks, nn.ModuleList):
                blocks = nn.ModuleList(blocks)
            if not isinstance(blocks, nn.ModuleList):
                raise TypeError("blocks must be a sequence of nn.Modules or a nn.ModuleList")
            num_grad_steps = len(blocks)
            num_repeat_steps = 0
        else:
            if not isinstance(num_grad_steps, int) or num_grad_steps <= 0:
                raise ValueError("num_grad_steps must be positive integer")
            num_repeat_steps = num_grad_steps

        if isinstance(step_sizes, Number):
            step_sizes = [
                torch.tensor([step_sizes], dtype=torch.float32) for _ in range(num_grad_steps)
            ]
        else:
            if len(step_sizes) != num_grad_steps:
                raise ValueError(
                    "step_sizes must be a single value or a list of the "
                    "same length as blocks or num_grad_steps"
                )
            step_sizes = [torch.tensor(s) for s in step_sizes]
        if not fix_step_size:
            step_sizes = nn.ParameterList([nn.Parameter(s) for s in step_sizes])
        self.step_sizes: Sequence[Union[torch.Tensor, nn.Parameter]] = step_sizes


        self.num_repeat_steps = num_repeat_steps
        self.num_emaps = num_emaps
        self.vis_period = vis_period


        self.noise_calculation = noise_calculation
        self.calculate_variance = calculate_variance
        self.variances_list = variances_list
        self.acc_rate = acc_rate
        self.noise_level = noise_level
        self.input_noise = input_noise
        self.model_meta_arch = model_meta_arch
        self.variance_calculation_method = variance_calculation_method
        self.method_name = method_name
        self.mask_name = mask_name
        self.dataset_name = "knee_data" if "knee" in dataset_name else "brain_data"

        if "knee" in dataset_name:
            self.sigma_k = torch.load("datasets/Sigma_k.pt")
        else:
            self.sigma_k = torch.load("datasets/Sigma_k_brain.pt")

        # Lists/dicts to store metrics for each slice
        self.hutchinson_metrics_list = []
        self.monte_carlo_metrics_list = []

        if self.__class__.__name__ == 'CGUnrolledCNN':
            self.method_name = 'MODL'


        if order not in [("dc", "reg"), ("reg", "dc")]:
            raise ValueError("order must be one of ('dc', 'reg') or ('reg', 'dc')")
        self.order = order
        self._dc_first = order[0] == "dc"

    def visualize_training(
        self, kspace: torch.Tensor, zfs: torch.Tensor, targets: torch.Tensor, preds: torch.Tensor
    ):
        """Visualize kspace data and reconstructions.

        Dimension ``(,2)`` indicates optional dimension for real-valued view of complex tensors.
        For example, a real-valued tensor of shape BxHxWx2 will be interpreted as
        a complex-valued tensor of shape BxHxW.

        Args:
            kspace: The complex-valued kspace. Shape: [batch, height, width, #coils, (,2)].
            zfs: The complex-valued zero-filled images.
                Shape: [batch, height, width, (,2)].
            targets: The complex-valued target (reference) images.
                Shape: [batch, height, width, (,2)].
            preds: The complex-valued predicted images.
                Shape: [batch, height, width, (,2)].
        """
        storage = get_event_storage()

        with torch.no_grad():
            if cplx.is_complex(kspace):
                kspace = torch.view_as_real(kspace)
            kspace = kspace[0, ..., 0, :].unsqueeze(0).cpu()  # calc mask for first coil only
            targets = targets[0, ...].unsqueeze(0).cpu()
            preds = preds[0, ...].unsqueeze(0).cpu()
            zfs = zfs[0, ...].unsqueeze(0).cpu()

            all_images = torch.cat([zfs, preds, targets], dim=2)

            imgs_to_write = {
                "phases": cplx.angle(all_images),
                "images": cplx.abs(all_images),
                "errors": cplx.abs(preds - targets),
                "masks": cplx.get_mask(kspace),
            }

            for name, data in imgs_to_write.items():
                data = data.squeeze(-1).unsqueeze(1)
                data = tv_utils.make_grid(data, nrow=1, padding=1, normalize=True, scale_each=True)
                storage.put_image("train/{}".format(name), data.numpy(), data_format="CHW")

    def dc(
        self,
        *,
        image: torch.Tensor,
        A: SenseModel,
        zf_image: torch.Tensor,
        step_size: Union[torch.Tensor, float]
    ):
        grad_x = A(A(image), adjoint=True) - zf_image
        return image + step_size * grad_x

    def reg(self, *, image: torch.Tensor, model: nn.Module, dims: torch.Size):
        # If the image is a complex tensor, we view it as a real image
        # where last dimension has 2 channels (real, imaginary).
        # This may take more time, but is done for backwards compatibility
        # reasons.
        # TODO (arjundd): Fix to auto-detect which version of the model is being used.
        if dims is None:
            dims = image.size()

        use_cplx = cplx.is_complex(image)
        if use_cplx:
            image = torch.view_as_real(image)

        # prox update
        image = image.reshape(dims[0:3] + (self.num_emaps * 2,)).permute(0, 3, 1, 2)
        if hasattr(model, "base_forward") and callable(model.base_forward):
            image = model.base_forward(image)
        else:
            image = model(image)
        # This doesn't work when padding is not the same.
        # i.e. when the output is a different shape than the input.
        # However, this should not ever happen.
        image = image.permute(0, 2, 3, 1).reshape(dims[0:3] + (self.num_emaps, 2))
        if not image.is_contiguous():
            image = image.contiguous()
        if use_cplx:
            image = torch.view_as_complex(image)
        return image

    def step(
        self,
        *,
        image: torch.Tensor,
        model: nn.Module,
        A: SenseModel,
        zf_image: torch.Tensor,
        step_size: Union[torch.Tensor, float],
        dims: torch.Size):
        
        if self._dc_first:
            image = self.dc(image=image, A=A, zf_image=zf_image, step_size=step_size)
            image = self.reg(image=image, model=model, dims=dims)
        else:
            image = self.reg(image=image, model=model, dims=dims)
            image = self.dc(image=image, A=A, zf_image=zf_image, step_size=step_size)
        return image
    
    def model_forward(self,zf_image, A):
        dims = self.dims
        if self.num_repeat_steps > 0:
            conv_blocks = [self.resnets] * self.num_repeat_steps
        else:
            conv_blocks = self.resnets
        device = next(conv_blocks[0].parameters()).device
        step_sizes = [x.to(device) for x in self.step_sizes]
        image = zf_image
        for resnet, step_size in zip(conv_blocks, step_sizes):
            image = self.step(
                image=image,
                model=resnet,
                A=A,
                zf_image=zf_image,
                step_size=step_size,
                dims=dims,
            )
        x_hat = image
        return x_hat
    
    def autograd_variance(self,zf_image, A):
        start_time = time.time()
        with torch.enable_grad():
            zf_image.requires_grad_(True)

            # perform forward pass
            x_hat = self.model_forward(zf_image, A)
            
            # Initialize the variance tensor
            variance_reconstructed = torch.zeros_like(x_hat)

            for c in range(x_hat.shape[0]):  # Loop over channels
                for h in tqdm(range(x_hat.shape[1]), desc="Height", leave=False):  # Loop over height
                    for w in range(x_hat.shape[2]):  # Loop over width
                        # Zero out previous gradients
                        if zf_image.grad is not None:
                            zf_image.grad.zero_()
                        self.zero_grad()
                        row_jacobian_conjugated = torch.autograd.grad(
                            x_hat[c, h, w],
                            zf_image,
                            retain_graph=True,
                            create_graph=False,
                            allow_unused=True
                        )[0]  
                        # [kspace.shape], gradients with respect to kspace
                        variance_reconstructed[c, h, w] = torch.sum(torch.abs(row_jacobian_conjugated)** 2)
        end_time = time.time()
        time_taken = end_time - start_time
        return variance_reconstructed, time_taken

    def J_sketch_variance(self, kspace, A,sigma_k, S=100):
        device = kspace.device
        dtype = kspace.dtype
        self.dims = tuple(kspace.size())

        zf_image = A(kspace, adjoint=True)[0]

        # Define the reconstruction function
        def model_func(zf_image):
            return self.model_forward(zf_image, A)

        # Function to compute variance sample
        def compute_variance_sample(v):
            _, u = jvp(model_func, (zf_image,), (v,))
            variance_sample = torch.abs(u)**2
            return variance_sample
        start = time.time()
        # Generate all random phase vectors at once
        v_shape = (S,) + kspace.shape[1:]  # (100, 320, 256, nc)
        real_part = torch.randint(0, 2, v_shape, device=device).float() * 2 - 1  # +1 or -1
        imag_part = torch.randint(0, 2, v_shape, device=device).float() * 2 - 1  # +1 or -1

        # Combine to form a batch of complex Rademacher vectors
        v_batch = real_part + 1j * imag_part

        print(sigma_k.shape)

        v_batch = torch.matmul(v_batch, sigma_k)  # Shape remains: (100, 320, 256, nc)

        print(v_batch.shape)

        u_batch = A(v_batch, adjoint=True)  # Shape: (100, 320, 256, 1)

        # Vectorize over the batch dimension using vmap
        variance_samples = vmap(compute_variance_sample)(u_batch)

        # Average all variance samples
        variance_reconstructed = variance_samples.sum(dim=0) / S

        end = time.time()
        time_taken = end - start
        print(f"Time taken for Jacobian Sketching: {time_taken:.2f} seconds")

        return variance_reconstructed, time_taken

    def monte_carlo_variance(self, kspace, A, mask, num_samples=100):
        start = time.time()

        device = kspace.device
        # Prepare a list to store all reconstructions
        reconstructions = []
        noise_std = self.input_noise

        for _ in tqdm(range(num_samples), desc="Monte Carlo Samples"):
            # Generate complex Gaussian noise
            real_noise = torch.randn(kspace.shape, device=device) * noise_std
            imag_noise = torch.randn(kspace.shape, device=device) * noise_std
            noise = real_noise + 1j * imag_noise
            noise = noise.to(kspace.dtype)

            # Apply mask to noise to add noise only at sampled k-space points
            noise_masked = noise * mask

            # Add masked noise to kspace
            kspace_noisy = kspace + noise_masked

            zf_noisy_image = A(kspace_noisy, adjoint=True)

            # Reconstruct image with noisy kspace
            x_hat_noisy = self.model_forward(zf_noisy_image, A)

            # Store the reconstruction
            reconstructions.append(x_hat_noisy.unsqueeze(0))  # Add batch dimension

        
        # Stack all reconstructions into a single tensor
        reconstructions = torch.cat(reconstructions, dim=0)  # Shape: [num_samples, *image_shape]

        # Compute the standard deviation across the samples
        std_reconstructed = torch.std(reconstructions, dim=0)#, unbiased=True)

        # Compute variance from standard deviation
        variance_reconstructed = std_reconstructed ** 2

        end = time.time()
        time_taken = end - start
        print(f"Time taken for Monte Carlo: {end - start:.2f} seconds")
        return variance_reconstructed, time_taken

    def forward(self, inputs: Dict[str, Any], return_pp: bool = False, vis_training: bool = False,idx=0):
        """Reconstructs the image from the kspace.

        Dimension ``(,2)`` indicates optional dimension for real-valued view of complex tensors.
        For example, a real-valued tensor of shape BxHxWx2 will be interpreted as
        a complex-valued tensor of shape BxHxW.

        ``#maps`` refers to the number of sensitivity maps used to estimate the image
        (i.e. ``self.num_emaps``).ou

        Args:
            inputs: Standard meddlr module input dictionary
                * "kspace": The kspace (typically undersampled).
                  Shape: [batch, height, width, #coils, (,2)].
                * "maps": The sensitivity maps used for SENSE coil combination.
                  Shape: [batch, height, width, #coils, #maps, (,2)].
                * "target" (optional): Target (reference) image.
                  Shape: [batch, height, width, #maps, (,2)].
                * "signal_model" (optional): The signal model. If provided,
                    "maps" will not be used to estimate the signal model.
                    Use with caution.
            return_pp (bool, optional): If `True`, return post-processing
                parameters "mean", "std", and "norm" if included in the input.
            vis_training (bool, optional): If `True`, force visualize training
                on this pass. Can only be `True` if model is in training mode.

        Returns:
            Dict: A standard meddlr output dict
                * "pred": The reconstructed image
                * "target" (optional): The target image.
                    Added if provided in the input.
                * "mean"/"std"/"norm" (optional): Pre-processing parameters.
                    Added if provided in the input.
                * "zf_image": The zero-filled image.
                    Added when model is in eval mode.
        """
        if self.num_repeat_steps > 0:
            conv_blocks = [self.resnets] * self.num_repeat_steps
        else:
            conv_blocks = self.resnets

        if vis_training and not self.training:
            raise ValueError("vis_training is only applicable in training mode.")
        # Need to fetch device at runtime for proper data transfer.
        device = next(conv_blocks[0].parameters()).device
        inputs = move_to_device(inputs, device)
        kspace = inputs["kspace"]
        noise = inputs.get("noise", None)
        target = inputs.get("target", None)
        mask = inputs.get("mask", None)
        A = inputs.get("signal_model", None)
        maps = inputs["maps"]
        num_maps_dim = -2 if cplx.is_complex_as_real(maps) else -1
        if self.num_emaps != maps.size()[num_maps_dim] and maps.size()[num_maps_dim] != 1:
            raise ValueError("Incorrect number of ESPIRiT maps! Re-prep data...")
        
        # Move step sizes to the right device.
        step_sizes = [x.to(device) for x in self.step_sizes]
        # print(f"step sizes: {step_sizes}")
        if mask is None:
            mask = cplx.get_mask(kspace)
        # Get data dimensions
        dims = tuple(kspace.size())
        self.dims = dims    
        # Declare signal model.
        if A is None:
            A = SenseModel(maps, weights=mask)

        kspace = mask * kspace
        # print(f"kspace shape: {kspace.shape}")

        if self.calculate_variance and idx in self.variances_list:
            S = 100
            num_samples_ref = 300
            variance_save_dir_png = f'calculated_variances_png/{self.dataset_name}/{self.method_name}/{self.mask_name}/{self.acc_rate[0]}/{self.input_noise}/{idx}'
            print(f"Calculating variance for slice {idx} with {S} samples")
            print(f"Saving to {variance_save_dir_png}")

            os.makedirs(variance_save_dir_png, exist_ok=True)

            #reconstruct the noisy zfimage
            # Generate complex Gaussian noise
            real_noise = torch.randn(kspace.shape, device=device) * self.input_noise
            imag_noise = torch.randn(kspace.shape, device=device) * self.input_noise
            noise = real_noise + 1j * imag_noise
            noise = noise.to(kspace.dtype)

            # Apply mask to noise to add noise only at sampled k-space points
            noise_masked = noise * mask

            # Add masked noise to kspace
            kspace_noisy = kspace + noise_masked
            zf_image = A(kspace_noisy, adjoint=True)
            # zf_image = A(kspace, adjoint=True)
            #mean of zero-filled
            mean_zf = zf_image.abs().mean()
            print(f"Mean of zero-filled image: {mean_zf}")
            #top magnitude of zero-filled

            
            image = self.model_forward(zf_image, A)
            #save the reconstructed image
            full_image_np = np.abs(image.cpu().numpy()[0, :, :, 0])
            plt.imsave(f'{variance_save_dir_png}/reconstructed_image.png', full_image_np, cmap='gray', vmin=full_image_np.min(), vmax=full_image_np.max())

            #save the zero-filled image
            zf_image_np = np.abs(zf_image.abs().cpu().numpy()[0, :, :, 0])
            top_mag_zf = normalize_image_by_top_magnitude(zf_image_np,percentile=0.1)[2]
            print(f"Top magnitude of zero-filled image: {top_mag_zf}")
            plt.imsave(f'{variance_save_dir_png}/zf_image.png', zf_image_np, cmap='gray', vmin=zf_image_np.min(), vmax=zf_image_np.max())
            #max of zero-filled
            max_zf = zf_image.abs().max()
            print(f"Max of zero-filled image: {max_zf}")
            #save the mask
            print(f"Mask shape: {mask.shape}")
            mask_np = mask.cpu().numpy()[0, :, :, 0]
            plt.imsave(f'{variance_save_dir_png}/mask.png', mask_np, cmap='gray') #, vmin=mask_np.min(), vmax=mask_np.max())


            #save the target image
            target_np = np.abs(target.cpu().numpy()[0, :, :, 0])
            plt.imsave(f'{variance_save_dir_png}/target.png', target_np, cmap='gray', vmin=target_np.min(), vmax=target_np.max())

            # Compute the reference variance map using a large number of Monte Carlo samples
            ref_variance_reconstructed_monte,ref_variance_time = self.monte_carlo_variance(kspace, A, mask, num_samples=num_samples_ref)
            std_reconstructed_ref_np = np.sqrt(ref_variance_reconstructed_monte.cpu().numpy().squeeze())
            # variance_reconstructed_hutch = None
            if "J_sketch" in self.variance_calculation_method:
                print(f"Now computing Jacobian Sketching variance map with num_samples = {S}")
                variance_reconstructed_J_sketch, j_sketch_time = self.J_sketch_variance(
                    kspace, A,self.sigma_k, S=S
                )

                variance_reconstructed_J_sketch_np = variance_reconstructed_J_sketch.cpu().numpy().squeeze()
                std_reconstructed_J_sketch_np = np.sqrt(variance_reconstructed_J_sketch_np)

                # Compare with reference (empirical) standard deviation
                metrics = measure_similarity(
                    std_theoretical_np=std_reconstructed_J_sketch_np,
                    std_empirical_np=std_reconstructed_ref_np,
                    dir_path=variance_save_dir_png,
                    slice_=idx,
                    variance_type='J_sketch',
                    method=f'J_sketch_ns_{S}',
                )

                diff = np.abs(std_reconstructed_J_sketch_np - std_reconstructed_ref_np)

                # Report specified metrics               
                print(f"Time taken for Jacobian Sketching: {j_sketch_time:.2f} seconds")
                print(f"Spatial Correlation (%): {metrics['pearson_correlation'] * 100:.2f}")
                print(f"NRMSE (×10⁻²): {metrics['nrmse'] * 100:.2f}")
                                
                top_mag = normalize_image_by_top_magnitude(std_reconstructed_J_sketch_np, percentile=0.005)[2]
                # Just save the variance map
                plt.imsave(
                    f'{variance_save_dir_png}/reference_n_{num_samples_ref}.png',
                    std_reconstructed_ref_np,
                    cmap='jet',
                    vmin=np.min(std_reconstructed_J_sketch_np),
                    vmax=top_mag
                )
                plt.imsave(
                    f'{variance_save_dir_png}/J_sketch_n_{S}.png',
                    std_reconstructed_J_sketch_np,
                    cmap='jet',
                    vmin=np.min(std_reconstructed_J_sketch_np),
                    vmax=top_mag
                )
                # Just save the difference map
                plt.imsave(
                    f'{variance_save_dir_png}/J_sketch_diff_n_{S}.png',
                    diff,
                    cmap='jet',
                    vmax=top_mag
                )
                # Amplified difference map
                plt.imsave(
                    f'{variance_save_dir_png}/J_sketch_diff_amplified_n_{S}.png',
                    diff * 10,
                    cmap='jet',
                    vmax=top_mag
                )


            # Compute variance using the naive method
            if "naive_method" in self.variance_calculation_method:
                print(f"Now computing Naive Method variance map")
                variance_reconstructed_naive, naive_time = self.autograd_variance(zf_image, A)
                
                variance_reconstructed_naive_np = variance_reconstructed_naive.cpu().numpy().squeeze()
                std_reconstructed_naive_np = np.sqrt(variance_reconstructed_naive_np)

                # Compare with reference (empirical) standard deviation
                metrics_naive = measure_similarity(
                    std_theoretical_np=std_reconstructed_naive_np,
                    std_empirical_np=std_reconstructed_ref_np,
                    dir_path=variance_save_dir_png,
                    slice_=idx,
                    variance_type='Naive Method',
                    method=f'naive',
                    acceleration=1  # Update if necessary
                )

                self.naive_metrics_list.append(metrics_naive)
                diff_naive = np.abs(std_reconstructed_naive_np - std_reconstructed_ref_np)

                # Report specified metrics               
                print(f"Metrics for Naive Method Variance Map")                
                print(f"Time taken for Naive Method: {naive_time:.2f} seconds")
                # Display window
                print(f"Display window [min,max]: [{np.min(std_reconstructed_naive_np)}, {np.max(std_reconstructed_naive_np)}]\n")
                print(f"Spatial Correlation (%): {metrics_naive['pearson_correlation'] * 100:.2f}")
                print(f"NRMSE (×10⁻²): {metrics_naive['nrmse'] * 100:.2f}")
                
                top_mag_naive = self.normalize_image_by_top_magnitude(std_reconstructed_naive_np, percentile=0.005)[2]
                # Just save the variance map
                plt.imsave(f'{variance_save_dir_png}/naive_n.png', std_reconstructed_naive_np, cmap='jet', vmin=np.min(std_reconstructed_naive_np), vmax=top_mag_naive)
                # Just save the difference map
                plt.imsave(f'{variance_save_dir_png}/naive_diff_n.png', diff_naive, cmap='jet', vmax=top_mag_naive)
                # Amplified difference map
                plt.imsave(f'{variance_save_dir_png}/naive_diff_amplified_n.png', diff_naive * 10, cmap='jet', vmax=top_mag_naive)

            
        else:
            # print("Not calculating pixel variances")
            zf_image = A(kspace, adjoint=True)
            image = zf_image
            for resnet, step_size in zip(conv_blocks, step_sizes):
                image = self.step(
                    image=image,
                    model=resnet,
                    A=A,
                    zf_image=zf_image,
                    step_size=step_size,
                    dims=dims,
                )


        # pred: shape [batch, height, width, #maps, 2]
        # target: shape [batch, height, width, #maps, 2]
        output_dict = {
            "pred": image,
            "target": target,
            "noise": noise,
            "signal_model": A,
        }


        if return_pp:
            output_dict.update({k: inputs[k] for k in ["mean", "std", "norm"]})

        if self.training and (vis_training or self.vis_period > 0):
            storage = get_event_storage()
            if vis_training or storage.iter % self.vis_period == 0:
                self.visualize_training(kspace, zf_image, target, image)

        output_dict["zf_image"] = zf_image

        return output_dict

    @classmethod
    def from_config(cls, cfg: CfgNode, **kwargs) -> Dict[str, Any]:
        """Build :cls:`GeneralizedUnrolledCNN` from a config.

        Args:
            cfg: The config.
            kwargs: Keyword arguments to override config-specified parameters.

        Returns:
            Dict[str, Any]: The parameters to pass to the constructor.
        """
        # Extract network parameters
        num_grad_steps = cfg.MODEL.UNROLLED.NUM_UNROLLED_STEPS
        share_weights = cfg.MODEL.UNROLLED.SHARE_WEIGHTS

        # Data dimensions
        num_emaps = cfg.MODEL.UNROLLED.NUM_EMAPS

        # Determine block to use for each unrolled step.
        if cfg.MODEL.UNROLLED.BLOCK_ARCHITECTURE == "ResNet":
            builder = lambda: _build_resblock(cfg)  # noqa: E731
        else:
            # TODO: Fix any inconsistencies between config's IN_CHANNELS
            # and the number of channels that the unrolled net expects.
            mcfg = cfg.clone().defrost()
            mcfg.MODEL.META_ARCHITECTURE = cfg.MODEL.UNROLLED.BLOCK_ARCHITECTURE
            mcfg = mcfg.freeze()
            builder = lambda: build_model(mcfg)  # noqa: E731

        # Declare ResNets and RNNs for each unrolled iteration
        if share_weights:
            blocks = builder()
        else:
            blocks = nn.ModuleList([builder() for _ in range(num_grad_steps)])

        # Step sizes
        step_sizes = cfg.MODEL.UNROLLED.STEP_SIZES
        if len(step_sizes) == 1:
            step_sizes = step_sizes[0]

        out = {
            "blocks": blocks,
            "step_sizes": step_sizes,
            "fix_step_size": cfg.MODEL.UNROLLED.FIX_STEP_SIZE,
            "num_emaps": num_emaps,
            "vis_period": cfg.VIS_PERIOD,
            "noise_calculation": cfg.TEST.CALCULATE_NOISE,
            "calculate_variance": cfg.TEST.CALCULATE_PIXEL_VARIANCES,
            "variances_list": cfg.TEST.VARIANCES_LIST,
            "num_grad_steps": num_grad_steps if share_weights else None,
            "acc_rate": cfg.AUG_TEST.UNDERSAMPLE.ACCELERATIONS,
            "noise_level": cfg.TEST.INPUT_NOISE_STD,
            "input_noise": cfg.TEST.INPUT_NOISE_STD,
            "model_meta_arch": cfg.MODEL.META_ARCHITECTURE,
            "variance_calculation_method": cfg.TEST.VARIANCE_CALCULATION_METHOD,
            "dataset_name": cfg.DATASETS.TEST[0],
            "mask_name": cfg.AUG_TRAIN.UNDERSAMPLE.NAME
        }
        out.update(kwargs)
        return out


@META_ARCH_REGISTRY.register()
class CGUnrolledCNN(GeneralizedUnrolledCNN):
    """Unrolled CNN with conjugate gradient descent (CG) data consistency.

    Identical to MoDL.
    """

    @configurable
    def __init__(
        self,
        blocks: Union[nn.Module, Sequence[nn.Module]],
        step_sizes: Union[float, Sequence[float]] = -2,
        fix_step_size: bool = False,
        num_emaps: int = 1,
        vis_period: int = -1,
        num_grad_steps: int = None,
        cg_max_iter: int = 10,
        cg_eps: float = 1e-4,
        cg_init: Literal["zeros", "reg"] = None,
        noise_calculation: bool = False,
        calculate_variance: bool = False,
        variances_list: Sequence[str] = None,
        acc_rate: Sequence[int] = None,
        method_name: str = 'MODL',
        noise_level: float = 0.01,
        input_noise: float = 0.01,
        model_meta_arch: str = 'CGUnrolledCNN',
        variance_calculation_method: str = "monte_carlo",
        dataset_name: str = "knee_data",
        mask_name: str = "PoissonDiskMaskFunc",


    ):
        super().__init__(
            blocks=blocks,
            step_sizes=step_sizes,
            fix_step_size=fix_step_size,
            num_emaps=num_emaps,
            vis_period=vis_period,
            num_grad_steps=num_grad_steps,
            order=("dc", "reg"),
            noise_calculation=noise_calculation,
            calculate_variance=calculate_variance,
            variances_list=variances_list,
            noise_level=noise_level,

            input_noise=input_noise,
            model_meta_arch=model_meta_arch,
            variance_calculation_method=variance_calculation_method,
            acc_rate=acc_rate,
            dataset_name=dataset_name,
            mask_name=mask_name,

        )
        self.cg_max_iter = cg_max_iter
        self.cg_eps = cg_eps
        self.cg_init = cg_init

        for step_size in self.step_sizes:
            if step_size < 0:
                raise ValueError("Step size must be non-negative.")

    def dc(
        self,
        *,
        image: torch.Tensor,
        A: SenseModel,
        zf_image: torch.Tensor,
        step_size: Union[torch.Tensor, float]
    ):
        def A_op(x):
            return A(A(x), adjoint=True)

        x_opt = conjgrad(
            x=image,
            b=zf_image + step_size * image,
            A_op=A_op,
            mu=step_size,
            max_iter=self.cg_max_iter,
            pbar=False,
            eps=self.cg_eps,
        )
        return x_opt

    def step(
        self,
        *,
        image: torch.Tensor,
        model: nn.Module,
        A: SenseModel,
        zf_image: torch.Tensor,
        step_size: Union[torch.Tensor, float],
        dims: torch.Size,
    ):
        def A_op(x):
            return A(A(x), adjoint=True)

        x_reg = self.reg(image=image, model=model, dims=dims)

        cg_init = image
        if self.cg_init == "zeros":
            cg_init = torch.zeros_like(image)
        elif self.cg_init == "reg":
            cg_init = x_reg

        x_opt = conjgrad(
            x=cg_init,
            b=zf_image + step_size * x_reg,
            A_op=A_op,
            mu=step_size,
            max_iter=self.cg_max_iter,
            pbar=False,
            eps=self.cg_eps,
        )
        return x_opt

    @classmethod
    def from_config(cls, cfg: CfgNode, **kwargs) -> Dict[str, Any]:
        """Build :cls:`CGUnrolledCNN` from a config.

        Args:
            cfg: The config.
            kwargs: Keyword arguments to override config-specified parameters.

        Returns:
            Dict[str, Any]: The parameters to pass to the constructor.
        """
        init_kwargs = super().from_config(cfg=cfg, **kwargs)
        init_kwargs["cg_max_iter"] = cfg.MODEL.UNROLLED.DC.MAX_ITER
        init_kwargs["cg_eps"] = cfg.MODEL.UNROLLED.DC.EPS
        #calculation of pixel variances
        # init_kwargs["calculate_variance"] = cfg.TEST.CALCULATE_PIXEL_VARIANCES
        # init_kwargs["variances_list"] = cfg.TEST.VARIANCES_LIST

        init_kwargs.update(kwargs)
        return init_kwargs


def _build_resblock(cfg: CfgNode) -> ResNetModel:
    """Build the resblock for unrolled network.

    Args:
        cfg (CfgNode): The network configuration.

    Note:
        This is a temporary method used as a base case for building
        unrolled networks with the default resblocks. In the future,
        this will be handled by :func:meddlr.modeling.meta_arch.build_model.
    """
    # Data dimensions
    num_emaps = cfg.MODEL.UNROLLED.NUM_EMAPS

    # ResNet parameters
    kernel_size = cfg.MODEL.UNROLLED.KERNEL_SIZE
    if len(kernel_size) == 1:
        kernel_size = kernel_size[0]
    resnet_params = dict(
        num_blocks=cfg.MODEL.UNROLLED.NUM_RESBLOCKS,
        in_channels=2 * num_emaps,  # complex -> real/imag
        channels=cfg.MODEL.UNROLLED.NUM_FEATURES,
        kernel_size=kernel_size,
        dropout=cfg.MODEL.UNROLLED.DROPOUT,
        circular_pad=cfg.MODEL.UNROLLED.PADDING == "circular",
        act_type=cfg.MODEL.UNROLLED.CONV_BLOCK.ACTIVATION,
        norm_type=cfg.MODEL.UNROLLED.CONV_BLOCK.NORM,
        norm_affine=cfg.MODEL.UNROLLED.CONV_BLOCK.NORM_AFFINE,
        order=cfg.MODEL.UNROLLED.CONV_BLOCK.ORDER,
    )

    return ResNetModel(**resnet_params)
