import torch
from torch import nn
import torch.nn.functional as F
import meddlr.ops as oF
import meddlr.ops.complex as cplx
import numpy as np
__all__ = ["SenseModel"]


# class SenseModel(nn.Module):
#     """
#     A module that computes forward and adjoint SENSE operations.

#     The forward operation converts a complex image -> multi-coil kspace.
#     The adjoint operation converts multi-coil kspace -> a complex image.

#     This module also supports multiple sensitivity maps. This is useful if
#     you would like to generate images from multiple estimated sensitivity maps.
#     This module also works with single coil inputs as long as the #coils dimension
#     is set to 1.

#     Attributes:
#         maps (torch.Tensor): Sensitivity maps. Shape ``(B, H, W, #coils, #maps, [2])``.
#         weights (torch.Tensor, optional): Undersampling masks (if applicable).
#             Shape ``(B, H, W)`` or ``(B, H, W, #coils, #coils)``.
#     """

#     def __init__(self, maps: torch.Tensor, weights: torch.Tensor = None):
#         """
#         Args:
#             maps (torch.Tensor): Sensitivity maps.
#             weights (torch.Tensor): Undersampling masks.
#                 If ``None``, it is assumed that inputs are fully-sampled.
#         """
#         super().__init__()

#         self.maps = maps
#         if weights is None:
#             self.weights = 1.0
#         else:
#             self.weights = weights

#     def _adjoint_op(self, kspace):
#         """
#         Args:
#             kspace: Shape (B,H,W,#coils), complex.
#         Returns:
#             image: Shape (B,H,W,#maps), complex.
#         """
#         # print("self.weights shape:", self.weights.shape)
#         # print("kspace shape:", kspace.shape)
#         # print("self.maps shape:", self.maps.shape)
#         image = oF.ifft2c(self.weights * kspace, channels_last=True)
#         if cplx.is_complex_as_real(kspace):
            
#             image = cplx.mul(image.unsqueeze(-2), cplx.conj(self.maps))  # [B,...,#coils,#maps,2]
#             return image.sum(-3)
#         else:
#             # This is a hacky solution managing multi-channel inputs.
#             # Note multi-channel inputs are only supported in complex tensors.
#             # TODO (arjundd, issue #18): Fix with tensor ordering.
#             if image.ndim != self.maps.ndim:
#                 image = image.unsqueeze(-1)
#             # print(image.shape)
#             image = cplx.mul(image, cplx.conj(self.maps))  # [B,...,#coils,#maps,1]
#             return image.sum(-2)
#     def _forward_op(self, image):
#         """
#         Args:
#             image: Shape (B,H,W,#maps,[2])
#         Returns:
#             kspace: Shape (B,H,W,#coils,[2])
#         """
        
#         if cplx.is_complex_as_real(image):
#             print(image.unsqueeze(-3).shape)
#             # print(self.maps.shape)
#             kspace = cplx.mul(image.unsqueeze(-3), self.maps)  # [B,...,1,#maps,2]
#             kspace = self.weights * oF.fft2c(kspace.sum(-2), channels_last=True)  # [B,...,#coils,2]
#         else:
#             kspace = cplx.mul(image.unsqueeze(-2), self.maps)
#             # This is a hacky solution managing multi-channel inputs.
#             # Note this change means that multiple maps with multi-channel inputs
#             # is not supported for forward operations. This will change in future updates.
#             # TODO (arjundd, issue #18): Fix with tensor ordering.
#             if image.shape[-1] == self.maps.shape[-1]:
#                 kspace = kspace.sum(-1)
#             kspace = self.weights * oF.fft2c(kspace, channels_last=True)
#         return kspace

#     def forward(self, input: torch.Tensor, adjoint: bool = False):
#         """Run forward or adjoint SENSE operation on the input.

#         Depending on if ``adjoint=True``, the input should either be the
#         k-space or the complex image. The shapes for these are as follows:
#             - kspace: ``(B, H, W, #coils, [2])
#             - image: ``(B, H, W, #maps, [2])``

#         Args:
#             input (torch.Tensor): If ``adjoint=True``, this is the multi-coil k-space,
#                 else it is the image.
#             adjoint (bool, optional): If ``True``, use adjoint operation.

#         Returns:
#             torch.Tensor: If ``adjoint=True``, the image, else multi-coil k-space.
#         """
#         if adjoint:
#             output = self._adjoint_op(input)
#         else:
#             output = self._forward_op(input)
#         return output


#Trying to correct the sense model
class SenseModel(nn.Module):
    """
    A module that computes forward and adjoint SENSE operations using an external SENSE function.
    """

    def __init__(self, maps: torch.Tensor, weights: torch.Tensor = None, noise_cov: torch.Tensor = None, acceleration_rate: float = 2):
        """
        Args:
            maps (torch.Tensor): Sensitivity maps with shape (B,  H, W,#coils,1).
            weights (torch.Tensor): Undersampling masks with shape (B, H, W).
            noise_cov (torch.Tensor): Noise covariance matrix with shape (#coils, #coils).
        """
        super().__init__()
        self.maps = maps
        self.noise_cov = noise_cov
        self.acceleration_rate = acceleration_rate

        if weights is None:
            # self.weights = None
            # self.weights = 1.0 
            self.weights = torch.ones((maps.size(0), maps.size(1), maps.size(2), maps.size(3)), dtype=torch.float32,device=self.maps.device)
        else:
            self.weights = weights

    def _adjoint_op_zf(self, kspace):
        """
        Args:
            kspace: Shape (B,H,W,#coils), complex.
        Returns:
            image: Shape (B,H,W,#maps), complex.
        """
        # print("zero filled adjoint")
        image = oF.ifft2c(self.weights * kspace, channels_last=True)
        if cplx.is_complex_as_real(kspace):
            
            image = cplx.mul(image.unsqueeze(-2), cplx.conj(self.maps))  # [B,...,#coils,#maps,2]
            return image.sum(-3)
        else:
            # This is a hacky solution managing multi-channel inputs.
            # Note multi-channel inputs are only supported in complex tensors.
            # TODO (arjundd, issue #18): Fix with tensor ordering.
            if image.ndim != self.maps.ndim:
                image = image.unsqueeze(-1)
            # print(image.shape)
            image = cplx.mul(image, cplx.conj(self.maps))  # [B,...,#coils,#maps,1]
            return image.sum(-2)
    # def calc_spatial_variance_map(self, kspace, noise_std=1.0):
    #     """
    #     Calculate the spatial variance map of the ZF image.

    #     Args:
    #         kspace: Shape (B,H,W,#coils), complex.
    #         noise_std: Standard deviation of the Gaussian noise to be added.

    #     Returns:
    #         variance_map: Spatial variance map, Shape (B,H,W).
    #     """
    #     # Ensure kspace, maps, and mask are on the same device
    #     device = kspace.device

    #     # Compute the inverse Fourier transform of the undersampling mask
    #     F_inv_mask = oF.ifft2c(self.weights)

    #     # Compute the variance map for each coil
    #     coil_variance_map = noise_std**2 * torch.abs(F_inv_mask)**2

    #     # Compute the squared magnitude of the sensitivity maps
    #     sensitivity_maps_squared = torch.abs(self.maps)**2

    #     # Prepare the convolution filter (PSF)
    #     psf = coil_variance_map.unsqueeze(1).unsqueeze(1)  # Shape (B,1,1,H,W)

    #     # Convolve PSF with squared sensitivity maps
    #     variance_maps = []
    #     for b in range(kspace.shape[0]):  # Iterate over batch dimension
    #         variance_map = torch.zeros_like(sensitivity_maps_squared[b,:,:,0])
    #         for c in range(sensitivity_maps_squared.shape[-1]):  # Iterate over coils
    #             variance_map += F.conv2d(
    #                 sensitivity_maps_squared[b:b+1, :, :, c:c+1].unsqueeze(0), 
    #                 psf[b:b+1], padding='same'
    #             ).squeeze()
    #         variance_maps.append(variance_map)
    #     combined_variance_map = torch.stack(variance_maps, dim=0)

    #     return combined_variance_map

    def calc_spatial_variance_map(self):
        """
        Calculate the spatial variance map of the ZF image.
        
        Args:
            kspace: Shape (B,H,W,#coils), complex.
            maps: Coil sensitivity maps, Shape (H,W,#coils,#maps), complex.
            mask: Undersampling mask, same shape as kspace.
            noise_std: Standard deviation of the Gaussian noise to be added.
            
        Returns:
            variance_map: Spatial variance map, Shape (B,H,W).
        """
        # Ensure kspace, maps, and mask are on the same device
        # device = kspace.device

        noise_std = 1

        # Compute the inverse Fourier transform of the undersampling mask
        # F_inv_mask = oF.ifft2c(self.weights,channels_last=True)
        
        # # Compute the variance map for each coil
        # coil_variance_map = noise_std**2 * torch.abs(F_inv_mask)**2

        # Combine the variance maps using the coil sensitivity maps
        combined_variance_map = (torch.abs(self.maps)**2)#.sum(dim=-2)

        # print("combined_variance_map shape:", combined_variance_map.shape)


        #take the FT of the combined variance map
        combined_variance_map = oF.fft2c(combined_variance_map,channels_last=True)

        print("combined_variance_map shape:", combined_variance_map.shape)
        print("self.weights shape:", self.weights.shape)
        #apply the undersampling mask
        combined_variance_map = self.weights.unsqueeze(-1) * combined_variance_map

        #take the inverse FT
        combined_variance_map = oF.ifft2c(combined_variance_map,channels_last=True)

        print("combined_variance_map shape:", combined_variance_map.shape)

        return combined_variance_map.sum(dim=-2).squeeze(-1)
    
    # def compute_covariance_diagonal(self):
    #     """
    #     Compute the diagonal entries of the covariance matrix for the noise in the image space.
        
    #     Returns:
    #         cov_diag: Diagonal entries of the covariance matrix.
    #     """
    #     B, H, W, C, _ = self.maps.shape
    #     cov_diag = torch.zeros((B, H, W), dtype=torch.float32, device=self.maps.device)
    #     # print(cov_diag.shape)
    #     for c in range(C):
    #         S_c = self.maps[..., c, 0]  # Sensitivity map for the c-th coil
    #         S_c_H_img_space = S_c # torch.conj(S_c)  # Hermitian transpose of the sensitivity map

    #         # # Step 1: Transform coil sensitivity to k-space
    #         # S_c_H_kspace = oF.fft2c(S_c_H, channels_last=True)  # [B, H, W]

    #         # # Step 2: Apply the undersampling mask
    #         # S_c_H_kspace_masked = self.weights[..., c]* S_c_H_kspace  # Apply mask

    #         # # Step 3: Transform back to image space
    #         # S_c_H_img_space = oF.ifft2c(S_c_H_kspace_masked, channels_last=True)  # [B, H, W]

    #         # Step 4: Multiply by the coil sensitivity and accumulate the contributions
    #         # temp = S_c * torch.conj(S_c_H_img_space)  # [B, H, W]
    #         # print(temp.shape)
    #         # Compute the contribution to the diagonal entries
    #         contribution_c = torch.abs(S_c_H_img_space**2)#.sum(dim=-1)  # Sum along the complex dimension

    #         # Accumulate the contributions
    #         cov_diag += contribution_c

    #     return cov_diag
    
    def _forward_op(self, image):
        """
        Args:
            image: Shape (B,H,W,#maps,[2])
        Returns:
            kspace: Shape (B,H,W,#coils,[2])
        """
        
        if cplx.is_complex_as_real(image):
            print(image.unsqueeze(-3).shape)
            # print(self.maps.shape)
            kspace = cplx.mul(image.unsqueeze(-3), self.maps)  # [B,...,1,#maps,2]
            kspace = self.weights * oF.fft2c(kspace.sum(-2), channels_last=True)  # [B,...,#coils,2]
        else:
            kspace = cplx.mul(image.unsqueeze(-2), self.maps)
            # This is a hacky solution managing multi-channel inputs.
            # Note this change means that multiple maps with multi-channel inputs
            # is not supported for forward operations. This will change in future updates.
            # TODO (arjundd, issue #18): Fix with tensor ordering.
            if image.shape[-1] == self.maps.shape[-1]:
                kspace = kspace.sum(-1)
            # print("kspace shape:", kspace.shape)
            if self.weights is not None:
                # kspace = self.weights.unsqueeze(-1) * kspace
                kspace = self.weights * oF.fft2c(kspace, channels_last=True)
            else:
                print("No weights")
                kspace = oF.fft2c(kspace, channels_last=True)
        return kspace

    def forward(self, input: torch.Tensor, adjoint: bool = False, zero_filled=True):
        """Run forward or adjoint SENSE operation on the input.

        Depending on if ``adjoint=True``, the input should either be the
        k-space or the complex image. The shapes for these are as follows:
            - kspace: ``(B, H, W, #coils, [2])
            - image: ``(B, H, W, #maps, [2])``

        Args:
            input (torch.Tensor): If ``adjoint=True``, this is the multi-coil k-space,
                else it is the image.
            adjoint (bool, optional): If ``True``, use adjoint operation.

        Returns:
            torch.Tensor: If ``adjoint=True``, the image, else multi-coil k-space.
        """
        if adjoint:
            if zero_filled:
                output = self._adjoint_op_zf(input)
            else:
                output = self._adjoint_op(input)
        else:
            output = self._forward_op(input)
        return output

    def _adjoint_op(self, kspace):
        """
        Apply the adjoint SENSE operation to reconstruct the image from k-space.
        Args:
            kspace: Shape (B, H, W, #coils), complex.
        Returns:
            image: Shape (B, H, W), complex.
        """
        #bring back to image space
        kspace = oF.ifft2c(kspace, channels_last=True)


        # print('maps shape:', self.maps.shape)
        if self.noise_cov is None:
            # Assume an identity matrix if noise covariance is not provided
            self.noise_cov = torch.eye(self.maps.size(-2), dtype=kspace.dtype, device=kspace.device)
            #for correlated noise

        # Ensure the sensitivity maps are complex as expected by the sense function
        if not torch.is_complex(self.maps):
            # print("Converting sensitivity maps to complex")
            self.maps = torch.view_as_complex(self.maps)

        #permute k space to match the expected shape, (B, #coils, 1, H, W)
        kspace = kspace.permute(0, 3, 1, 2)  # Shape (B, #coils, H, W)
        # The sense function expects data with an extra temporal or frequency dimension,
        # which we don't have, so we need to add a singleton dimension
        kspace = kspace.unsqueeze(2)  # Shape (B, #coils, 1, H, W)
        # print("kspace shape:", kspace.shape)
        #check the shape of the maps
        # print("maps shape:", self.maps.shape)
        #adjust the shape of the maps ((B,  H, W,#coils,1)) to match the expected shape (B, C, 1, H, W)
        csm_corrected = self.maps.permute(0, 3, 4, 1, 2)  # Reshape to (B, #coils, 1, H, W)
        # print("csm_corrected shape:", csm_corrected.shape)
        # print(self.acceleration_rate)
        # Call the sense function
        image_recon,self.calc_noise = sense(data=kspace, csm=csm_corrected, noise_cov=self.noise_cov, acceleration_rate=self.acceleration_rate, return_noise=True)


        # print("image_recon type:", image_recon) #recon is complex
        # Remove the singleton temporal dimension to return to the original expected shape
        image_recon = image_recon.squeeze(2)  # Shape (B, 1, H, W)
        
        # #bring the image to the expected shape (B, H, W)
        image_recon = image_recon.permute(0, 2, 3, 1)  # Shape (B, H, W, 1)
        # print("image_recon shape:", image_recon.shape)

        return image_recon
    
    def get_noise_level(self):
        # self.calc_noise = self.calc_noise.to('cuda')
        # print(self.calc_noise)
        noise_magnitude = torch.abs(self.calc_noise)
        return noise_magnitude  # * torch.sqrt(torch.tensor(self.acceleration_rate, dtype=self.calc_noise.dtype, device=self.calc_noise.device))
    
    


def hard_data_consistency(
    image: torch.Tensor, acq_kspace: torch.Tensor, mask: torch.Tensor, maps: torch.Tensor
):
    """Hard project acquired k-space into reconstructed k-space.

    Args:
        image: The reconstructed image. Shape ``(B, H, W, #maps, [2])``.
        acq_kspace: The acquired k-space. Shape ``(B, H, W, #coils, [2])``.
        mask: The consistency mask. Shape ``(B, H, W)``.
        maps: The sensitivity maps. Shape ``(B, H, W, #coils, #maps, [2])``.

    Returns:
        torch.Tensor: The projected image. Shape ``(B, H, W, #maps, [2])``.
    """
    # Do not pass the mask to the SenseModel. We do not want to mask out any k-space values.
    device = image.device
    A = SenseModel(maps=maps.to(device))
    kspace = A(image, adjoint=False)
    # TODO (arjundd): Profile this operation. It may be faster to do torch.where.
    # Performance may also depend on the device.
    if mask.dtype != torch.bool:
        mask = mask.bool()
    mask = mask.to(device)
    acq_kspace = acq_kspace.to(device)
    kspace = mask * acq_kspace + (~mask) * kspace
    recon = A(kspace, adjoint=True)
    return recon


def sense(
    data: torch.Tensor,
    csm: torch.Tensor,
    noise_cov: torch.Tensor,
    acceleration_rate: int = 1,
    return_noise: bool = False,
) -> torch.Tensor:
    r"""SENSE reconstruction for parallel imaging in MRI. Combines complex-valued image signal acquired from set of
    coils.
    :math:`(S^H \Psi^{-1} S)^{-1} S^H \Psi^{-1} D`, where :math:`S` is a coil sensitivity maps, :math:`\Psi` - noise
    covariance, :math:`D` - signal acquired from coils.
    :math:`N` - number of slices, :math:`C` - number of coils, :math:`T` - number of time frames,
    :math:`F` - number of temporal frequencies, :math:`H` - image resolution in phase-encoding direction,
    :math:`W` - image resolution in readout direction.

    Args:
        data: input image data from set of coils. Size :math:`(N, C, T, H, W)` or :math:`(N, C, F, H, W)`.
        csm: coil sensitivity data. Size :math:`(N, C, 1, H, W)`.
        noise_cov: noise covariance matrix between coils. Size :math:`(C, C)`.
        acceleration_rate: acceleration factor in phase-encoding direction

    Returns:
        Combined image from all coils. Size :math:`(N, 1, T, H, W)` or :math:`(N, 1, F, H, W)`.

    References:
        Pruessmann, Klaas P., et al.
        "SENSE: sensitivity encoding for fast MRI."
        Magnetic Resonance in Medicine:
        An Official Journal of the International Society for Magnetic Resonance in Medicine 42.5 (1999): 952-962.
    """
    assert data.dim() == 5, f"Expected 5D tensor, got {data.dim()}D tensor"
    sense_matrix,calc_noise = _get_sense_matrix(
        csm=csm, noise_cov=noise_cov, acceleration_rate=acceleration_rate
    )
    sense_data = data * sense_matrix
    return sense_data.sum(dim=-4, keepdim=True),calc_noise


def _get_sense_matrix(
    csm: torch.Tensor, noise_cov: torch.Tensor, acceleration_rate: int = 1
) -> torch.Tensor:
    r"""Create the operator matrix, which help to combine the signal from coils into signal SENSE reconstruction.
    Args:
        csm: coil sensitivity data. Size :math:`(N, C, 1, H, W)`.
        noise_cov: noise covariance matrix between coils. Size :math:`(C, C)`.
        acceleration_rate: acceleration factor in phase encoding direction
    Returns:
        Matrix to use for SENSE reconstruction. Size :math:`(N, C, 1, H, W)`
    References:
        Pruessmann, Klaas P., et al.
        "SENSE: sensitivity encoding for fast MRI."
        Magnetic Resonance in Medicine:
        An Official Journal of the International Society for Magnetic Resonance in Medicine 42.5 (1999): 952-962.
    """
    assert (
        acceleration_rate > 0
    ), f"Expected acceleration rate greater 0, got {acceleration_rate}"
    sense_matrix = torch.zeros_like(csm)
    #the same shape as the csm only instead of C, we have R
    noise_var = sense_matrix.clone()[:,:acceleration_rate,:,:,:]
    # print("noise_var shape:", noise_var.shape)
    psi_inv = torch.inverse(noise_cov)
    pe_size = csm.size(-2)
    aliases = torch.arange(
        start=0, end=pe_size, step=pe_size // acceleration_rate, device=csm.device
    ).view(-1, 1)
    mask_for_reduction = (
        torch.arange(pe_size // acceleration_rate, device=csm.device) + aliases
    )
    mask_csm = torch.where(csm[:, :, :, mask_for_reduction].sum(dim=(1, -3)) != 0)
    slice_locations = mask_csm[0].repeat(acceleration_rate, 1)
    frame_locations = mask_csm[1].repeat(acceleration_rate, 1)
    pe_locations = mask_csm[2] + aliases
    ro_locations = mask_csm[3].repeat(acceleration_rate, 1)

    s = csm[slice_locations, :, frame_locations, pe_locations, ro_locations].permute(
        1, 2, 0
    )
    s_h = torch.conj(s).transpose(-1, -2)
    unmix = torch.matmul(
        torch.pinverse(torch.matmul(s_h, torch.matmul(psi_inv, s))),
        torch.matmul(s_h, psi_inv),
    )
    # print("unmix shape:", unmix.shape)
    sense_matrix[slice_locations, :, frame_locations, pe_locations, ro_locations] = unmix.transpose(0, 1)
    # print("sense_matrix shape:", sense_matrix.shape)
    noise_matrix = torch.pinverse(torch.matmul(s_h, torch.matmul(psi_inv, s)))
    # print("noise_matrix shape:", noise_matrix.shape)
    noise_var[slice_locations, :, frame_locations, pe_locations, ro_locations] = noise_matrix.transpose(0, 1)
    #Take the norm acroos the first dimension
    noise_var = noise_var[:,0]
    return sense_matrix,noise_var
