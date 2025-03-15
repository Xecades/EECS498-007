"""
Implements a style transfer in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from a6_helper import *


def hello():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print('Hello from style_transfer.py!')


def content_loss(content_weight, content_current, content_original):
    """
    Compute the content loss for style transfer.

    Inputs:
    - content_weight: Scalar giving the weighting for the content loss.
    - content_current: features of the current image; this is a PyTorch Tensor of shape
      (1, C_l, H_l, W_l).
    - content_original: features of the content image, Tensor with shape (1, C_l, H_l, W_l).

    Returns:
    - scalar content loss
    """
    ############################################################################
    # TODO: Compute the content loss for style transfer.                       #
    ############################################################################
    # Replace "pass" statement with your code
    Cl = content_current.shape[1]
    Fl = content_current.view(Cl, -1)
    Pl = content_original.view(Cl, -1)
    loss = content_weight * F.mse_loss(Fl, Pl, reduction="sum")
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
    return loss


def gram_matrix(features, normalize=True):
    """
    Compute the Gram matrix from features.

    Inputs:
    - features: PyTorch Tensor of shape (N, C, H, W) giving features for
      a batch of N images.
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)

    Returns:
    - gram: PyTorch Tensor of shape (N, C, C) giving the
      (optionally normalized) Gram matrices for the N input images.
    """
    gram = None
    ############################################################################
    # TODO: Compute the Gram matrix from features.                             #
    # Don't forget to implement for both normalized and non-normalized version #
    ############################################################################
    # Replace "pass" statement with your code
    N, C, H, W = features.shape
    features = features.view(N, C, -1)
    gram = features @ features.transpose(1, 2)
    if normalize:
        gram = gram / (C * H * W)
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
    return gram


def style_loss(feats, style_layers, style_targets, style_weights):
    """
    Computes the style loss at a set of layers.

    Inputs:
    - feats: list of the features at every layer of the current image, as produced by
      the extract_features function.
    - style_layers: List of layer indices into feats giving the layers to include in the
      style loss.
    - style_targets: List of the same length as style_layers, where style_targets[i] is
      a PyTorch Tensor giving the Gram matrix of the source style image computed at
      layer style_layers[i].
    - style_weights: List of the same length as style_layers, where style_weights[i]
      is a scalar giving the weight for the style loss at layer style_layers[i].

    Returns:
    - loss: A PyTorch Tensor holding a scalar giving the style loss.
    """
    ############################################################################
    # TODO: Computes the style loss at a set of layers.                        #
    # Hint: you can do this with one for loop over the style layers, and       #
    # should not be very much code (~5 lines).                                 #
    # You will need to use your gram_matrix function.                          #
    ############################################################################
    # Replace "pass" statement with your code
    loss = 0
    for l in range(len(style_layers)):
        layer = style_layers[l]
        target = style_targets[l]
        weight = style_weights[l]
        current = feats[layer]
        G = gram_matrix(current)
        loss += weight * F.mse_loss(G, target, reduction="sum")
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
    return loss


def tv_loss(img, tv_weight):
    """
    Compute total variation loss.

    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.

    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    ############################################################################
    # TODO: Compute total variation loss.                                      #
    # Your implementation should be vectorized and not require any loops!      #
    ############################################################################
    # Replace "pass" statement with your code
    _, _, H, W = img.shape
    tv_x = ((img[:, :, :, 1:] - img[:, :, :, :-1]) ** 2).sum()
    tv_y = ((img[:, :, 1:, :] - img[:, :, :-1, :]) ** 2).sum()
    loss = tv_weight * (tv_x + tv_y)
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
    return loss


def guided_gram_matrix(features, masks, normalize=True):
    """
    Inputs:
      - features: PyTorch Tensor of shape (N, R, C, H, W) giving features for
        a batch of N images.
      - masks: PyTorch Tensor of shape (N, R, H, W)
      - normalize: optional, whether to normalize the Gram matrix
          If True, divide the Gram matrix by the number of neurons (H * W * C)

      Returns:
      - gram: PyTorch Tensor of shape (N, R, C, C) giving the
        (optionally normalized) guided Gram matrices for the N input images.
    """
    gram = None
    ##############################################################################
    # TODO: Compute the guided Gram matrix from features.                        #
    # Apply the regional guidance mask to its corresponding feature and          #
    # calculate the Gram Matrix. You are allowed to use one for-loop in          #
    # this problem.                                                              #
    ##############################################################################
    # Replace "pass" statement with your code
    N, R, C, H, W = features.shape
    features = features.view(N * R, C, H, W)
    masks = masks.view(N * R, 1, H, W)
    gram = features * masks
    gram = gram.view(N * R, C, -1)
    gram = gram @ gram.transpose(1, 2)
    if normalize:
        gram = gram / (C * H * W)
    gram = gram.view(N, R, C, C)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return gram


def guided_style_loss(feats, style_layers, style_targets, style_weights, content_masks):
    """
    Computes the style loss at a set of layers.

    Inputs:
    - feats: list of the features at every layer of the current image, as produced by
      the extract_features function.
    - style_layers: List of layer indices into feats giving the layers to include in the
      style loss.
    - style_targets: List of the same length as style_layers, where style_targets[i] is
      a PyTorch Tensor giving the guided Gram matrix of the source style image computed at
      layer style_layers[i].
    - style_weights: List of the same length as style_layers, where style_weights[i]
      is a scalar giving the weight for the style loss at layer style_layers[i].
    - content_masks: List of the same length as feats, giving a binary mask to the
      features of each layer.

    Returns:
    - style_loss: A PyTorch Tensor holding a scalar giving the style loss.
    """
    ############################################################################
    # TODO: Computes the guided style loss at a set of layers.                 #
    ############################################################################
    # Replace "pass" statement with your code
    loss = 0
    for l in range(len(style_layers)):
        layer = style_layers[l]
        target = style_targets[l]
        weight = style_weights[l]
        current = feats[layer]
        mask = content_masks[layer]
        G = guided_gram_matrix(current, mask)
        loss += weight * F.mse_loss(G, target, reduction="sum")
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
    return loss
