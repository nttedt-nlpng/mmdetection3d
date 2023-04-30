import torch

from mmdet3d.registry import MODELS
from mmdet3d.models.detectors import CenterPoint


@MODELS.register_module()
class BEVDet(CenterPoint):
    def __init__(self,
                 img_view_transformer=None,
                 img_bev_encoder_backbone=None,
                 img_bev_encoder_neck=None,
                 **kwargs):
        super(BEVDet, self).__init__(**kwargs)

        if img_view_transformer is not None:
            self.img_view_transformer = MODELS.build(img_view_transformer)
        else:
            self.img_view_transformer = None

        if img_bev_encoder_backbone is not None:
            self.img_bev_encoder_backbone = MODELS.build(
                img_bev_encoder_backbone)
        else:
            self.img_bev_encoder_backbone = torch.nn.Identity()

        if img_bev_encoder_neck is not None:
            self.img_bev_encoder_neck = MODELS.build(img_bev_encoder_neck)
        else:
            self.img_bev_encoder_neck = torch.nn.Identity()

    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        x = self.img_backbone(imgs)
        if self.with_img_neck:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x

    # TODO: mmengine 2.0 not supporting force_fp32 anymore
    # need to consider whether this decorator.
    # @force_fp32()
    def bev_encoder(self, x):
        x = self.img_bev_encoder_backbone(x)
        x = self.img_bev_encoder_neck(x)
        if type(x) in [list, tuple]:
            x = x[0]
        return x

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        x = self.image_encoder(img[0])
        x = self.img_view_transformer([x] + img[1:7])
        x = self.bev_encoder(x)
        return [x]


class BEVDepth_Base(object):
    def extract_feat(self, batch_inputs_dict,
                     batch_input_metas):
        """(Modified) Extract features from images and points.

        Args:
            batch_inputs_dict (dict): Dict of batch inputs. It
                contains

                - points (List[tensor]):  Point cloud of multiple inputs.
                - imgs (tensor): Image tensor with shape (B, C, H, W).
            batch_input_metas (list[dict]): Meta information of multiple inputs
                in a batch.

        Returns:
             tuple: Two elements in tuple arrange as
             image features and point cloud features.
        """
        voxel_dict = batch_inputs_dict.get('voxels', None)
        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        img_feats = self.extract_img_feat(imgs, batch_input_metas)
        pts_feats = self.extract_pts_feat(
            voxel_dict,
            points=points,
            img_feats=img_feats,
            batch_input_metas=batch_input_metas)
        return (img_feats, pts_feats)

    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        img_feats, depth = self.extract_img_feat(img, img_metas)
        pts_feats = None
        return (img_feats, pts_feats, depth)

    def simple_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats, _, _ = self.extract_feat(points, img=img, img_metas=img_metas)
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """

        img_feats, pts_feats, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas)
        assert self.with_pts_bbox
        # assert len(img_inputs) == 8
        depth_gt = img_inputs[7]
        loss_depth = self.img_view_transformer.get_depth_loss(depth_gt, depth)
        losses = dict(loss_depth=loss_depth)
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)

        # some modifications
        if hasattr(self.img_view_transformer, 'loss_depth_reg_weight') and self.img_view_transformer.loss_depth_reg_weight > 0:
            losses['loss_depth_reg'] = self.img_view_transformer.get_depth_reg_loss(depth_gt, depth)

        return losses


class BEVDepth(BEVDepth_Base, BEVDet):
    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        x = self.image_encoder(img[0])

        # img: imgs, rots, trans, intrins, post_rots, post_trans, gt_depths, sensor2sensors
        rots, trans, intrins, post_rots, post_trans, bda = img[1:7]

        mlp_input = self.img_view_transformer.get_mlp_input(
            rots, trans, intrins, post_rots, post_trans, bda)
        geo_inputs = [rots, trans, intrins,
                      post_rots, post_trans, bda, mlp_input]

        x, depth = self.img_view_transformer([x] + geo_inputs)
        x = self.bev_encoder(x)

        return [x], depth
