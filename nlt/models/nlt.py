# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=relative-beyond-top-level,arguments-differ

from os.path import join, exists
from glob import glob
import numpy as np
from tqdm import tqdm

import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import tensorflow_addons as tfa

import xiuminglib as xm

import losses
from networks import convnet
from util import logging as logutil, io as ioutil, img as imgutil, \
    tensor as tutil
from .base import Model as BaseModel


logger = logutil.Logger(loggee="models/nlt", debug_mode=False)


class Model(BaseModel):
    def __init__(self, config):
        # Needed by Barron loss
        self.imh = config.getint('DEFAULT', 'imh')
        self.imw = config.getint('DEFAULT', 'imw')
        super().__init__(config)
        # Networks
        depth0 = config.getint('DEFAULT', 'depth0')
        depth = config.getint('DEFAULT', 'depth')
        kernel = config.getint('DEFAULT', 'kernel')
        stride = config.getint('DEFAULT', 'stride')
        norm = config.get('DEFAULT', 'norm')
        act = config.get('DEFAULT', 'act')
        pool = config.get('DEFAULT', 'pool')
        net_args = (depth0, depth, kernel, stride)
        net_kwargs = {'norm_type': norm, 'act_type': act, 'pool_type': pool}
        self.net = {
            'query': convnet.Network(*net_args, **net_kwargs),
            'obs': convnet.Network(*net_args, **net_kwargs)}
        self.net['obs'].layers = [
            x for i, x in enumerate(self.net['obs'].layers)
            if self.net['obs'].is_contracting[i]] # remove decoding layers
        # Other parameters shared by more than one method
        self.uvh = self.config.getint('DEFAULT', 'uvh')
        self.uvw = self.config.getint('DEFAULT', 'uvw')
        #
        self.psnr = xm.metric.PSNR(np.float32)

    def _init_loss(self):
        """Overriding the base because Barron loss requires image resolutions,
        which may not be generic to any model.
        """
        wloss = []
        loss_str = self.config.get('DEFAULT', 'loss')
        for x in loss_str.split(','):
            loss_name, weight = self._parse_loss_and_weight(x)
            if loss_name == 'lpips':
                loss = losses.LPIPS(per_ch=False)
            elif loss_name == 'l1':
                loss = losses.L1()
            elif loss_name == 'l2':
                loss = losses.L2()
            elif loss_name == 'ssim':
                loss = losses.SSIM(1 - 0)
            elif loss_name == 'barron':
                loss = losses.Barron(self.imw, self.imh)
            else:
                raise NotImplementedError(loss_name)
            wloss.append((weight, loss))
        return wloss

    def call(self, batch, mode, obs_override=None):
        self._validate_mode(mode)
        id_, base, cvis, lvis, warp, rgb, rgb_camspc, nn_id, nn_base, \
            nn_rgb, nn_rgb_camspc = batch # *rgb* are placeholders for testing
        # NOTE: When the neighbor paths don't exist, nn_* are
        # all-zero placeholders
        x = tf.concat((base, cvis, lvis), axis=3)
        y_obs = [nn_rgb - nn_base] # only one neighbor
        # Pass them through the layers
        pred = self._call(x, y_obs, obs_override=obs_override)
        skip_connect_base = self.config.getboolean(
            'DEFAULT', 'skip_connect_base')
        if skip_connect_base:
            pred += base
        # Warp to camera space
        warp = tf.stack((
            warp[:, :, :, 0] * self.uvw,
            warp[:, :, :, 1] * self.uvh), axis=3)
        fg = tf.ones(pred.shape, dtype=pred.dtype)
        fg = imgutil.set_left_top_corner(fg, 0)
        base = imgutil.set_left_top_corner(base, 0)
        pred = imgutil.set_left_top_corner(pred, 0) # make top left corner
        # black because background colors will be sampled from here
        fg_camspc = tfa.image.resampler(fg, warp) # better than using alpha
        base_camspc = tfa.image.resampler(base, warp)
        pred_camspc = tfa.image.resampler(pred, warp) # full-resolution
        # because warp is full-resolution
        fg_camspc = imgutil.resize(fg_camspc, new_h=self.imh, new_w=self.imw)
        base_camspc = imgutil.resize(
            base_camspc, new_h=self.imh, new_w=self.imw)
        pred_camspc = imgutil.resize(
            pred_camspc, new_h=self.imh, new_w=self.imw)
        # For summary
        to_vis = {
            'id': id_,
            'nn_id': nn_id,
            'base_camspc': base_camspc,
            'pred': pred,
            'pred_camspc': pred_camspc,
            'nn_camspc': nn_rgb_camspc}
        # Training or validation
        if mode in ('train', 'vali'):
            # For computing loss
            gt_camspc = imgutil.alpha_blend(
                rgb_camspc, fg_camspc) # alternatively, use query['cam_alpha']
            loss_kwargs = {}
            to_vis['gt'] = rgb
            to_vis['gt_camspc'] = gt_camspc
            return pred_camspc, gt_camspc, loss_kwargs, to_vis
        # Testing
        return pred_camspc, None, None, to_vis

    def _call(self, query_x, obs_xs, obs_weights=None, obs_override=None):
        use_obs = self.config.getboolean('DEFAULT', 'use_obs')
        if obs_weights is not None:
            obs_weights = tf.reshape(
                obs_weights, (obs_weights.shape[0], 1, 1, 1, -1))
        query_featmaps = []
        for layer_i, (layer_query, is_contracting) in enumerate(zip(
                self.net['query'].layers,
                self.net['query'].is_contracting)):
            if is_contracting:
                layer_obs = self.net['obs'].layers[layer_i]
                # Process each observation with the same network
                obs_ys = []
                for x in obs_xs:
                    obs_ys.append(layer_obs(x))
                logger.debug("Spatially contracting...")
                logger.debug("[O] %s -> %s" % (
                    [tutil.shape_as_list(x) for x in obs_xs],
                    [tutil.shape_as_list(x) for x in obs_ys]))
                # Aggregate the feature maps from different observations
                obs_agg = tf.concat([tf.expand_dims(t, -1) for t in obs_ys], -1)
                if obs_weights is not None:
                    obs_agg = obs_weights * obs_agg
                obs_agg = tf.reduce_mean(obs_agg, axis=-1)
                # Don't concat the aggregated feature map in observation network
                obs_xs = obs_ys
                # Concatenate it only to the query network
                query_y = layer_query(query_x)
                logger.debug("[Q] %s -> %s" % (
                    tutil.shape_as_list(query_x), tutil.shape_as_list(query_y)))
                if use_obs:
                    if obs_override is not None:
                        obs_agg = obs_override[layer_i]
                    query_x = tf.concat((query_y, obs_agg), axis=-1)
                    logger.debug("[Q] Aggregated %s concat to %s" % (
                        tutil.shape_as_list(obs_agg),
                        tutil.shape_as_list(query_y)))
                else:
                    query_x = query_y
                query_featmaps.append(query_x)
                logger.debug("[Q] New size: %s" % tutil.shape_as_list(query_x))
            else:
                # Skip connections between encoder and decoder
                if query_featmaps:
                    enc_featmap = query_featmaps.pop()
                    logger.debug("Spatially expanding...")
                    logger.debug("[Q] Encoder feature %s concat to %s" % (
                        tutil.shape_as_list(enc_featmap),
                        tutil.shape_as_list(query_x)))
                    query_x = tf.concat((query_x, enc_featmap), axis=-1)
                    logger.debug(
                        "[Q] New size: %s" % tutil.shape_as_list(query_x))
                else:
                    logger.debug("[Q] All feature maps concat; no concat")
                query_y = layer_query(query_x)
                logger.debug("[Q] %s -> %s" % (
                    tutil.shape_as_list(query_x), tutil.shape_as_list(query_y)))
                query_x = query_y
        return query_y

    def compute_loss(self, pred, gt, **kwargs):
        loss = 0
        for weight, loss_func in self.wloss:
            loss += weight * loss_func(gt, pred, **kwargs)
        return loss

    def vis_batch(
            self, data_dict, outdir, mode, dump_raw_to=None,
            text_loc_ratio=0.05, text_size_ratio=0.05, text_color=(1, 1, 1)):
        is_linear = self.config.getboolean('DEFAULT', 'linear_space')
        self._validate_mode(mode)
        # To NumPy
        ids = [x.numpy().decode() for x in data_dict['id']]
        nn_ids = [x.numpy().decode() for x in data_dict['nn_id']]
        bases = data_dict['base_camspc'].numpy()
        preds = data_dict['pred_camspc'].numpy()
        nns = data_dict['nn_camspc'].numpy()
        gts = None if mode == 'test' else data_dict['gt_camspc'].numpy()
        #
        for i in range(len(ids)):
            imgs = {}
            base = np.clip(bases[i], 0, 1)
            pred = np.clip(preds[i], 0, 1)
            nn = np.clip(nns[i], 0, 1)
            gt = None if gts is None else np.clip(gts[i], 0, 1)
            # Linear to sRGB
            if is_linear:
                base = xm.img.linear2srgb(base)
                pred = xm.img.linear2srgb(pred)
                nn = xm.img.linear2srgb(nn)
                gt = None if gt is None else xm.img.linear2srgb(gt)
            # Write to disk
            imgs['base'] = xm.io.img.write_arr(
                base, join(outdir, '%d_base.png' % i))
            imgs['pred'] = xm.io.img.write_arr(
                pred, join(outdir, '%d_pred.png' % i))
            xm.io.img.write_arr(nn, join(outdir, '%d_nn.png' % i))
            imgs['gt'] = None if gt is None else xm.io.img.write_arr(
                gt, join(outdir, '%d_gt.png' % i))
            # Make .apng
            hw = base.shape[:2]
            label_loc = (
                int(text_loc_ratio * hw[1]), int(text_loc_ratio * hw[0]))
            font_size = int(text_size_ratio * hw[0])
            xm.vis.video.make_apng(
                (imgs['base'], imgs['pred']),
                labels=('Diffuse Base', 'Prediction'),
                label_top_left_xy=label_loc, font_size=font_size,
                font_color=text_color,
                outpath=join(outdir, '%d_base-vs-pred.apng' % i))
            if imgs['gt'] is not None:
                xm.vis.video.make_apng(
                    (imgs['gt'], imgs['pred']),
                    labels=('Ground Truth', 'Prediction'),
                    label_top_left_xy=label_loc, font_size=font_size,
                    font_color=text_color,
                    outpath=join(outdir, '%d_gt-vs-pred.apng' % i))
        # Write metadata
        for i, id_ in enumerate(ids):
            metadata = {'id': id_, 'nn_id': nn_ids[i]}
            pred = np.clip(preds[i], 0, 1)
            base = np.clip(bases[i], 0, 1)
            if gts is not None:
                gt = np.clip(gts[i], 0, 1)
                pred_psnr = self.psnr(gt, pred)
                base_psnr = self.psnr(gt, base)
                metadata['pred_psnr'] = pred_psnr
                metadata['base_psnr'] = base_psnr
            ioutil.write_json(metadata, join(outdir, '%d_metadata.json' % i))
        # Optionally dump raw to disk
        if dump_raw_to is not None:
            ioutil.write_pickle(data_dict, dump_raw_to)

    def compile_batch_vis(
            self, batch_vis_dirs, outpref, mode, fps=6,
            file_explorer='http://vision38.csail.mit.edu'): # FIXME
        self._validate_mode(mode)
        if mode in ('train', 'vali'):
            outpath = outpref + '.html'
            self._compile_into_webpage(
                batch_vis_dirs, outpath, title="NLT (%s)" % mode)
        else:
            outpath = outpref + '.mp4'
            self._compile_into_video(batch_vis_dirs, outpath, fps=fps)
        view_at = file_explorer + outpath
        return view_at # to be logged into TensorBoard

    @staticmethod
    def _compile_into_webpage(batch_dirs, out_html, title=None):
        rows, caps, types = [], [], []
        # For each batch (which has just one sample)
        for batch_dir in batch_dirs:
            for metadata_path in sorted(
                    glob(join(batch_dir, '?_metadata.json'))):
                path_prefix = metadata_path[:-len('metadata.json')]
                metadata = ioutil.read_json(metadata_path)
                metadata = str(metadata)
                row = [
                    metadata,
                    path_prefix + 'base-vs-pred.apng',
                    path_prefix + 'gt-vs-pred.apng',
                    path_prefix + 'nn.png']
                rowcaps = [
                    "Metadata", "Prediction vs. Diffuse Base",
                    "Prediction vs. Ground Truth",
                    "Nearest Neighbor"]
                rowtypes = ['text', 'image', 'image', 'image']
                rows.append(row)
                caps.append(rowcaps)
                types.append(rowtypes)
        n_rows = len(rows)
        assert n_rows > 0, "No row"
        # Write HTML
        html = xm.vis.html.HTML()
        if title is not None:
            html.add_header(title)
        img_table = html.add_table()
        for r, rcaps, rtypes in zip(rows, caps, types):
            img_table.add_row(r, rtypes, captions=rcaps)
        html.save(out_html)

    @staticmethod
    def _compile_into_video(batch_dirs, out_mp4, fps=12):
        frames = {}
        for batch_dir in tqdm(batch_dirs, desc="Compiling visualized batches"):
            for metadata_path in glob(join(batch_dir, '?_metadata.json')):
                path_prefix = metadata_path[:-len('metadata.json')]
                pred_path = path_prefix + 'pred.png'
                if not exists(pred_path):
                    logger.warn(
                        "Skipping because of missing file:\n\t%s" % pred_path)
                    continue
                # Metadata
                metadata = ioutil.read_json(metadata_path)
                id_ = metadata['id']
                # Prediction
                pred = xm.io.img.load(pred_path, as_array=True)
                frame = pred
                frames[id_] = frame
        # Make video
        frames_sorted = [frames[k] for k in sorted(frames)]
        ioutil.write_video(frames_sorted, out_mp4, fps=fps)
