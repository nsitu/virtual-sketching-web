"""Export a single autoregressive inference step for browser ONNX inference.

This script is intentionally written for the TensorFlow 1.x environment used by
the original project.  It exports a frozen GraphDef; conversion to ONNX is kept
as a separate step so that the exporter can run with TensorFlow 1.15 while a
modern tf2onnx installation handles the frozen graph.

The exported graph does not contain image cropping, canvas compositing, or the
48-step sampling loop.  Those operations are host-side operations in the
browser.  Keeping them outside the graph avoids TensorFlow control-flow and
CropAndResize conversion problems and exposes the HyperLSTM state explicitly.
"""

from __future__ import print_function

import argparse
import os

import tensorflow as tf

import hyper_parameters as hparams
from model_common_test import VirtualSketchingModel
from utils import load_checkpoint, update_hyperparams


class VirtualSketchingStepModel(VirtualSketchingModel):
    """Batch-one model that predicts exactly one stroke step.

    The variable scopes used by ``config_model`` and the decoder output heads
    match the normal sampling graph so the trained generator checkpoint can be
    restored.  Inputs are deliberately static except for scalar image/window
    metadata; the browser performs the variable-size image crop and resize.
    """

    def __init__(self, hps, gpu_mode=False):
        self.hps = hps
        assert hps.model_mode in ['eval_sample', 'sample']
        if gpu_mode:
            self.build_model()
        else:
            with tf.device('/cpu:0'):
                self.build_model()

    def build_model(self):
        # This creates the same decoder cell configuration as the original
        # sampling graph, but does not build the 48-step sequence graph.
        self.config_model()

        batch_size = self.hps.batch_size
        raster_size = self.hps.raster_size
        input_channel = self.hps.input_channel

        # Patch tensors are already normalized to [-1, 1].  Full-image tensors
        # are raw [0, 1] tensors; build_combined_encoder applies the original
        # normalization after its fixed-size resize.
        self.step_patch_photo = tf.placeholder(
            tf.float32, [batch_size, raster_size, raster_size, input_channel],
            name='step_patch_photo')
        self.step_patch_canvas = tf.placeholder(
            tf.float32, [batch_size, raster_size, raster_size, 1],
            name='step_patch_canvas')
        self.step_entire_photo = tf.placeholder(
            tf.float32, [batch_size, raster_size, raster_size, input_channel],
            name='step_entire_photo')
        self.step_entire_canvas = tf.placeholder(
            tf.float32, [batch_size, raster_size, raster_size, 1],
            name='step_entire_canvas')
        self.step_cursor = tf.placeholder(
            tf.float32, [batch_size, 1, 2], name='step_cursor')
        self.step_image_size = tf.placeholder(
            tf.int32, [], name='step_image_size')
        self.step_window_size = tf.placeholder(
            tf.float32, [batch_size, 1, 1], name='step_window_size')
        self.step_prev_width = tf.placeholder(
            tf.float32, [batch_size, 1, 1], name='step_prev_width')

        state_size = int(self.dec_cell.state_size)
        self.step_state_in = tf.placeholder(
            tf.float32, [batch_size, state_size], name='step_state_in')

        combined_z = self.build_combined_encoder(
            self.step_patch_canvas,
            self.step_patch_photo,
            self.step_entire_canvas,
            self.step_entire_photo,
            self.step_cursor,
            self.step_image_size,
            self.step_window_size)
        combined_z = tf.expand_dims(combined_z, axis=1)

        window_top = tf.stop_gradient(
            self.step_window_size / tf.cast(self.step_image_size, tf.float32))
        window_bottom = tf.stop_gradient(
            self.step_window_size / tf.cast(self.hps.min_window_size, tf.float32))

        if self.hps.concat_win_size:
            decoder_input = tf.concat([
                tf.stop_gradient(self.step_prev_width),
                window_top,
                window_bottom,
                combined_z,
            ], axis=2)
        else:
            decoder_input = tf.concat([
                tf.stop_gradient(self.step_prev_width), combined_z
            ], axis=2)

        if self.hps.concat_cursor:
            decoder_input = tf.concat(
                [tf.stop_gradient(self.step_cursor), decoder_input], axis=2)

        # Call the custom cell directly.  This avoids dynamic_rnn/Loop nodes
        # and makes the HyperLSTM state an ordinary ONNX tensor input/output.
        # dynamic_rnn creates a child scope named ``rnn`` beneath the
        # surrounding RNN_DEC scope.  Reuse that exact scope so checkpoint
        # variable names remain compatible with the original graph.
        with tf.variable_scope('RNN_DEC/rnn', reuse=tf.AUTO_REUSE):
            h_output, next_state = self.dec_cell(
                decoder_input[:, 0, :], self.step_state_in)

        output_w_pen = tf.get_variable(
            'DEC_RNN_out_pen/output_w',
            [self.hps.dec_rnn_size, 2])
        output_b_pen = tf.get_variable(
            'DEC_RNN_out_pen/output_b', [2],
            initializer=tf.constant_initializer(0.0))
        output_w_params = tf.get_variable(
            'DEC_RNN_out_params/output_w',
            [self.hps.dec_rnn_size, 6])
        output_b_params = tf.get_variable(
            'DEC_RNN_out_params/output_b', [6],
            initializer=tf.constant_initializer(0.0))

        pen_logits = tf.matmul(h_output, output_w_pen) + output_b_pen
        params_logits = tf.matmul(h_output, output_w_params) + output_b_params
        outputs = tf.concat([pen_logits, params_logits], axis=1)

        other_params, pen_ras = self.get_mixture_coef(outputs)
        pen_state_soft = self.differentiable_argmax(
            pen_logits, self.hps.soft_beta)

        self.step_other_params = tf.identity(other_params, name='other_params')
        self.step_pen_ras = tf.identity(pen_ras, name='pen_ras')
        self.step_pen_state_soft = tf.identity(
            pen_state_soft, name='pen_state_soft')
        self.step_state_out = tf.identity(next_state, name='state_out')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-base-dir', default='outputs/snapshot')
    parser.add_argument('--model-name', default='pretrain_clean_line_drawings')
    parser.add_argument('--infer-dataset', default='clean_line_drawings')
    parser.add_argument('--output-graphdef', default='outputs/onnx/virtual_sketching_step.pb')
    parser.add_argument(
        '--gpu', action='store_true',
        help='Build/export on GPU. CPU is the default for portability.')
    return parser.parse_args()


def main():
    args = parse_args()
    defaults = {'clean_line_drawings': hparams.get_default_hparams_clean,
                'rough_sketches': hparams.get_default_hparams_rough,
                'faces': hparams.get_default_hparams_normal}
    model_params = defaults[args.infer_dataset]()
    model_params = update_hyperparams(
        model_params, args.model_base_dir, args.model_name, args.infer_dataset)

    # The browser runs one sample at a time.  Dropout is disabled by the
    # dataset test configuration and is forced off here for export safety.
    model_params.batch_size = 1
    model_params.model_mode = 'sample'
    model_params.use_recurrent_dropout = False
    model_params.use_input_dropout = False
    model_params.use_output_dropout = False

    tf.reset_default_graph()
    model = VirtualSketchingStepModel(model_params, gpu_mode=args.gpu)

    output_path = os.path.abspath(args.output_graphdef)
    output_dir = os.path.dirname(output_path)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        checkpoint_dir = os.path.join(args.model_base_dir, args.model_name)
        load_checkpoint(sess, checkpoint_dir, gen_model_pretrain=True)

        frozen_graph = tf.graph_util.convert_variables_to_constants(
            sess,
            sess.graph_def,
            [
                model.step_other_params.op.name,
                model.step_pen_ras.op.name,
                model.step_pen_state_soft.op.name,
                model.step_state_out.op.name,
            ])
        with tf.gfile.GFile(output_path, 'wb') as graph_file:
            graph_file.write(frozen_graph.SerializeToString())

    print('Wrote frozen step graph: {}'.format(output_path))
    print('Convert it with a modern tf2onnx environment using:')
    print('python -m tf2onnx.convert --graphdef "{}"'.format(output_path) +
          ' --inputs step_patch_photo:0,step_patch_canvas:0,step_entire_photo:0,'
          'step_entire_canvas:0,step_cursor:0,step_image_size:0,step_window_size:0,'
          'step_prev_width:0,step_state_in:0'
          ' --outputs other_params:0,pen_ras:0,pen_state_soft:0,state_out:0'
          ' --opset 18 --output virtual_sketching_step.onnx')


if __name__ == '__main__':
    main()
