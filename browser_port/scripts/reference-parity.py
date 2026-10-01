"""Generate independent TF1/OpenCV reference data for the browser parity check.

Run with the original project's TensorFlow 1.x environment from the repo root.
Artifacts go into ignored outputs/parity; no training/model files are changed.
"""
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import tensorflow as tf
from PIL import Image
import hyper_parameters as hparams
from model_common_test import VirtualSketchingModel, DiffPastingV3
from utils import draw, load_checkpoint, update_hyperparams, image_pasting_v3_testing
from test_vectorization import sample

OUT = ROOT / 'outputs' / 'parity'
OUT.mkdir(parents=True, exist_ok=True)

def array(name, value):
    value = np.asarray(value, dtype=np.float32)
    value.tofile(str(OUT / (name + '.f32')))
    return {'file': name + '.f32', 'dims': list(value.shape)}

class ReferenceModel(VirtualSketchingModel):
    def build_combined_encoder(self, canvas, photo, entire_canvas, entire_photo, cursor, image_size, window):
        self.browser_inputs = {
            'step_patch_photo:0': photo,
            'step_patch_canvas:0': canvas,
            'step_entire_photo:0': tf.image.resize_images(entire_photo, [128, 128], method=tf.image.ResizeMethod.AREA),
            'step_entire_canvas:0': tf.image.resize_images(entire_canvas, [128, 128], method=tf.image.ResizeMethod.AREA),
        }
        return super(ReferenceModel, self).build_combined_encoder(canvas, photo, entire_canvas, entire_photo, cursor, image_size, window)

hp = update_hyperparams(hparams.get_default_hparams_clean(), 'outputs/snapshot', 'pretrain_clean_line_drawings', 'clean_line_drawings')
hp.batch_size = 1
hp.max_seq_len = 1
hp.model_mode = 'sample'
hp.use_input_dropout = hp.use_output_dropout = hp.use_recurrent_dropout = False
model = ReferenceModel(hp, gpu_mode=False)
paste = DiffPastingV3(128)
with tf.gfile.GFile('outputs/onnx/virtual_sketching_step.pb', 'rb') as f:
    gd = tf.GraphDef()
    gd.ParseFromString(f.read())
tf.import_graph_def(gd, name='frozen')
photo = np.asarray(Image.open('sample_inputs/clean_line_drawings/duck.png').convert('RGB'), np.float32)[:, :, 0] / 255
size = photo.shape[0]
report = {'size': size, 'photo': array('photo', photo), 'cases': [], 'strokes': []}
with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
    sess.run(tf.global_variables_initializer())
    load_checkpoint(sess, 'outputs/snapshot/pretrain_clean_line_drawings', gen_model_pretrain=True)
    rng = np.random.RandomState(42)
    for index, (cursor, window) in enumerate([([.5, .5], 128.), ([.22, .4], 63.75), ([.01, .98], 157.3)]):
        canvas = np.zeros_like(photo)
        if index:
            canvas[100:110, 70:600] = .75
            canvas[250:500, 340:344] = 1
        state = np.zeros((1, 1024), np.float32) if not index else rng.normal(0, .05, (1, 1024)).astype(np.float32)
        cursor = np.array(cursor, np.float32).reshape(1, 1, 2)
        feed = {model.input_photo: photo[None, :, :, None], model.curr_canvas_hard: canvas[None],
                model.initial_state: state, model.cursor_position: cursor, model.image_size: size,
                model.init_width: [.01], model.init_scaling: [1.], model.init_window_size: [window]}
        inputs, params, pen, next_state = sess.run([model.browser_inputs, model.other_params, model.pen_ras, model.final_state], feed)
        inputs.update({'step_cursor:0': cursor, 'step_image_size:0': np.int32(size),
                       'step_window_size:0': np.array([[[window]]], np.float32),
                       'step_prev_width:0': np.array([[[.01]]], np.float32), 'step_state_in:0': state})
        expected = dict(zip(['other_params:0', 'pen_ras:0', 'state_out:0'], [params, pen, next_state]))
        frozen = sess.run({k: sess.graph.get_tensor_by_name('frozen/' + k) for k in expected},
                          {sess.graph.get_tensor_by_name('frozen/' + k): v for k, v in inputs.items()})
        errors = {k: float(np.max(np.abs(expected[k] - frozen[k]))) for k in expected}
        print('Original -> frozen', index, errors, flush=True)
        assert max(errors.values()) < .0001
        report['cases'].append({'cursor': cursor.reshape(-1).tolist(), 'window': window,
            'canvas': array('canvas%d' % index, canvas), 'frozenErrors': errors,
            'inputs': {k: (int(v) if k == 'step_image_size:0' else array('in%d_%s' % (index, k.split(':')[0]), v)) for k, v in inputs.items()},
            'outputs': {k: array('out%d_%s' % (index, k.split(':')[0]), v) for k, v in expected.items()}})

    stroke_params = [[.2, .8, -.6, .7, .01, 1], [.8, .1, .7, -.5, .15, 1], [.5, .5, 0, 0, .01, 1]]
    stroke_params += rng.uniform(size=(12, 6)).tolist()
    for index, params in enumerate(stroke_params):
        width = .01 if index < 3 else .1
        p = list(params)
        patch = 1 - draw([.5, .5, p[0], p[1], (p[2]+1)/2, (p[3]+1)/2, width, p[4], 1., 1.])
        cursor = np.array([.023, .91], np.float32)
        pasted = image_pasting_v3_testing(patch, cursor, size, 93.7, paste, sess)
        report['strokes'].append({'params': p, 'width': width, 'patch': array('patch%d' % index, patch),
                                 'pasted': array('paste%d' % index, pasted), 'cursor': cursor.tolist(), 'window': 93.7})
    random.seed(42)
    np.random.seed(42)
    result = sample(sess, model, photo[None], np.array([[[[.5, .5]]]], np.float32), size, 48,
                    [500] * 10, False, paste, 12, .95)
    params, raw, soft, canvas, windows, cursors, lengths = result
    report['sequence'] = {'params': params[0], 'lengths': lengths,
                          'cursors': [[.5, .5]] + [c.reshape(-1).tolist() for c in cursors],
                          'canvas': array('reference_canvas', canvas[0])}
    Image.fromarray(np.round((1 - canvas[0]) * 255).astype(np.uint8)).save(str(OUT / 'reference.png'))
with open(str(OUT / 'reference.json'), 'w') as f:
    json.dump(report, f)
print('Wrote', OUT, flush=True)
