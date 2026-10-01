"""Independent TF1 RGB references; pass --rough for rough-sketch references."""
import json
import os
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
from dataset_utils import GeneralRawDataLoader
from model_common_test import VirtualSketchingModel, DiffPastingV3
from utils import load_checkpoint, update_hyperparams
from test_photograph_to_line import sample
from test_rough_sketch_simplification import sample as sample_rough, move_cursor_to_undrawn
from unittest.mock import patch

ROUGH = '--rough' in sys.argv
DATASET = 'rough_sketches' if ROUGH else 'faces'
CHECKPOINT = 'pretrain_' + DATASET
STEM = 'rough' if ROUGH else 'faces'
OUT = ROOT / 'outputs' / ('rough-parity' if ROUGH else 'photo-parity')
OUT.mkdir(parents=True, exist_ok=True)

def array(name, value):
    value = np.asarray(value, np.float32)
    value.tofile(str(OUT / (name + '.f32')))
    return {'file': name + '.f32', 'dims': list(value.shape)}

class ReferenceModel(VirtualSketchingModel):
    def build_combined_encoder(self, canvas, photo, entire_canvas, entire_photo, cursor, image_size, window):
        self.browser_inputs = {
            'step_patch_photo:0': photo, 'step_patch_canvas:0': canvas,
            'step_entire_photo:0': tf.image.resize_images(entire_photo, [128, 128], method=tf.image.ResizeMethod.AREA),
            'step_entire_canvas:0': tf.image.resize_images(entire_canvas, [128, 128], method=tf.image.ResizeMethod.AREA),
        }
        return super(ReferenceModel, self).build_combined_encoder(canvas, photo, entire_canvas, entire_photo, cursor, image_size, window)

hp = update_hyperparams(hparams.get_default_hparams_rough() if ROUGH else hparams.get_default_hparams_normal(), 'outputs/snapshot', CHECKPOINT, DATASET)
hp.batch_size = hp.max_seq_len = 1
hp.model_mode = 'sample'
hp.use_input_dropout = hp.use_output_dropout = hp.use_recurrent_dropout = False
model = ReferenceModel(hp, gpu_mode=False)
paste = DiffPastingV3(128)
with tf.gfile.GFile('outputs/onnx/virtual_sketching_%s_step.pb' % STEM, 'rb') as f:
    gd = tf.GraphDef()
    gd.ParseFromString(f.read())
tf.import_graph_def(gd, name='frozen')
source = 'sample_inputs/rough_sketches/penguin1.png' if ROUGH else 'sample_inputs/faces/1390.png'
photo, size = GeneralRawDataLoader(source, 128, DATASET).gen_input_images(source)
report = {'size': size, 'photo': array('photo', photo[0]), 'cases': [], 'preprocessing': [], 'moves': []}
rng = np.random.RandomState(17)
# Non-square inputs exercise right/bottom padding, upsampling, downsampling,
# odd dimensions and RGB separation independently of the model.
for index, (w, h) in enumerate([(513, 337), (71, 129), (256, 256), (147, 511)]):
    raw = rng.randint(0, 256, (h, w, 3)).astype(np.uint8)
    path = str(OUT / ('input%d.png' % index))
    Image.fromarray(raw, 'RGB').save(path)
    expected, _ = GeneralRawDataLoader(path, 128, DATASET).gen_input_images(path)
    report['preprocessing'].append({'width': w, 'height': h, 'raw': array('raw%d' % index, raw / 255.),
                                     'expected': array('prepared%d' % index, expected[0])})

with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
    sess.run(tf.global_variables_initializer())
    load_checkpoint(sess, 'outputs/snapshot/' + CHECKPOINT, gen_model_pretrain=True)
    for index, (cursor, window) in enumerate([([.5, .5], 128.), ([.23, .67], 63.75), ([.01, .98], 157.3)]):
        canvas = np.zeros((size, size), np.float32)
        if index:
            canvas[100:110, 50:200] = .75
        state = np.zeros((1, 1024), np.float32) if not index else rng.normal(0, .05, (1, 1024)).astype(np.float32)
        cursor = np.array(cursor, np.float32).reshape(1, 1, 2)
        feed = {model.input_photo: photo, model.curr_canvas_hard: canvas[None], model.initial_state: state,
                model.cursor_position: cursor, model.image_size: size, model.init_width: [.01],
                model.init_scaling: [1.], model.init_window_size: [window]}
        inputs, params, pen, next_state = sess.run([model.browser_inputs, model.other_params, model.pen_ras, model.final_state], feed)
        inputs.update({'step_cursor:0': cursor, 'step_image_size:0': np.int32(size),
                       'step_window_size:0': np.array([[[window]]], np.float32),
                       'step_prev_width:0': np.array([[[.01]]], np.float32), 'step_state_in:0': state})
        expected = dict(zip(['other_params:0', 'pen_ras:0', 'state_out:0'], [params, pen, next_state]))
        frozen = sess.run({k: sess.graph.get_tensor_by_name('frozen/' + k) for k in expected},
                          {sess.graph.get_tensor_by_name('frozen/' + k): v for k, v in inputs.items()})
        errors = {k: float(np.max(np.abs(expected[k] - frozen[k]))) for k in expected}
        print(DATASET, 'original -> frozen', index, errors, flush=True)
        assert max(errors.values()) < .0001
        report['cases'].append({'cursor': cursor.reshape(-1).tolist(), 'window': window,
            'canvas': array('canvas%d' % index, canvas), 'frozenErrors': errors,
            'inputs': {k: (int(v) if k == 'step_image_size:0' else array('in%d_%s' % (index, k.split(':')[0]), v)) for k, v in inputs.items()},
            'outputs': {k: array('out%d_%s' % (index, k.split(':')[0]), v) for k, v in expected.items()}})
    if ROUGH:
        np.random.seed(42)
        # Capture Python's integer RNG draws so JS can consume the same choices,
        # including retries/fallback, without assuming matching RNG algorithms.
        randint = np.random.randint
        for index, (current, source_image) in enumerate([([.5, .5], photo), ([.01, .98], photo), ([.5, .5], np.ones_like(photo))]):
            draws = []
            def capture(low, high, size):
                values = randint(low, high, size=size)
                draws.extend(((values.reshape(-1) - low + .25) / (high - low)).tolist())
                return values
            current = np.asarray(current, np.float32).reshape(1, 1, 2)
            with patch('numpy.random.randint', side_effect=capture):
                moved = move_cursor_to_undrawn(current, source_image, 128, .3, .9)
            report['moves'].append({'cursor': current.reshape(-1).tolist(), 'random': draws,
                'blank': index == 2, 'expected': moved.reshape(-1).tolist()})
        np.random.seed(42)
        result = sample_rough(sess, model, photo, np.array([[[[.5, .5]]]], np.float32), size, 48, [128] * 10, False, paste, 12, .3, .9)
        params, raw, soft, canvas, windows, cursors, lengths = result
        report['sequence'] = {'params': params[0], 'canvas': array('reference_canvas', canvas[0]),
                              'lengths': lengths, 'cursors': [[.5, .5]] + [c.reshape(-1).tolist() for c in cursors]}
    else:
        result = sample(sess, model, photo, np.array([[[[.5, .5]]]], np.float32), size, 48, 100, False, paste)
        params, raw, soft, canvas, windows = result
        report['sequence'] = {'params': params[0], 'canvas': array('reference_canvas', canvas[0])}
    Image.fromarray(np.round((1 - canvas[0]) * 255).astype(np.uint8)).save(str(OUT / 'reference.png'))
with open(str(OUT / 'reference.json'), 'w') as f:
    json.dump(report, f)
print('Wrote', OUT, flush=True)
