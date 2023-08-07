
import numpy as np
import scipy, cv2, os, sys, argparse, audio
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch, face_detection
from models import LNet as Wav2Lip
import platform
sys.path.append('third_part')
# 3dmm extraction
from third_part.face3d.util.preprocess import align_img
from third_part.face3d.util.load_mats import load_lm3d
from third_part.face3d.extract_kp_videos import KeypointExtractor
# face enhancement
from third_part.GPEN.gpen_face_enhancer import FaceEnhancement
from third_part.GFPGAN.gfpgan import GFPGANer
# expression control
from third_part.ganimation_replicate.model.ganimation import GANimationModel

from utils import audio
from utils.ffhq_preprocess import Croper
from utils.alignment_stit import crop_faces, calc_alignment_coefficients, paste_image
from utils.inference_utils import Laplacian_Pyramid_Blending_with_mask, face_detect, load_model, options, split_coeff, \
                                  trans_image, transform_semantic, find_crop_norm_ratio, load_face3d_net, exp_aus_dict
import warnings

parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--checkpoint_path', type=str,
					help='Name of saved checkpoint to load weights from', required=True)

parser.add_argument('--face', type=str,
					help='Filepath of happy_level2/image that contains faces to use', required=True)
parser.add_argument('--audio', type=str,
					help='Filepath of happy_level2/audio file to use as raw audio source', required=True)
parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.',
					default='results/result_voice.mp4')

parser.add_argument('--static', type=bool,
					help='If True, then use only first happy_level2 frame for inference', default=False)
parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)',
					default=25., required=False)

parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0],
					help='Padding (top, bottom, left, right). Please adjust to include chin at least')

parser.add_argument('--face_det_batch_size', type=int,
					help='Batch size for face detection', default=16)
parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=128)

parser.add_argument('--resize_factor', default=1, type=int,
					help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1],
					help='Crop happy_level2 to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. '
						 'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1],
					help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
						 'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

parser.add_argument('--rotate', default=False, action='store_true',
					help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip happy_level2 right by 90deg.'
						 'Use if you get a flipped result, despite feeding a normal looking happy_level2')

parser.add_argument('--nosmooth', default=False, action='store_true',
					help='Prevent smoothing face detections over a short temporal window')

args = parser.parse_args()
args.img_size = 96

# if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
# 	args.static = True


def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i: i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes


def face_detect(images):
	detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
											flip_input=False, device=device)

	batch_size = args.face_det_batch_size

	while 1:
		predictions = []
		try:
			for i in tqdm(range(0, len(images), batch_size)):
				predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
		except RuntimeError:
			if batch_size == 1:
				raise RuntimeError(
					'Image too big to run face detection on GPU. Please use the --resize_factor argument')
			batch_size //= 2
			print('Recovering from OOM error; New batch size: {}'.format(batch_size))
			continue
		break

	results = []
	pady1, pady2, padx1, padx2 = args.pads
	for rect, image in zip(predictions, images):
		if rect is None:
			cv2.imwrite('temp/faulty_frame.jpg', image)  # check this frame where the face was not detected.
			raise ValueError('Face not detected! Ensure the happy_level2 contains a face in all the frames.')

		y1 = max(0, rect[1] - pady1)
		y2 = min(image.shape[0], rect[3] + pady2)
		x1 = max(0, rect[0] - padx1)
		x2 = min(image.shape[1], rect[2] + padx2)

		results.append([x1, y1, x2, y2])

	boxes = np.array(results)
	if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
	results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

	del detector
	return results


def datagen(frames, mels, full_frames):
	img_batch, mel_batch, frame_batch, coords_batch,  full_frame_batch = [], [], [], [], []

	if args.box[0] == -1:
		if not args.static:
			face_det_results = face_detect(frames)  # BGR2RGB for CNN face detection
		else:
			face_det_results = face_detect([frames[0]])
	else:
		print('Using the specified bounding box instead of face detection...')
		y1, y2, x1, x2 = args.box
		face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

	for i, m in enumerate(mels):
		idx = 0 if args.static else i % len(frames)
		frame_to_save = frames[idx].copy()
		face, coords = face_det_results[idx].copy()
		full_frame_batch.append(full_frames[idx].copy())
		face = cv2.resize(face, (args.img_size, args.img_size))

		img_batch.append(face)
		mel_batch.append(m)
		frame_batch.append(frame_to_save)
		coords_batch.append(coords)

		if len(img_batch) >= args.wav2lip_batch_size:
			img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

			img_masked = img_batch.copy()
			img_masked[:, args.img_size // 2:] = 0

			img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
			mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

			yield img_batch, mel_batch, frame_batch, coords_batch, full_frame_batch
			img_batch, mel_batch, frame_batch, coords_batch, full_frame_batch = [], [], [], [], []

	if len(img_batch) > 0:
		img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

		img_masked = img_batch.copy()
		img_masked[:, args.img_size // 2:] = 0

		img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
		mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

		yield img_batch, mel_batch, frame_batch, coords_batch, full_frame_batch


mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))


def _load(checkpoint_path):
	if device == 'cuda':
		checkpoint = torch.load(checkpoint_path)
	else:
		checkpoint = torch.load(checkpoint_path,
								map_location=lambda storage, loc: storage)
	return checkpoint


def load_model(path):
	model = Wav2Lip()
	print("Load checkpoint from: {}".format(path))
	checkpoint = _load(path)
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)

	model = model.to(device)
	return model.eval()


def main():
	enhancer = FaceEnhancement(base_dir='checkpoints', size=512, model='GPEN-BFR-512', use_sr=False, \
							   sr_model='rrdb_realesrnet_psnr', channel_multiplier=2, narrow=1, device=device)
	restorer = GFPGANer(model_path='checkpoints/GFPGANv1.3.pth', upscale=1, arch='clean', \
						channel_multiplier=2, bg_upsampler=None)
	faces = os.listdir(args.face)
	faces_frames, out_files = [], []
	for face in faces:
		if face.split('.')[1] in ['mp4', 'mpg']:
			out = os.path.join('results', face.split('.')[0]) + '.mp4'
			out_files.append(out)
			full_face = os.path.join(args.face, face)
			faces_frames.append(full_face)

	audios = os.listdir(args.audio)
	ads = []
	for ad in audios:
		if ad.split('.')[1] in ['wav', 'm4a']:
			ad_file = os.path.join(args.audio, ad)
			ads.append(ad_file)
	faces_frames.sort()
	ads.sort()
	out_files.sort()
	for man, talk, out_vid in zip(faces_frames, ads, out_files):
		if not os.path.isfile(man):
			raise ValueError('--face argument must be a valid path to happy_level2/image file')

		elif man.split('.')[1] in ['jpg', 'png', 'jpeg']:
			full_frames = [cv2.imread(man)]
			fps = args.fps
			args.static = True
		else:
			video_stream = cv2.VideoCapture(man)
			fps = video_stream.get(cv2.CAP_PROP_FPS)

			print('Reading happy_level2 frames...')

			full_frames = []
			while 1:
				still_reading, frame = video_stream.read()
				if not still_reading:
					video_stream.release()
					break
				if args.resize_factor > 1:
					frame = cv2.resize(frame, (frame.shape[1] // args.resize_factor, frame.shape[0] // args.resize_factor))

				if args.rotate:
					frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

				y1, y2, x1, x2 = args.crop
				if x2 == -1: x2 = frame.shape[1]
				if y2 == -1: y2 = frame.shape[0]

				frame = frame[y1:y2, x1:x2]

				full_frames.append(frame)

		print("Number of frames available for inference: " + str(len(full_frames)))

		if not talk.endswith('.wav'):
			print('Extracting raw audio...')
			command = 'ffmpeg -y -i {} -strict -2 {}'.format(talk, 'temp/temp.wav')

			subprocess.call(command, shell=True)
			talk = 'temp/temp.wav'

		wav = audio.load_wav(talk, 16000)
		mel = audio.melspectrogram(wav)
		print(mel.shape)

		if np.isnan(mel.reshape(-1)).sum() > 0:
			raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

		mel_chunks = []
		mel_idx_multiplier = 80. / fps
		i = 0
		while 1:
			start_idx = int(i * mel_idx_multiplier)
			if start_idx + mel_step_size > len(mel[0]):
				mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
				break
			mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
			i += 1

		print("Length of mel chunks: {}".format(len(mel_chunks)))

		full_frames = full_frames[:len(mel_chunks)]

		batch_size = args.wav2lip_batch_size
		imgs_enhanced = []
		for idx in tqdm(range(len(full_frames)), desc='[Step 5] Reference Enhancement'):
			img = full_frames[idx]
			# pred = img
			pred, _, _ = enhancer.process(img, img, face_enhance=True, possion_blending=False)
			imgs_enhanced.append(pred)
		gen = datagen(imgs_enhanced.copy(), mel_chunks, full_frames)


		for i, (img_batch, mel_batch, frames, coords, f_frames) in enumerate(tqdm(gen,
																		total=int(
																			np.ceil(float(len(mel_chunks)) / batch_size)))):
			if i == 0:
				model = load_model(args.checkpoint_path)
				print("Model loaded")

				frame_h, frame_w = full_frames[0].shape[:-1]
				out = cv2.VideoWriter('temp/result.avi',
									  cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

			img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
			mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

			with torch.no_grad():
				pred = model(mel_batch, img_batch)

			pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

			for p, f, xf, c in zip(pred, frames, f_frames, coords):


				y1, y2, x1, x2 = c
				# p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
				#
				# f[y1:y2, x1:x2] = p
				# out.write(f)
				p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

				ff = xf.copy()
				ff[y1:y2, x1:x2] = p
				cropped_faces, restored_faces, restored_img = restorer.enhance(
					ff, has_aligned=False, only_center_face=True, paste_back=True)
				# 0,   1,   2,   3,   4,   5,   6,   7,   8,  9, 10,  11,  12,
				mm = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0]
				mouse_mask = np.zeros_like(restored_img)
				tmp_mask = enhancer.faceparser.process(restored_img[y1:y2, x1:x2], mm)[0]
				mouse_mask[y1:y2, x1:x2] = cv2.resize(tmp_mask, (x2 - x1, y2 - y1))[:, :, np.newaxis] / 255.

				height, width = ff.shape[:2]
				restored_img, ff, full_mask = [cv2.resize(x, (512, 512)) for x in
											   (restored_img, ff, np.float32(mouse_mask))]
				# BLENDING0
				img = Laplacian_Pyramid_Blending_with_mask(restored_img, ff, full_mask[:, :, 0], 10)
				pp = np.uint8(cv2.resize(np.clip(img, 0, 255), (width, height)))

				pp, orig_faces, enhanced_faces = enhancer.process(pp, xf, bbox=c, face_enhance=False, possion_blending=True)
				out.write(pp)

		out.release()

		command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(talk, 'temp/result.avi', out_vid)
		subprocess.call(command, shell=platform.system() != 'Windows')


if __name__ == '__main__':
	main()

#不需要GPF注释line299.331-351  添加327-330   开头LNet--->Wav2Lip,且将model 的init修改