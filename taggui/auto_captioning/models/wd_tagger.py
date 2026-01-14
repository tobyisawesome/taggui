# Based on
# https://huggingface.co/spaces/SmilingWolf/wd-tagger/blob/main/app.py.
import csv
import re
from datetime import datetime
from pathlib import Path

import huggingface_hub
from huggingface_hub.utils import HfHubHTTPError
import numpy as np
from PIL import Image as PilImage
from onnxruntime import InferenceSession

import auto_captioning.captioning_thread as captioning_thread
from auto_captioning.auto_captioning_model import AutoCaptioningModel
from utils.image import Image

KAOMOJIS = ['0_0', '(o)_(o)', '+_+', '+_-', '._.', '<o>_<o>', '<|>_<|>', '=_=',
            '>_<', '3_3', '6_9', '>_o', '@_@', '^_^', 'o_o', 'u_u', 'x_x',
            '|_|', '||_||']


def get_tags_to_exclude(tags_to_exclude_string: str) -> list[str]:
    if not tags_to_exclude_string.strip():
        return []
    tags = re.split(r'(?<!\\),', tags_to_exclude_string)
    tags = [tag.strip().replace(r'\,', ',') for tag in tags]
    return tags


class WdTaggerModel:
    def __init__(self, model_id: str):
        model_path = self._get_model_path(model_id)
        tags_path = self._get_tags_path(model_id)
        self.inference_session = InferenceSession(model_path)
        self.tags = []
        self.rating_tags_indices = []
        self.general_tags_indices = []
        self.character_tags_indices = []
        with open(tags_path, 'r') as tags_file:
            reader = csv.DictReader(tags_file)
            for index, line in enumerate(reader):
                if 'name' in line:
                    tag = line['name']
                    category = line.get('category')
                else:
                    tag = line['tag']
                    category = None
                if tag not in KAOMOJIS:
                    tag = tag.replace('_', ' ')
                self.tags.append(tag)
                if category == '9':
                    self.rating_tags_indices.append(index)
                elif category == '0':
                    self.general_tags_indices.append(index)
                elif category == '4':
                    self.character_tags_indices.append(index)

    def _get_model_path(self, model_id: str) -> str | Path:
        local_path = Path(model_id)
        candidate_files = [
            'model.onnx',
            'ml_caformer_m36_dec-5-97527.onnx',
            'ml_caformer_m36_dec-3-80000.onnx',
            'caformer_m36-3-80000.onnx',
            'TResnet-D-FLq_ema_6-30000.onnx',
            'TResnet-D-FLq_ema_6-10000.onnx',
            'TResnet-D-FLq_ema_4-10000.onnx',
            'TResnet-D-FLq_ema_2-40000.onnx',
        ]
        if local_path.is_dir():
            for filename in candidate_files:
                model_path = local_path / filename
                if model_path.is_file():
                    return model_path
            for model_path in sorted(local_path.glob('*.onnx')):
                return model_path
        for filename in candidate_files:
            try:
                return huggingface_hub.hf_hub_download(model_id,
                                                       filename=filename)
            except HfHubHTTPError:
                continue
        raise FileNotFoundError(
            f'No ONNX model file found for "{model_id}".')

    def _get_tags_path(self, model_id: str) -> str | Path:
        local_path = Path(model_id)
        candidate_files = ['selected_tags.csv', 'tags.csv']
        if local_path.is_dir():
            for filename in candidate_files:
                tags_path = local_path / filename
                if tags_path.is_file():
                    return tags_path
        for filename in candidate_files:
            try:
                return huggingface_hub.hf_hub_download(model_id,
                                                       filename=filename)
            except HfHubHTTPError:
                continue
        raise FileNotFoundError(
            f'No tags file found for "{model_id}".')

    def generate_tags(self, image_array: np.ndarray,
                      wd_tagger_settings: dict) -> tuple[tuple, tuple]:
        input_name = self.inference_session.get_inputs()[0].name
        output_name = self.inference_session.get_outputs()[0].name
        probabilities = self.inference_session.run(
            [output_name], {input_name: image_array})[0][0].astype(np.float32)
        # Exclude the rating tags.
        tags = [tag for index, tag in enumerate(self.tags)
                if index not in self.rating_tags_indices]
        probabilities = np.array([
            probability for index, probability in enumerate(probabilities)
            if index not in self.rating_tags_indices
        ])
        tags_to_exclude = get_tags_to_exclude(
            wd_tagger_settings['tags_to_exclude'])
        tags_and_probabilities = []
        for tag, probability in zip(tags, probabilities):
            if (probability < wd_tagger_settings['min_probability']
                    or tag in tags_to_exclude):
                continue
            tags_and_probabilities.append((tag, probability))
        # Sort the tags by probability.
        tags_and_probabilities.sort(key=lambda x: x[1], reverse=True)
        tags_and_probabilities = tags_and_probabilities[
                                 :wd_tagger_settings['max_tags']]
        if tags_and_probabilities:
            tags, probabilities = zip(*tags_and_probabilities)
        else:
            tags, probabilities = (), ()
        return tags, probabilities


class WdTagger(AutoCaptioningModel):
    image_mode = 'RGBA'

    def __init__(self,
                 captioning_thread_: 'captioning_thread.CaptioningThread',
                 caption_settings: dict):
        super().__init__(captioning_thread_, caption_settings)
        self.wd_tagger_settings = self.caption_settings['wd_tagger_settings']
        self.show_probabilities = self.wd_tagger_settings['show_probabilities']

    def get_error_message(self) -> str | None:
        return None

    def get_processor(self):
        return None

    def get_model(self):
        return WdTaggerModel(self.model_id)

    def get_captioning_message(self, are_multiple_images_selected: bool,
                               captioning_start_datetime: datetime) -> str:
        if are_multiple_images_selected:
            captioning_start_datetime_string = (
                self.get_captioning_start_datetime_string(
                    captioning_start_datetime))
            return (f'Generating tags... (start time: '
                    f'{captioning_start_datetime_string})')
        return 'Generating tags...'

    def get_model_inputs(self, image_prompt: str, image: Image) -> np.ndarray:
        pil_image = self.load_image(image)
        # Add a white background to the image in case it has transparent areas.
        canvas = PilImage.new('RGBA', pil_image.size, (255, 255, 255))
        canvas.alpha_composite(pil_image)
        pil_image = canvas.convert('RGB')
        # Pad the image to make it square.
        max_dimension = max(pil_image.size)
        canvas = PilImage.new('RGB', (max_dimension, max_dimension),
                              (255, 255, 255))
        horizontal_padding = (max_dimension - pil_image.width) // 2
        vertical_padding = (max_dimension - pil_image.height) // 2
        canvas.paste(pil_image, (horizontal_padding, vertical_padding))
        # Resize the image to the model's input dimensions.
        input_shape = self.model.inference_session.get_inputs()[0].shape
        input_dimension = self._get_input_dimension(input_shape,
                                                    max_dimension)
        if max_dimension != input_dimension:
            input_dimensions = (input_dimension, input_dimension)
            canvas = canvas.resize(input_dimensions,
                                   resample=PilImage.Resampling.BICUBIC)
        # Convert the image to a numpy array.
        image_array = np.array(canvas, dtype=np.float32)
        # Reverse the order of the color channels (RGB -> BGR).
        image_array = image_array[:, :, ::-1]
        # Normalize pixel values to the expected [0, 1] range.
        image_array /= 255.0
        # Add a batch dimension and arrange channels if needed.
        image_array = self._prepare_input_tensor(image_array, input_shape)
        return np.ascontiguousarray(image_array)

    def _get_input_dimension(self, input_shape: list | tuple,
                             fallback_dimension: int) -> int:
        if len(input_shape) != 4:
            return fallback_dimension
        for axis in (2, 3):
            axis_value = input_shape[axis]
            if isinstance(axis_value, int) and axis_value > 3:
                return axis_value
        for axis in (1, 2, 3):
            axis_value = input_shape[axis]
            if isinstance(axis_value, int) and axis_value > 3:
                return axis_value
        return fallback_dimension

    def _prepare_input_tensor(self, image_array: np.ndarray,
                              input_shape: list | tuple) -> np.ndarray:
        image_array = np.expand_dims(image_array, axis=0)
        if len(input_shape) != 4:
            return image_array
        channel_axis = input_shape[1]
        if isinstance(channel_axis, int) and channel_axis == 3:
            return np.transpose(image_array, (0, 3, 1, 2))
        return image_array

    def generate_caption(self, model_inputs: np.ndarray,
                         image_prompt: str) -> tuple[str, str]:
        tags, probabilities = self.model.generate_tags(model_inputs,
                                                       self.wd_tagger_settings)
        caption = self.thread.tag_separator.join(tags)
        if self.show_probabilities:
            console_output_caption = self.thread.tag_separator.join(
                f'{tag} ({probability:.2f})'
                for tag, probability in zip(tags, probabilities)
            )
        else:
            console_output_caption = caption
        return caption, console_output_caption
