# Based on
# https://huggingface.co/spaces/SmilingWolf/wd-tagger/blob/main/app.py.
import csv
import re
from datetime import datetime
from pathlib import Path

import huggingface_hub
import numpy as np
import onnxruntime
from huggingface_hub.errors import EntryNotFoundError
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
        providers = self._get_providers()
        self.inference_session = InferenceSession(model_path,
                                                   providers=providers)
        self.tags = []
        self.rating_tags_indices = []
        self.general_tags_indices = []
        self.character_tags_indices = []
        with open(tags_path, 'r') as tags_file:
            reader = csv.DictReader(tags_file)
            for index, line in enumerate(reader):
                tag = line.get('name') or line.get('tag')
                if not tag:
                    continue
                if tag not in KAOMOJIS:
                    tag = tag.replace('_', ' ')
                self.tags.append(tag)
                category = line.get('category')
                if category == '9':
                    self.rating_tags_indices.append(index)
                elif category == '0':
                    self.general_tags_indices.append(index)
                elif category == '4':
                    self.character_tags_indices.append(index)

    @staticmethod
    def _get_model_path(model_id: str) -> str:
        preferred_model_files = {
            'deepghs/ml-danbooru-onnx': [
                'ml_caformer_m36_dec-5-97527.onnx',
                'ml_caformer_m36_dec-3-80000.onnx',
                'caformer_m36-3-80000.onnx',
                'TResnet-D-FLq_ema_6-30000.onnx',
                'TResnet-D-FLq_ema_6-10000.onnx',
                'TResnet-D-FLq_ema_4-10000.onnx',
                'TResnet-D-FLq_ema_2-40000.onnx',
            ]
        }
        return WdTaggerModel._resolve_repo_file(
            model_id, preferred_model_files.get(model_id, ['model.onnx']))

    @staticmethod
    def _get_tags_path(model_id: str) -> str:
        preferred_tag_files = {
            'deepghs/ml-danbooru-onnx': [
                'tags.csv',
            ]
        }
        return WdTaggerModel._resolve_repo_file(
            model_id, preferred_tag_files.get(model_id, ['selected_tags.csv']))

    @staticmethod
    def _get_providers() -> list[str]:
        available_providers = onnxruntime.get_available_providers()
        provider_priority = [
            'CUDAExecutionProvider',
            'TensorrtExecutionProvider',
            'ROCMExecutionProvider',
            'DmlExecutionProvider',
            'CoreMLExecutionProvider',
            'OpenVINOExecutionProvider',
            'CPUExecutionProvider',
        ]
        preferred_providers = [
            provider for provider in provider_priority
            if provider in available_providers
        ]
        return preferred_providers or available_providers

    @staticmethod
    def _resolve_repo_file(model_id: str, filenames: list[str]) -> str:
        last_error = None
        for filename in filenames:
            candidate_path = Path(model_id) / filename
            if candidate_path.is_file():
                return str(candidate_path)
            try:
                return huggingface_hub.hf_hub_download(
                    model_id, filename=filename)
            except EntryNotFoundError as error:
                last_error = error
        if last_error:
            raise last_error
        raise FileNotFoundError(
            f'No matching files found for {model_id}: {filenames}')

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
        input_layout = self._get_input_layout(input_shape)
        input_dimension = self._get_input_dimension(input_shape, input_layout)
        if input_dimension and max_dimension != input_dimension:
            input_dimensions = (input_dimension, input_dimension)
            canvas = canvas.resize(input_dimensions,
                                   resample=PilImage.Resampling.BICUBIC)
        # Convert the image to a numpy array.
        image_array = np.array(canvas, dtype=np.float32)
        # Reverse the order of the color channels.
        image_array = image_array[:, :, ::-1]
        if input_layout == 'NCHW':
            image_array = np.transpose(image_array, (2, 0, 1))
        # Add a batch dimension.
        image_array = np.expand_dims(image_array, axis=0)
        return image_array

    @staticmethod
    def _get_input_layout(input_shape: list | tuple) -> str:
        if len(input_shape) >= 4:
            if input_shape[1] == 3:
                return 'NCHW'
            if input_shape[3] == 3:
                return 'NHWC'
        return 'NHWC'

    @staticmethod
    def _get_input_dimension(input_shape: list | tuple,
                             input_layout: str) -> int | None:
        if len(input_shape) < 4:
            return None
        if input_layout == 'NCHW':
            return input_shape[2] or input_shape[3]
        return input_shape[1] or input_shape[2]

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
