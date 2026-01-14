# Based on
# https://huggingface.co/spaces/SmilingWolf/wd-tagger/blob/main/app.py.
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import huggingface_hub
import numpy as np
import onnxruntime
from PIL import Image as PilImage
from onnxruntime import InferenceSession
from huggingface_hub.errors import EntryNotFoundError

import auto_captioning.captioning_thread as captioning_thread
from auto_captioning.auto_captioning_model import AutoCaptioningModel
from utils.image import Image

KAOMOJIS = ['0_0', '(o)_(o)', '+_+', '+_-', '._.', '<o>_<o>', '<|>_<|>', '=_=',
            '>_<', '3_3', '6_9', '>_o', '@_@', '^_^', 'o_o', 'u_u', 'x_x',
            '|_|', '||_||']


@dataclass(frozen=True)
class PreprocessConfig:
    size: tuple[int, int]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


def get_tags_to_exclude(tags_to_exclude_string: str) -> list[str]:
    if not tags_to_exclude_string.strip():
        return []
    tags = re.split(r'(?<!\\),', tags_to_exclude_string)
    tags = [tag.strip().replace(r'\,', ',') for tag in tags]
    return tags


MODEL_FILENAME_BY_REPO = {
    'deepghs/ml-danbooru-onnx': 'ml_caformer_m36_dec-5-97527.onnx',
    'deepghs/pixai-tagger-v0.9-onnx': 'model.onnx',
}
TAGS_FILENAME_BY_REPO = {
    'deepghs/ml-danbooru-onnx': 'tags.csv',
    'deepghs/pixai-tagger-v0.9-onnx': 'selected_tags.csv',
}


class WdTaggerModel:
    def __init__(self, model_id: str):
        model_path = self._resolve_model_path(model_id)
        tags_path = self._resolve_tags_path(model_id)
        self.preprocess = self._resolve_preprocess(model_id)
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
                if category is None:
                    self.general_tags_indices.append(index)
                elif category == '9':
                    self.rating_tags_indices.append(index)
                elif category == '0':
                    self.general_tags_indices.append(index)
                elif category == '4':
                    self.character_tags_indices.append(index)
        self.output_name = self._get_output_name()

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

    def _get_output_name(self) -> str:
        outputs = self.inference_session.get_outputs()
        output_names = [output.name for output in outputs]
        for output in outputs:
            if output.name.lower() == 'prediction':
                return output.name
        for output in outputs:
            if 'prob' in output.name.lower():
                return output.name
        tag_count = len(self.tags)
        for output in outputs:
            shape = output.shape
            if not shape:
                continue
            last_dim = shape[-1]
            if isinstance(last_dim, int) and last_dim == tag_count:
                return output.name
        return output_names[0]

    @staticmethod
    def _resolve_repo_file(model_id: str, candidates: list[str],
                           extension: str) -> str:
        model_path = Path(model_id)
        if model_path.is_dir():
            for candidate in candidates:
                candidate_path = model_path / candidate
                if candidate_path.is_file():
                    return str(candidate_path)
            for candidate_path in sorted(model_path.glob(f'*{extension}')):
                return str(candidate_path)
        for candidate in candidates:
            try:
                return huggingface_hub.hf_hub_download(model_id,
                                                       filename=candidate)
            except EntryNotFoundError:
                continue
        repo_files = huggingface_hub.list_repo_files(model_id)
        for repo_file in sorted(repo_files):
            if repo_file.endswith(extension):
                return huggingface_hub.hf_hub_download(model_id,
                                                       filename=repo_file)
        raise FileNotFoundError(
            f'Could not locate {extension} file for model {model_id}')

    @classmethod
    def _resolve_model_path(cls, model_id: str) -> str:
        model_id_lower = model_id.lower()
        candidates = []
        if model_id_lower in MODEL_FILENAME_BY_REPO:
            candidates.append(MODEL_FILENAME_BY_REPO[model_id_lower])
        candidates.append('model.onnx')
        return cls._resolve_repo_file(model_id, candidates, '.onnx')

    @classmethod
    def _resolve_tags_path(cls, model_id: str) -> str:
        model_id_lower = model_id.lower()
        candidates = []
        if model_id_lower in TAGS_FILENAME_BY_REPO:
            candidates.append(TAGS_FILENAME_BY_REPO[model_id_lower])
        candidates.extend(['selected_tags.csv', 'tags.csv'])
        return cls._resolve_repo_file(model_id, candidates, '.csv')

    @classmethod
    def _resolve_preprocess(cls, model_id: str) -> PreprocessConfig | None:
        model_path = Path(model_id)
        preprocess_path = model_path / 'preprocess.json'
        if preprocess_path.is_file():
            preprocess_path = preprocess_path
        else:
            try:
                preprocess_path = huggingface_hub.hf_hub_download(
                    model_id, filename='preprocess.json')
            except EntryNotFoundError:
                return None
        with open(preprocess_path, 'r') as preprocess_file:
            preprocess_data = json.load(preprocess_file)
        size = mean = std = None
        for stage in preprocess_data.get('stages', []):
            if stage.get('type') == 'resize':
                size_value = stage.get('size')
                if isinstance(size_value, list) and len(size_value) == 2:
                    size = (int(size_value[0]), int(size_value[1]))
            if stage.get('type') == 'normalize':
                mean_value = stage.get('mean')
                std_value = stage.get('std')
                if isinstance(mean_value, list) and len(mean_value) == 3:
                    mean = tuple(float(value) for value in mean_value)
                if isinstance(std_value, list) and len(std_value) == 3:
                    std = tuple(float(value) for value in std_value)
        if not size or not mean or not std:
            return None
        return PreprocessConfig(size=size, mean=mean, std=std)

    def generate_tags(self, image_array: np.ndarray,
                      wd_tagger_settings: dict) -> tuple[tuple, tuple]:
        input_name = self.inference_session.get_inputs()[0].name
        output_name = self.output_name
        probabilities = self.inference_session.run(
            [output_name], {input_name: image_array})[0][0].astype(np.float32)
        if ('logit' in output_name.lower()
                or np.min(probabilities) < 0
                or np.max(probabilities) > 1):
            probabilities = 1 / (1 + np.exp(-probabilities))
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

    def _get_input_layout(self) -> tuple[str, int | None]:
        input_shape = self.model.inference_session.get_inputs()[0].shape
        if len(input_shape) != 4:
            return 'nhwc', None
        _, dim_1, dim_2, dim_3 = input_shape
        if dim_1 == 3 and isinstance(dim_2, int) and isinstance(dim_3, int):
            return 'nchw', dim_2
        if dim_3 == 3 and isinstance(dim_1, int) and isinstance(dim_2, int):
            return 'nhwc', dim_1
        if dim_1 == 3:
            target = dim_2 if isinstance(dim_2, int) else dim_3
            return 'nchw', target if isinstance(target, int) else None
        if dim_3 == 3:
            target = dim_1 if isinstance(dim_1, int) else dim_2
            return 'nhwc', target if isinstance(target, int) else None
        return 'nhwc', None

    def get_model_inputs(self, image_prompt: str, image: Image) -> np.ndarray:
        pil_image = self.load_image(image)
        # Add a white background to the image in case it has transparent areas.
        canvas = PilImage.new('RGBA', pil_image.size, (255, 255, 255))
        canvas.alpha_composite(pil_image)
        pil_image = canvas.convert('RGB')
        preprocess = self.model.preprocess
        if preprocess:
            canvas = pil_image.resize(preprocess.size,
                                      resample=PilImage.Resampling.BILINEAR)
            image_array = np.array(canvas, dtype=np.float32) / 255.0
            image_array = np.transpose(image_array, (2, 0, 1))
            mean = np.array(preprocess.mean, dtype=np.float32)[:, None, None]
            std = np.array(preprocess.std, dtype=np.float32)[:, None, None]
            image_array = (image_array - mean) / std
            layout, _ = self._get_input_layout()
            if layout == 'nhwc':
                image_array = np.transpose(image_array, (1, 2, 0))
            image_array = np.expand_dims(image_array, axis=0)
            return image_array
        # Pad the image to make it square.
        max_dimension = max(pil_image.size)
        canvas = PilImage.new('RGB', (max_dimension, max_dimension),
                              (255, 255, 255))
        horizontal_padding = (max_dimension - pil_image.width) // 2
        vertical_padding = (max_dimension - pil_image.height) // 2
        canvas.paste(pil_image, (horizontal_padding, vertical_padding))
        # Resize the image to the model's input dimensions.
        layout, input_dimension = self._get_input_layout()
        if input_dimension and max_dimension != input_dimension:
            input_dimensions = (input_dimension, input_dimension)
            canvas = canvas.resize(input_dimensions,
                                   resample=PilImage.Resampling.BICUBIC)
        # Convert the image to a numpy array.
        image_array = np.array(canvas, dtype=np.float32)
        # Reverse the order of the color channels.
        image_array = image_array[:, :, ::-1]
        if layout == 'nchw':
            image_array = np.transpose(image_array, (2, 0, 1))
        # Add a batch dimension.
        image_array = np.expand_dims(image_array, axis=0)
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
