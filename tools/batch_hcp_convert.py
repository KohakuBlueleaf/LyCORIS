import os
import math
import argparse
from typing import List
from collections import defaultdict

from hcpdiff.ckpt_manager import auto_manager


class LoraConverter(object):

    com_name_unet = [
        'down_blocks', 'up_blocks', 'mid_block', 'transformer_blocks', 'to_q',
        'to_k', 'to_v', 'to_out', 'proj_in', 'proj_out', 'input_blocks',
        'middle_block', 'output_blocks'
    ]
    com_name_TE = [
        'self_attn', 'q_proj', 'v_proj', 'k_proj', 'out_proj', 'text_model'
    ]
    prefix_unet = 'lora_unet_'
    prefix_TE = 'lora_te_'

    def __init__(self):
        self.com_name_unet_tmp = [
            x.replace('_', '%') for x in self.com_name_unet
        ]
        self.com_name_TE_tmp = [x.replace('_', '%') for x in self.com_name_TE]

    def convert_from_webui(self,
                           state,
                           network_type='lora',
                           auto_scale_alpha=False,
                           sdxl=False):
        assert network_type in ['lora', 'plugin']
        if not sdxl:
            sd_unet = self.convert_from_webui_(
                state,
                network_type=network_type,
                prefix=self.prefix_unet,
                com_name=self.com_name_unet,
                com_name_tmp=self.com_name_unet_tmp)
            sd_TE = self.convert_from_webui_(state,
                                             network_type=network_type,
                                             prefix=self.prefix_TE,
                                             com_name=self.com_name_TE,
                                             com_name_tmp=self.com_name_TE_tmp)
        else:
            sd_unet = self.convert_from_webui_xl_unet_(
                state,
                network_type=network_type,
                prefix=self.prefix_unet,
                com_name=self.com_name_unet,
                com_name_tmp=self.com_name_unet_tmp)
            sd_TE = self.convert_from_webui_xl_te_(
                state,
                network_type=network_type,
                prefix=self.prefix_TE_xl_clip_B,
                com_name=self.com_name_TE,
                com_name_tmp=self.com_name_TE_tmp)
            sd_TE2 = self.convert_from_webui_xl_te_(
                state,
                network_type=network_type,
                prefix=self.prefix_TE_xl_clip_bigG,
                com_name=self.com_name_TE,
                com_name_tmp=self.com_name_TE_tmp)
            sd_TE.update(sd_TE2)
        if auto_scale_alpha and network_type == 'lora':
            sd_unet = self.alpha_scale_from_webui(sd_unet)
            sd_TE = self.alpha_scale_from_webui(sd_TE)
        return {network_type: sd_TE}, {network_type: sd_unet}

    def convert_to_webui(self,
                         sd_unet,
                         sd_TE,
                         network_type='lora',
                         auto_scale_alpha=False,
                         sdxl=False):
        assert network_type in ['lora', 'plugin']
        sd_unet = self.convert_to_webui_(sd_unet,
                                         network_type=network_type,
                                         prefix=self.prefix_unet)
        if sdxl:
            sd_TE = self.convert_to_webui_xl_(sd_TE,
                                              network_type=network_type,
                                              prefix=self.prefix_TE)
        else:
            sd_TE = self.convert_to_webui_(sd_TE,
                                           network_type=network_type,
                                           prefix=self.prefix_TE)
        sd_unet.update(sd_TE)
        if auto_scale_alpha and network_type == 'lora':
            sd_unet = self.alpha_scale_to_webui(sd_unet)
        return sd_unet

    def convert_from_webui_(self, state, network_type, prefix, com_name,
                            com_name_tmp):
        state = {k: v for k, v in state.items() if k.startswith(prefix)}
        prefix_len = len(prefix)
        sd_covert = {}
        for k, v in state.items():
            model_k, lora_k = k[prefix_len:].split('.', 1)
            model_k = self.replace_all(model_k, com_name,
                                       com_name_tmp).replace('_', '.').replace(
                                           '%', '_')
            if lora_k == 'alpha' or network_type == 'plugin':
                sd_covert[f'{model_k}.___.{lora_k}'] = v
            else:
                sd_covert[f'{model_k}.___.layer.{lora_k}'] = v
        return sd_covert

    def convert_to_webui_(self, state, network_type, prefix):
        sd_covert = {}
        for k, v in state.items():
            if network_type == 'plugin' or 'alpha' in k or 'scale' in k:
                separator = '.___.'
            else:
                separator = '.___.layer.'
            model_k, lora_k = k.split(separator, 1)
            sd_covert[f"{prefix}{model_k.replace('.', '_')}.{lora_k}"] = v
        return sd_covert

    def convert_to_webui_xl_(self, state, network_type, prefix):
        sd_convert = {}
        for k, v in state.items():
            if network_type == 'plugin' or 'alpha' in k or 'scale' in k:
                separator = '.___.'
            else:
                separator = '.___.layer.'
            model_k, lora_k = k.split(separator, 1)
            new_k = f"{prefix}{model_k.replace('.', '_')}.{lora_k}"
            if 'clip' in new_k:
                new_k = new_k.replace(
                    '_clip_B', '1') if 'clip_B' in new_k else new_k.replace(
                        '_clip_bigG', '2')
            sd_convert[new_k] = v
        return sd_convert

    def convert_from_webui_xl_te_(self, state, network_type, prefix, com_name,
                                  com_name_tmp):
        state = {k: v for k, v in state.items() if k.startswith(prefix)}
        sd_covert = {}
        prefix_len = len(prefix)

        for k, v in state.items():
            model_k, lora_k = k[prefix_len:].split('.', 1)
            model_k = self.replace_all(model_k, com_name,
                                       com_name_tmp).replace('_', '.').replace(
                                           '%', '_')
            if prefix == 'lora_te1_':
                model_k = f'clip_B.{model_k}'
            else:
                model_k = f'clip_bigG.{model_k}'

            if lora_k == 'alpha' or network_type == 'plugin':
                sd_covert[f'{model_k}.___.{lora_k}'] = v
            else:
                sd_covert[f'{model_k}.___.layer.{lora_k}'] = v
        return sd_covert

    def convert_from_webui_xl_unet_(self, state, network_type, prefix,
                                    com_name, com_name_tmp):
        # Down:
        # 4 -> 1, 0  4 = 1 + 3 * 1 + 0
        # 5 -> 1, 1  5 = 1 + 3 * 1 + 1
        # 7 -> 2, 0  7 = 1 + 3 * 2 + 0
        # 8 -> 2, 1  8 = 1 + 3 * 2 + 1

        # Up
        # 0 -> 0, 0  0 = 0 * 3 + 0
        # 1 -> 0, 1  1 = 0 * 3 + 1
        # 2 -> 0, 2  2 = 0 * 3 + 2
        # 3 -> 1, 0  3 = 1 * 3 + 0
        # 4 -> 1, 1  4 = 1 * 3 + 1
        # 5 -> 1, 2  5 = 1 * 3 + 2

        down = {
            '4': [1, 0],
            '5': [1, 1],
            '7': [2, 0],
            '8': [2, 1],
        }
        up = {
            '0': [0, 0],
            '1': [0, 1],
            '2': [0, 2],
            '3': [1, 0],
            '4': [1, 1],
            '5': [1, 2],
        }

        import re

        m = []

        def match(key, regex_text):
            regex = re.compile(regex_text)
            r = re.match(regex, key)
            if not r:
                return False

            m.clear()
            m.extend(r.groups())
            return True

        state = {k: v for k, v in state.items() if k.startswith(prefix)}
        sd_covert = {}
        prefix_len = len(prefix)
        for k, v in state.items():
            model_k, lora_k = k[prefix_len:].split('.', 1)

            model_k = self.replace_all(model_k, com_name,
                                       com_name_tmp).replace('_', '.').replace(
                                           '%', '_')

            if match(model_k, r'input_blocks.(\d+).1.(.+)'):
                new_k = (f'down_blocks.{down[m[0]][0]}.attentions'
                         f'.{down[m[0]][1]}.{m[1]}')
            elif match(model_k, r'middle_block.1.(.+)'):
                new_k = f'mid_block.attentions.0.{m[0]}'
                pass
            elif match(model_k, r'output_blocks.(\d+).(\d+).(.+)'):
                new_k = (f'up_blocks.{up[m[0]][0]}.attentions'
                         f'.{up[m[0]][1]}.{m[2]}')
            else:
                raise NotImplementedError

            if lora_k == 'alpha' or network_type == 'plugin':
                sd_covert[f'{new_k}.___.{lora_k}'] = v
            else:
                sd_covert[f'{new_k}.___.layer.{lora_k}'] = v

        return sd_covert

    @staticmethod
    def replace_all(data: str, srcs: List[str], dsts: List[str]):
        for src, dst in zip(srcs, dsts):
            data = data.replace(src, dst)
        return data

    @staticmethod
    def alpha_scale_from_webui(state):
        # Apply to "lora_down" and "lora_up" respectively to prevent overflow
        for k, v in state.items():
            if 'lora_up' in k:
                state[k] = v * math.sqrt(v.shape[1])
            elif 'lora_down' in k:
                state[k] = v * math.sqrt(v.shape[0])
        return state

    @staticmethod
    def alpha_scale_to_webui(state):
        for k, v in state.items():
            if 'lora_up' in k:
                state[k] = v * math.sqrt(v.shape[1])
            elif 'lora_down' in k:
                state[k] = v * math.sqrt(v.shape[0])
        return state


def save_and_print_path(sd, path):
    os.makedirs(args.dump_path, exist_ok=True)
    ckpt_manager._save_ckpt(sd, save_path=path)
    print('Saved to:', path)


def get_unet_te_pairs(lora_path):
    file_pairs = defaultdict(lambda: {'TE': None, 'unet': None})
    for filename in os.listdir(lora_path):
        if filename.endswith('.safetensors'):
            parts = os.path.splitext(filename)[0].split('-')
            prefix, name = parts[0], '-'.join(parts[1:])

            if 'text_encoder' in prefix:
                file_pairs[name]['TE'] = os.path.join(
                    args.lora_path, filename)
            elif 'unet' in prefix:
                file_pairs[name]['unet'] = os.path.join(
                    args.lora_path, filename)
    return file_pairs


def get_network_type(sd_unet, sd_TE):
    if 'lora' in sd_unet.keys() and 'lora' in sd_TE.keys():
        return 'lora'
    elif 'plugin' in sd_unet.keys() and 'plugin' in sd_TE.keys():
        return 'plugin'
    else:
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert LoRA models.")
    parser.add_argument(
        "--lora_path",
        required=True,
        type=str,
        help="Path to the LoRA or folder containing LoRA models.")
    parser.add_argument("--lora_path_TE",
                        type=str,
                        help="Path to the HCP Text Encoder LoRA.")
    parser.add_argument("--dump_path",
                        required=True,
                        type=str,
                        help="Path to save the converted state dict.")
    parser.add_argument("--from_webui",
                        action="store_true",
                        help="Convert from webui format.")
    parser.add_argument("--save_network_type",
                        type=str,
                        help="Specify the network type for conversion.")
    parser.add_argument("--to_webui",
                        action="store_true",
                        help="Convert to webui format.")
    parser.add_argument("--output_prefix",
                        default="",
                        type=str,
                        help="Prefix for output filenames.")
    parser.add_argument("--auto_scale_alpha",
                        action="store_true",
                        help="Automatically scale alpha.")
    parser.add_argument("--sdxl",
                        action="store_true",
                        help="Enable SDXL conversion.")
    args = parser.parse_args()

    converter = LoraConverter()

    if os.path.isdir(args.lora_path):
        ckpt_manager = auto_manager('.safetensors')()
        if args.from_webui:
            for filename in os.listdir(args.lora_path):
                if filename.endswith('.safetensors'):
                    file_path = os.path.join(args.lora_path, filename)
                    print(f'Converting {file_path}')
                    state = ckpt_manager.load_ckpt(file_path)
                    sd_TE, sd_unet = converter.convert_from_webui(
                        state,
                        network_type=args.save_network_type,
                        auto_scale_alpha=args.auto_scale_alpha,
                        sdxl=args.sdxl)

                    TE_path = os.path.join(args.dump_path,
                                           'text_encoder-' + filename)
                    unet_path = os.path.join(args.dump_path,
                                             'unet-' + filename)
                    save_and_print_path(sd_TE, TE_path)
                    save_and_print_path(sd_unet, unet_path)

        elif args.to_webui:
            file_pairs = get_unet_te_pairs(args.lora_path)

            for name, paths in file_pairs.items():
                if paths['TE'] and paths['unet']:
                    sd_unet = ckpt_manager.load_ckpt(paths['unet'])
                    sd_TE = ckpt_manager.load_ckpt(paths['TE'])
                    network_type = get_network_type(sd_unet, sd_TE)
                    if network_type is None:
                        print('no saved lora/lycoris found, skip')
                        continue
                    print(f'Converting pair: {paths["TE"]} and {paths["unet"]}'
                          f' with key "{network_type}"')
                    state = converter.convert_to_webui(
                        sd_unet[network_type],
                        sd_TE[network_type],
                        network_type=network_type,
                        auto_scale_alpha=args.auto_scale_alpha,
                        sdxl=args.sdxl)

                    output_path = os.path.join(
                        args.dump_path,
                        f'{args.output_prefix}-{name}.safetensors')
                    save_and_print_path(state, output_path)

    else:
        print('Converting LoRA model')
        ckpt_manager = auto_manager(args.lora_path)()
        if args.from_webui:
            state = ckpt_manager.load_ckpt(args.lora_path)
            sd_TE, sd_unet = converter.convert_from_webui(
                state,
                network_type=args.save_network_type,
                auto_scale_alpha=args.auto_scale_alpha,
                sdxl=args.sdxl)
            TE_path = os.path.join(
                args.dump_path,
                'text_encoder-' + os.path.basename(args.lora_path))
            unet_path = os.path.join(
                args.dump_path, 'unet-' + os.path.basename(args.lora_path))
            save_and_print_path(sd_TE, TE_path)
            save_and_print_path(sd_unet, unet_path)
        elif args.to_webui:
            sd_unet = ckpt_manager.load_ckpt(args.lora_path)
            sd_TE = ckpt_manager.load_ckpt(args.lora_path_TE)
            network_type = get_network_type(sd_unet, sd_TE)
            if network_type is None:
                print('no saved lora/lycoris found, terminating')
                exit(1)
            print(f'Converting with key "{network_type}"')
            state = converter.convert_to_webui(
                sd_unet[network_type],
                sd_TE[network_type],
                network_type=network_type,
                auto_scale_alpha=args.auto_scale_alpha,
                sdxl=args.sdxl)
            lora_name = os.path.basename(args.lora)
            if '-' in lora_name:
                lora_name = '-'.join(lora_name.split('-')[1:])
            output_path = os.path.join(args.dump_path,
                                       args.output_prefix + '-' + lora_name)
            save_and_print_path(state, output_path)
