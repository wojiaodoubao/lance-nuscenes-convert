# coding: utf-8
"""
------------------------------------------------------------------------------
Copyright 2025 jinglun
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
 http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
------------------------------------------------------------------------------
"""

import lance
from nuscenes.nuscenes import NuScenes
import pyarrow as pa
import argparse

metrics = {}


def compute_schema(nusc: NuScenes, data_root: str, compression_algo="zstd", compression_level="0"):
    scene = nusc.scene[0]
    sample_token = scene['first_sample_token']
    (_, sample_dict) = extend_sample(nusc, sample_token, data_root=data_root)

    data_table = {}
    for k, v in sample_dict.items():
        data_table[k] = [v]
    arrow_table = pa.Table.from_pydict(data_table)

    schema = arrow_table.schema
    new_fields = []
    for field in schema:
        if pa.types.is_binary(field.type):
            new_field = field.with_metadata(
                {"lance-encoding:compression": compression_algo, "lance-encoding:compression-level": compression_level})
            new_fields.append(new_field)
        else:
            new_fields.append(field)

    new_schema = pa.schema(new_fields)
    return new_schema


def convert_nuscenes_to_lance(data_root: str, version: str, lance_root: str, compression_algo="zstd",
                              compression_level="0"):
    """
    Convert nuScenes dataset into pyarrow table.

    :param data_root: The root path of nuScenes dataset.
    :param version: The version of nuScenes dataset.
    :param lance_root: The root path of lance dataset.
    :param compression_algo: The compression algorithm.
    :param compression_level: The lance compression level.
    :return: the converted pyarrow table.
    """
    nusc = NuScenes(version=version, dataroot=data_root, verbose=True)
    schema = compute_schema(nusc, data_root, compression_algo, compression_level)

    data_table = {}
    for scene in nusc.scene:
        update_metric('scene', 1)
        next_sample_token = scene['first_sample_token']
        while next_sample_token:
            update_metric('sample', 1)
            (next_sample_token, sample_dict) = extend_sample(nusc, next_sample_token, data_root=data_root)

            for k, v in sample_dict.items():
                if k not in data_table:
                    data_table[k] = []
                data_table[k].append(v)

            if get_metric('sample') % 100 == 0:
                arrow_table = pa.Table.from_pydict(data_table, schema=schema)
                lance.write_dataset(arrow_table, lance_root, mode="append")
                data_table = {}

    if len(data_table) > 0:
        arrow_table = pa.Table.from_pydict(data_table, schema=schema)
        lance.write_dataset(arrow_table, lance_root, mode="append")

    print('Statistics', metrics)
    return pa.Table.from_pydict(data_table)


def extend_sample(nusc: NuScenes, sample_token: str, data_root=None):
    sample_dict = {}
    sample = nusc.get('sample', sample_token)
    extend_sample_data(nusc, sample['data'], sample_dict, data_root)
    extend_sample_anns(nusc, sample['anns'], sample_dict)
    return sample['next'], sample_dict


def extend_sample_data(nusc: NuScenes, sample_data: dict, sample_dict: dict, data_root=None):
    for sensor, token in sample_data.items():
        sensor_data = nusc.get('sample_data', sample_data[sensor])
        for k, v in sensor_data.items():
            if k not in {'ego_pose_token', 'calibrated_sensor_token', 'filename', 'prev', 'next'} \
                    and 'token' not in k:
                sample_dict[sensor + '-' + k] = v
        # extend ego_pose.
        ego_pose_token = sensor_data['ego_pose_token']
        extend_ego_pose(nusc, ego_pose_token, sensor, sample_dict)
        # extend calibrated sensor.
        calibrated_sensor_token = sensor_data['calibrated_sensor_token']
        extend_calibrated_sensor(nusc, calibrated_sensor_token, sensor, sample_dict)
        # extend file.
        file_path = sensor_data['filename']
        extend_file(data_root + '/' + file_path, sensor, sample_dict)


def extend_ego_pose(nusc: NuScenes, ego_pose_token: str, sensor: str, sample_dict: dict):
    ego_pose = nusc.get('ego_pose', ego_pose_token)
    for k, v in ego_pose.items():
        if 'token' not in k:
            sample_dict[sensor + '-ego_pose-' + k] = v
    pass


def extend_calibrated_sensor(nusc: NuScenes, extend_calibrated_sensor_token: str, sensor: str, sample_dict: dict):
    calibrated_sensor = nusc.get('calibrated_sensor', extend_calibrated_sensor_token)
    for k, v in calibrated_sensor.items():
        if 'token' not in k:
            sample_dict[sensor + '-calibrated_sensor-' + k] = v
    pass


def extend_file(file_path: str, sensor: str, sample_dict: dict):
    with open(file_path, 'rb') as file:
        binary_data = file.read()
        sample_dict[sensor + '-file'] = binary_data
        update_metric('filecount', 1)
        update_metric('filesize', len(binary_data))
        if file_path.endswith('.pcd'):
            update_metric('filecount-pcd', 1)
            update_metric('filesize-pcd', len(binary_data))
        elif file_path.endswith('.jpg'):
            update_metric('filecount-jpg', 1)
            update_metric('filesize-jpg', len(binary_data))


def extend_sample_anns(nusc: NuScenes, sample_anns: list, sample_dict: dict):
    ann_tokens = []
    for i in range(len(sample_anns)):
        ann_token = sample_anns[i]
        ann_tokens.append(extend_sample_ann(nusc, ann_token))
    sample_dict['sample_annotations'] = ann_tokens


def extend_sample_ann(nusc: NuScenes, sample_ann_token: str):
    sample_ann = nusc.get('sample_annotation', sample_ann_token)
    annotation_dict = {}
    for k, v in sample_ann.items():
        if k not in {'instance_token', 'attribute_tokens', 'prev', 'next'} and 'token' not in k:
            annotation_dict['ann-' + k] = v
    # instance_token
    instance_token = sample_ann['instance_token']
    extend_instance(nusc, instance_token, annotation_dict)
    # attribute_tokens
    attr_tokens = sample_ann['attribute_tokens']
    extend_attribute_tokens(nusc, attr_tokens, annotation_dict)
    return annotation_dict


def extend_instance(nusc: NuScenes, instance_token: str, annotation: dict):
    instance = nusc.get('instance', instance_token)
    # parse elements.
    for k, v in instance.items():
        if k not in {'category_token'} and 'token' not in k:
            annotation[k] = v
    # parse category.
    category_token = instance['category_token']
    category = nusc.get('category', category_token)
    annotation['category'] = category


def extend_attribute_tokens(nusc: NuScenes, attr_tokens: list, annotation_dict: dict):
    array = []
    for attr_token in attr_tokens:
        array.append(extend_attribute_token(nusc, attr_token))
    annotation_dict['attributes'] = array


def extend_attribute_token(nusc: NuScenes, attr_token: str):
    attr_dict = {}
    attribute = nusc.get('attribute', attr_token)
    for k, v in attribute.items():
        if 'token' not in k:
            attr_dict[k] = v
    return attr_dict


def update_metric(key: str, delta: int):
    global metrics
    if key not in metrics:
        metrics[key] = 0
    metrics[key] += delta


def get_metric(key: str):
    global metrics
    return metrics[key]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script to transform nuScenes dataset to lance dataset.")
    parser.add_argument("nuscenes_root", help="The root path of nuScenes dataset.")
    parser.add_argument("version", help="The nuScenes dataset version.")
    parser.add_argument("lance_root", help="The root path of lance dataset.")
    parser.add_argument("--compression_algo", help="The compression algorithm of lance dataset.", default="zstd")
    parser.add_argument("--compression_level", help="The compression level of lance dataset.", default="22")

    args = parser.parse_args()

    convert_nuscenes_to_lance(data_root=args.nuscenes_root, version=args.version, lance_root=args.lance_root,
                              compression_algo=args.compression_algo, compression_level=args.compression_level)
