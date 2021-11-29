from lib.config import cfg


class DatasetCatalog(object):
    dataset_attrs = {
        'LinemodTest': {
            'id': 'linemod',
            'data_root': 'data/linemod/{}/JPEGImages'.format(cfg.cls_type),
            'ann_file': 'data/linemod/{}/test.json'.format(cfg.cls_type),
            'split': 'test'
        },
        'LinemodTrain': {
            'id': 'linemod',
            'data_root': 'data/linemod/{}/JPEGImages'.format(cfg.cls_type),
            'ann_file': 'data/linemod/{}/train.json'.format(cfg.cls_type),
            'split': 'train'
        },
        'TlessTest': {
            'id': 'tless_test',
            'ann_file': 'data/tless/test_primesense/test.json',
            'split': 'test'
        },
        'TlessMini': {
            'id': 'tless_test',
            'ann_file': 'data/tless/test_primesense/test.json',
            'split': 'mini'
        },
        'TlessTrain': {
            'id': 'tless',
            'ann_file': 'data/tless/renders/assets/asset.json',
            'split': 'train'
        },
        'TlessAgTrain': {
            'id': 'tless_ag',
            'ann_file': 'data/tless/t-less-mix/train.json',
            'split': 'test'
        },
        'TlessAgAsTrain': {
            'id': 'tless_train',
            'ann_file': 'data/tless/train_primesense/assets/train.json',
            'split': 'train'
        },
        'LinemodOccTest': {
            'id': 'linemod',
            'data_root': 'data/occlusion_linemod/RGB-D/rgb_noseg',
            'ann_file': 'data/linemod/{}/occ.json'.format(cfg.cls_type),
            'split': 'test'
        },
        'TlessPoseTrain': {
            'id': 'tless_train',
            'ann_file': 'data/cache/tless_pose/{}/train.json'.format(cfg.cls_type),
            'split': 'train'
        },
        'TlessPoseMini': {
            'id': 'tless_test',
            'det_file': 'data/cache/tless_ct/results.json',
            'ann_file': 'data/cache/tless_pose/test.json',
            'det_gt_file': 'data/tless/test_primesense/test.json',
            'obj_id': cfg.cls_type,
            'split': 'mini'
        },
        'TlessPoseTest': {
            'id': 'tless_test',
            'det_file': 'data/cache/tless_ct/results.json',
            'ann_file': 'data/cache/tless_pose/test.json',
            'det_gt_file': 'data/tless/test_primesense/test.json',
            'obj_id': cfg.cls_type,
            'split': 'test'
        },
        'YcbTest': {
            'id': 'ycb',
            'ann_file': 'data/YCB/posedb/{}_val.pkl'.format(cfg.cls_type),
            'data_root': 'data/YCB'
        },
        'CustomTrain': {
            'id': 'custom',
            'data_root': 'data/custom',
            'ann_file': 'data/custom/train.json',
            'split': 'train'
        },
        # change to custom_test to evaluate on the test folder, change to custom for the training folder
        'CustomTest': {
            'id': 'custom',
            'data_root': 'data/custom_test',
            'ann_file': 'data/custom_test/train.json',
            'split': 'test'
        },
        'CustomGlass': {
            'id': 'custom',
            'data_root': 'data/custom_glass',
            'ann_file': 'data/custom_glass/train.json',
            'split': 'train'
        },
        # change to custom_test to evaluate on the test folder, change to custom for the training folder
        'CustomTestGlass': {
            'id': 'custom',
            'data_root': 'data/custom_test_glass',
            'ann_file': 'data/custom_test_glass/train.json',
            'split': 'test'
        },
        'CustomComplex': {
            'id': 'custom',
            'data_root': 'data/custom_cscene',
            'ann_file': 'data/custom_cscene/train.json',
            'split': 'train'
        },
        'CustomComplexVal': {
            'id': 'custom',
            'data_root': 'data/custom_cscene_val',
            'ann_file': 'data/custom_cscene_val/train.json',
            'split': 'test'
        }
        # change to custom_test to evaluate on the test folder, change to custom for the training folder

    }

    @staticmethod
    def get(name):
        attrs = DatasetCatalog.dataset_attrs[name]
        return attrs.copy()
