

import os
os.chdir('~\\log_repairing_extension\\DeepAign')
import numpy as np
import pandas as pd

import arrow
import tensorflow as tf
from tqdm.notebook import tqdm

from deepalign import Dataset
from deepalign import fs
from deepalign.alignments import ConfNet

from deepalign import Dataset
from deepalign import fs
from deepalign.alignments import ALIGNERS
import datetime
from IPython.display import display
pd.set_option('display.max_columns', 999)
pd.set_option('display.max_rows', 999)


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")


def get_model(aligner, dataset_name, case_attributes=False, event_attributes=False):
    ea = ca = 0

    if aligner == 'confnet':
        ea = int(event_attributes)
        ca = int(case_attributes)
        model_name = f'{dataset_name}_{aligner}{ea}{ca}'
        print(model_name)
    else:
        model_name = f'{dataset_name}_{aligner}'

    dataset = Dataset(dataset_name, use_case_attributes=ca, use_event_attributes=ea)
    models = list(set([f.name.replace('_forward', '').replace('_backward', '')
                       for f in fs.get_model_files()
                       if model_name in f.name]))

    print(models)
    if aligner == 'confnet':
        aligner = ALIGNERS[aligner](dataset, use_case_attributes=ca, use_event_attributes=ea)
    else:
        aligner = ALIGNERS[aligner]()

    aligner.load(str(fs.MODEL_DIR / models[0]))

    return aligner, dataset

def display_alignment(alignment, decode=None):
    a = alignment[alignment != -1]
    a = a.reshape(2, a.shape[0] // 2)
    if decode is not None:
        a = decode(a)
    df = pd.DataFrame(a, index=['Log', 'Model'])
    return df


##########################################

cost = list()

datasets = sorted([f.name for f in fs.get_event_log_files() ])

print(datasets)


for dataset_name in datasets:
    print(dataset_name)
    start= datetime.datetime.now()
    for ea, ca in [(0, 0)]:
        start_time = arrow.now()
        dataset = Dataset(dataset_name, use_case_attributes=ca, use_event_attributes=ea)
        if ca and dataset.num_case_attributes == 0:
            continue
        confnet = ConfNet(dataset, use_case_attributes=ca, use_event_attributes=ea)
        confnet.fit(dataset, batch_size=100, epochs=50, validation_split=0.0,
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)])
        confnet.save(
            str(fs.MODEL_DIR / f'{dataset_name}_{confnet.identifier}'))


    ALIGNERS.keys()
    confnet, dataset = get_model('confnet', dataset_name, False, False)
    alignments, corrected_cases, costs = confnet.align(dataset, k=5, steps=10, delete_max=3, hot_start=True)
    decode = dict((k, v) for k, v in enumerate(dataset.encoders['name'].classes_))
    decode[-1] = decode[0]  # Padding
    decode[0] = '»'
    decode = np.vectorize(decode.get)
    end= datetime.datetime.now()
    cost.append(int((end-start).total_seconds())/60)
    c = list()
    activity = list()
    for i, case in enumerate(dataset.event_log.cases):
        df = display_alignment(alignments[i][0], decode=decode)
        align = list(list(df.iloc(0))[1])
        for k in align:
            if (k != '▶') and (k != '»') and (k != '■'):
                c.append(case.id)
                activity.append(k)
    d = {'Case': c, 'Activity': activity}
    df2 = pd.DataFrame(d)

    os.chdir('~\\log_repairing_extension\\DeepAign\\result')
    df2.to_csv( str(dataset_name) + "_DeepAlign.csv", index= False)

print(datasets)
print(cost)
