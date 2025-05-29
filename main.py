import warnings
warnings.filterwarnings(action='ignore')
import os
import pandas as pd
import numpy as np
import time

my_path = os.path.abspath('') 

from func import discover_NBG, process_case_id_wrapper

from multiprocessing import Pool, cpu_count
from functools import partial

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data', default=None, help='log')
parser.add_argument('--num_epi', default=100, help='episodes')
parser.add_argument('--alpha', default=0, help='episodes')

args = parser.parse_args() 
dat_name = args.data
num_epi = int(args.num_epi)
alpha = float(args.alpha)


# alphas = [0, 0.01, 0.02, 0.03, 0.04, 0.05]

# File directory
anomalous_data_path = os.path.join(my_path, "encoded_anomaly", dat_name + "_1.00.csv")
output_path = os.path.join(my_path, "output")
os.makedirs(output_path, exist_ok=True)


# data load
dtype_data = {'Case': 'category', 'Activity': 'category', "type_res_trace": 'category'}
data = pd.read_csv(anomalous_data_path, usecols=['Case', 'Activity', "Timestamp", "type_res_trace"], dtype=dtype_data)
data = data.rename(columns={"Case": "case:concept:name", "Activity": "concept:name",
                           "Timestamp": "time:timestamp", "type_res_trace": "label"}, copy=False)

# Add 'Start' and 'End' activity
data['order'] = data.groupby('case:concept:name', observed=True).cumcount() + 1

start_df_orig = data[data['order'] == 1].copy() 
start_df_orig.loc[:, 'order'] = 0
start_df_orig.loc[:, 'concept:name'] = "Start"

end_indices = data.groupby('case:concept:name', observed=True)['order'].idxmax()
end_df_orig = data.loc[end_indices].copy()
end_df_orig.loc[:, 'order'] = end_df_orig['order'] + 1
end_df_orig.loc[:, 'concept:name'] = "End"

data2 = pd.concat([data, start_df_orig, end_df_orig], ignore_index=True).sort_values(by=["case:concept:name", 'order'], kind='stable').reset_index(drop=True)


# Split normal data 
clean = data2[data2.label.isna()].reset_index(drop=True)
anomaly = data2.reset_index(drop=True) 
del data, data2, start_df_orig, end_df_orig  


# utils
actset = clean['concept:name'].unique()
filtered_actset = [item for item in actset if item not in ['Start', 'End']]

act_freq = clean['concept:name'].value_counts()

if not act_freq.empty:
    act_freq = act_freq.drop(index=['Start','End'], errors='ignore')

if not act_freq.empty:
    act_freq = act_freq.reset_index()
    act_freq.columns = ['concept:name', 'count'] if 'index' not in act_freq.columns else [act_freq.columns[0],'count']

    act_freq['prob'] = act_freq['count']/sum(act_freq['count'])
    act_freq = act_freq.set_index('concept:name').reindex(filtered_actset).reset_index()
else:
    act_freq = pd.DataFrame(columns=['concept:name', 'count', 'prob'])
    if filtered_actset: 
        act_freq['concept:name'] = filtered_actset
        act_freq = act_freq.fillna(0) 



start_time = time.time() 

NBGs= discover_NBG(clean)


if __name__ == "__main__":

    processing_start_time = time.time() 

    print('Start training P-BEAR-RL (Parallelized)')
    anomaly_nolabel = anomaly[['case:concept:name', 'concept:name', 'order']]
    caseids = anomaly_nolabel['case:concept:name'].unique().tolist() # 리스트로 변환


    traces = anomaly_nolabel.groupby('case:concept:name')['concept:name'].apply(lambda x: '>>'.join(x))
    traces = traces.rename_axis('case:concept:name').reset_index(name='variant_id')
    traces = traces.groupby('variant_id', observed=True)['case:concept:name'].apply(lambda x: list(x))
    traces = traces.rename_axis('variant_id').reset_index(name='caseids')
    casezip = [cid[0] for cid in traces['caseids']]
    anomaly_nolabel_zip = anomaly_nolabel.loc[anomaly_nolabel['case:concept:name'].isin(casezip)].reset_index(drop=True)

    func_for_pool = partial(process_case_id_wrapper,
                            p_anomaly_nolabel=anomaly_nolabel_zip,
                            p_filtered_actset=filtered_actset,
                            p_act_freq=act_freq,
                            p_actset=actset,
                            p_NBGs=NBGs,
                            p_num_epi=num_epi,
                            alpha = alpha)

    num_processes = cpu_count() 
    print(f"Using {num_processes} processes for parallel execution.")

    with Pool(processes=num_processes) as pool:
        list_of_repaired_dfs_for_cases = pool.map(func_for_pool, casezip)

    final_repaired_df_zip = pd.concat([df for df in list_of_repaired_dfs_for_cases], ignore_index=True)
    final_repaired_df = pd.DataFrame()

    for cid in range(len(casezip)):
        filtered_df =  final_repaired_df_zip.loc[final_repaired_df_zip['case:concept:name']==casezip[cid] ]
        len1 = len(filtered_df)
        filtered_df =  pd.concat([filtered_df] * len(traces['caseids'][cid]), ignore_index=True)
        filtered_df['case:concept:name']=  np.repeat(traces['caseids'][cid], len1)
        
        final_repaired_df= pd.concat([final_repaired_df, filtered_df], ignore_index=True)


    end_time = time.time() 
    total_execution_time = end_time - start_time 
    parallel_processing_execution_time = end_time - processing_start_time 

    print(f"\nTotal execution time (including NBG discovery): {total_execution_time:.4f} 초")
    print(f"Parallel P-BEAR-RL training time: {parallel_processing_execution_time:.4f} 초")
    
    
    # Save result
    output_file_path = os.path.join(output_path, dat_name + "_" + str(num_epi) + "_ppm"+ str(alpha)+ ".csv")
    final_repaired_df.to_csv(output_file_path, index=False)
    print(f"Parallel processing complete. Results saved to {output_file_path}")


