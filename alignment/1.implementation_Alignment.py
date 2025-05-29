#For PM4PY

import os
import tensorflow as tf

my_path = os.path.abspath('') 
parent_path = os.chdir("..")

import pm4py
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.conformance.alignments.petri_net import algorithm
import pandas as pd
import numpy
import datetime
from pm4py.util import exec_utils
import pkgutil
from pm4py.algo.conformance.alignments.edit_distance import algorithm as logs_alignments


datas = ['Small', 'Medium', 'Large', 'Huge', 'Wide', 'credit-card', 'mccloud',
         'Road_Traffic', 'b17'] 
rate = ['1.00'] # change
cost = list()

for data in datas:
    
    # Aligner.TR
    for r in rate:
    ##  os.path.join(parent_path,"input", "encoded_anomaly", data + ".csv")
        before = pd.read_csv(os.path.join(parent_path,"input", "encoded_anomaly", data + ".csv"))
        after = pd.read_csv(os.path.join(parent_path,"input", "encoded_anomaly", data + "_"+ r + ".csv")  )
    ##
        before = before[['Case', 'Activity', "Complete.Timestamp", "Event"]]
        before = before.rename(columns={"Case": "case:concept:name", "Activity": "concept:name",
                            "Complete.Timestamp": "time:timestamp", "Event": "Resource"})
        after = after[['Case', 'Activity', "Timestamp", "type_res_trace"]]
        after = after.rename(columns={"Case": "case:concept:name", "Activity": "concept:name",
                            "Timestamp": "time:timestamp", "type_res_trace": "label"})

        clean = after[after.label.isnull() ]
        # anomaly = after[~after.label.isnull() ]
        clean = clean.reset_index(drop=True)
        # anomaly = anomaly.reset_index(drop=True)
        anomaly = after.reset_index(drop=True)
        
        
        clean = clean[["case:concept:name", "concept:name","time:timestamp" ]]
        anomaly = anomaly[["case:concept:name", "concept:name","time:timestamp" ]]

        correct_align = before[before["case:concept:name"].isin( anomaly["case:concept:name"].unique()) ]
        correct_align = correct_align[["case:concept:name", "concept:name","time:timestamp" ]]

        anomalyid = anomaly["case:concept:name"].unique()

        log = log_converter.apply(clean)
        log2 = log_converter.apply(anomaly)


        start= datetime.datetime.now()
        net, initial_marking, final_marking = inductive_miner.apply(log)

        alignments = algorithm.apply_log(log2, net, initial_marking, final_marking)


        end= datetime.datetime.now()
        
        if r == '1.00':
            cost.append(float((end-start).total_seconds())/60)
        cases = anomaly['case:concept:name'].unique()
        length = len(cases)
        w=0
        case_l = list()
        case2_l = list()
        act3_l = list()

        score=0
        for i in alignments:
            if i is None:
                print("None!!")
                #print(w)


            else:
                act_l = list()
                act2_l = list()
                org = correct_align[correct_align["case:concept:name"].isin([anomalyid[w]])]
                act_org = org["concept:name"].values.tolist()
                caseid = anomalyid[w]
                for k in i['alignment']:
                    if (k[1] != None) and (k[1] != '>>'):
                        act2_l.append(k[1])
                        case2_l.append(caseid)
                        act3_l.append(k[1])
                    if k[0] != '>>':
                        act_l.append(k[0])
                        case_l.append(caseid)
                w = w + 1
                if act_org != act2_l:
                    score = score + 1




        d = {'Case':case2_l,'Activity':act3_l}
        df = pd.DataFrame(d)

        # print(1-score/length)
        os.chdir("~\\log_repairing_extension\\alignment\\result_trd")
        df.to_csv(data+ "_" + r + ".csv", index= False)
        


    # Aligner.ED
    for r in rate:
    ##
        before = pd.read_csv(os.path.join(parent_path,"input", "encoded_anomaly", data + ".csv"))
        after = pd.read_csv(os.path.join(parent_path,"input", "encoded_anomaly", data + "_"+ r + ".csv")  )
    ##
    ##
        before = before[['Case', 'Activity', "Complete.Timestamp", "Event"]]
        before = before.rename(columns={"Case": "case:concept:name", "Activity": "concept:name",
                            "Complete.Timestamp": "time:timestamp", "Event": "Resource"})
        after = after[['Case', 'Activity', "Timestamp", "type_res_trace"]]
        after = after.rename(columns={"Case": "case:concept:name", "Activity": "concept:name",
                            "Timestamp": "time:timestamp", "type_res_trace": "label"})

        
        clean = after[after.label.isnull() ]
        clean = clean.reset_index(drop=True)
        anomaly = after.reset_index(drop=True)

        clean = clean[["case:concept:name", "concept:name","time:timestamp" ]]
        anomaly = anomaly[["case:concept:name", "concept:name","time:timestamp" ]]

        correct_align = before[before["case:concept:name"].isin( anomaly["case:concept:name"].unique()) ]
        correct_align = correct_align[["case:concept:name", "concept:name","time:timestamp" ]]

        anomalyid = anomaly["case:concept:name"].unique()

        log = log_converter.apply(clean)
        log2 = log_converter.apply(anomaly)


        start= datetime.datetime.now()
        parameters = {}
        alignments = logs_alignments.apply(log2, log, parameters=parameters)
        end= datetime.datetime.now()
        
        if r == '1.00':
            cost.append(float((end-start).total_seconds())/60)

        cases = anomaly['case:concept:name'].unique()
        length = len(cases)
        w=0
        case_l = list()
        case2_l = list()
        act3_l = list()

        score=0
        for i in alignments:
            if i is None:
                print("None!!")
                #print(w)


            else:
                act_l = list()
                act2_l = list()
                org = correct_align[correct_align["case:concept:name"].isin([anomalyid[w]])]
                act_org = org["concept:name"].values.tolist()
                caseid = anomalyid[w]
                for k in i['alignment']:
                    if (k[1] != None) and (k[1] != '>>'):
                        act2_l.append(k[1])
                        case2_l.append(caseid)
                        act3_l.append(k[1])
                    if k[0] != '>>':
                        act_l.append(k[0])
                        case_l.append(caseid)
                w = w + 1
                if act_org != act2_l:
                    score = score + 1


        d = {'Case':case2_l,'Activity':act3_l}
        df = pd.DataFrame(d)

        os.chdir("~\\log_repairing_extension\\alignment\\result_edit")
        df.to_csv(data+ "_" + r + ".csv", index= False)
    
print(cost)
