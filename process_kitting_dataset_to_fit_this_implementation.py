'''
TODO: Evaluate the performance of each skill
1. loading the successful skill data, and label each timestep as norminal: 
2. loading the unsuccessful skill data, and label the anomalous timesteps as abnormal: 1, otherwise, norminal: 0,
we can analysize the length of anomalous region,...
3. concatate the labelled successful and unsuccessful data, and feed into the model
'''
import coloredlogs, logging
import os, ipdb
import glob
import pandas as pd
import numpy as np

coloredlogs.install()

if __name__=="__main__":
    skill = 3
    datasets_of_filtering_schemes_folder = '/home/birl_wu/baxter_ws/src/SPAI/smach_based_introspection_framework/introspection_data_folder.AC_offline_test/anomaly_detection_feature_selection_folder-18dims/'
    logger = logging.getLogger('GetDataOfSKill')
    logger.setLevel(logging.INFO)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.INFO)
    logger.addHandler(consoleHandler)

    succ_folders = glob.glob(os.path.join(
        datasets_of_filtering_schemes_folder,
        'No.* filtering scheme',
        'successful_skills',
        'skill %s'%skill,
    ))
    
    for succ_folder in succ_folders:
        logger.info(succ_folder)
        csvs = glob.glob(os.path.join(
        succ_folder,
        '*', '*.csv',
        ))
        list_of_mat = []
        list_of_label = []
        for j in csvs:
            df = pd.read_csv(j, sep=',')
            # Exclude 1st column which is time index
            mat = df.values[:, 1:]
            label = np.zeros(mat.shape[0])
            list_of_mat.append(mat)
            list_of_label.append(label)
        ipdb.set_trace()


