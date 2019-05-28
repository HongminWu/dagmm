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
import pickle

coloredlogs.install()

if __name__=="__main__":
    skill = 3
    anomaly_region = .5 # (s) observation located at the forward and backward region are marked as anomalies 
    datasets_of_filtering_schemes_folder = '/home/birl_wu/baxter_ws/src/SPAI/smach_based_introspection_framework/introspection_data_folder.AC_offline_test/anomaly_detection_feature_selection_folder-18dims/'
    logger = logging.getLogger('GetDataOfSKill')
    logger.setLevel(logging.INFO)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.INFO)
    logger.addHandler(consoleHandler)

    succ_folder = glob.glob(os.path.join(
        datasets_of_filtering_schemes_folder,
        'No.* filtering scheme',
        'successful_skills',
        'skill %s'%skill,
    ))[0]

    with open(os.path.join(datasets_of_filtering_schemes_folder, 'No.0 filtering scheme', 'filtering scheme info.txt')) as f:
        lines = f.readlines()
    hz = float(lines[1].rstrip('\n')[-2:]) # extract the hz at second row
    
    list_of_mat = []
    list_of_label = []
    for csv in glob.glob(os.path.join(succ_folder, '*', '*.csv')):
        logger.info(csv)        
        df = pd.read_csv(csv, sep=',')
        # Exclude 1st column which is time index
        mat = df.values[:, 1:]
        label = np.ones(mat.shape[0]) # 1 represents the norminal
        list_of_mat.append(mat)
        list_of_label.append(label)


    unsucc_folder = glob.glob(os.path.join(
        datasets_of_filtering_schemes_folder,
        'No.* filtering scheme',
        'unsuccessful_skills',
        'skill %s'%skill,
    ))[0]
    
    for csv in glob.glob(os.path.join(unsucc_folder, '*', '*.csv')):
        logger.info(csv)

        anomaly_type, anomaly_gentime = pickle.load(open(os.path.join(os.path.dirname(csv), "anomaly_label_and_signal_time.pkl"), 'rb'))
        df = pd.read_csv(csv, sep=',')
        s_t = df.iloc[0, 0]
        anomaly_time = anomaly_gentime.to_sec()-s_t
        logger.warning(anomaly_type)
        logger.warning(anomaly_time)
        timesteps = df.values[:,0] - s_t
        anomalies_index = np.where((timesteps > (anomaly_time - anomaly_region)) & (timesteps < (anomaly_time + anomaly_region)))
        
        # Exclude 1st column which is time index
        mat = df.values[:, 1:]
        label = np.ones(mat.shape[0])
        label[anomalies_index[0]] = 0 # 0 represents the anomalies        
        list_of_mat.append(mat)
        list_of_label.append(label)


    # stack all the signals into an array and saved as kitting_exp_skill_?.npz
    # formatted as [feature_1, feature_2,...,label]
    data = None
    for mat, label in zip(list_of_mat, list_of_label):
        label = label.reshape(-1,1)
        data = np.hstack((mat, label)) if data is None else np.vstack((data, np.hstack((mat, label))))

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1)) # scaling the data into range(-1, 1)
    scaledData  = scaler.fit_transform(data[:,:-1]) # ignore the label column
    data[:,:-1] = scaledData      
    
    np.save('kitting_exp_skill_%s'%skill, data)
    logger.warning("data with shape of ")
    logger.warning(data.shape)
    logger.warning('data saved as kitting_exp_skill_%s.npz'%skill)   


        


