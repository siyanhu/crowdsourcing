global_settings={}

###################################################################################### used in A2,A3 modules
global_settings['fingerprint_txt_file_folder']='./Data/'
global_settings['fingerprint_txt_file_prefix']='1002'
global_settings['crowd_batch_vectors_txt_file_prefix']='crowd'
global_settings['rssi_negative_2_positive_shift']=120 #dBm
global_settings['one_meter_equals_pixels']=13.3
global_settings['estimated_max_AP_num_in_each_RP']=150
global_settings['ap_mac_address_string_length']=12  # e.g. mac=022939d083cb
global_settings['background_floor_map']='Data_FloorMap/map.jpg'
###################################################################################### used in A2,A3 modules


###################################################################################### only used in A2 module
global_settings['subset_localization_topK']=10
global_settings['subset_localization_subset_num']=30
global_settings['subset_sample_rate']=0.7  # should between 0~1
global_settings['subset_localization_subset_vector_length_min']=3
global_settings['Subset_clustering_visualization']=True
global_settings['Subset_localization_visualization']=True
global_settings['To_localize_even_with_locations']=True
###################################################################################### only used in A2 module


###################################################################################### only used in A3 module
global_settings['AP_RP_rssi_weak_threshold']=-65 #dBm
global_settings['rssi_change_up_to_?_SD']=4
global_settings['rssi_change_up_to_?_dB']=10 #dB

######## radial_basis_functional_kernel_v2()
global_settings['RBF_sigma']=1.65
global_settings['RBF_weight']=0.6

global_settings['crowd_freshed_ap_included_for_updating']=True
global_settings['prior_missing_ap_excluded_for_updating']=True
global_settings['Visualization_APdetection']=True
global_settings['BCS_updating_visualization']=True
###################################################################################### only used in A3 module