from Ensembles import quality_diversity_scores_threshold
from Ensembles import GMM_MCA 
from Ensembles import Consensus_GMM_torch
from Ensembles import Voting_MM_GMM_torch
import numpy as np

def do_ensemble (data, base_clustering, consensus_function, iterations, ensembel_selection, alpha=0.5, threshold='mean', n_max_clusters='mean'):
    
    #Use clustering Ensembel Selection
    if ensembel_selection=='yes':
        subset_base_c=quality_diversity_scores_threshold.ensembles_above_threshold(base_clustering, data,alpha, threshold)
        
    else:
        subset_base_c=base_clustering
        
    print('Number of base clusterings to be part in the Ensemble:', len(subset_base_c))
        
    #Find max number of k    
    if n_max_clusters=='mean':
        if len(subset_base_c)!=0:
            k_num=[]
            for i in subset_base_c:
                k_num.append(len(np.unique(i)))
            n_clusters=int(np.mean((k_num)))
    else:
        n_clusters=n_max_clusters
    
    print('Number of maximum clusters in the Ensemble:', n_clusters)
        
        
    #Call the Consensus Function
    
    if consensus_function=='GMM_MCA':
        ensemble=GMM_MCA.do_gmm_mca
    elif consensus_function=='GMM_Voting':
        ensemble=Voting_MM_GMM_torch.do_voting_gmm_torch
    elif consensus_function=='GMM_Pair':
        ensemble=Consensus_GMM_torch.do_consensus_gmm_torch
    else:
        print('Give a valid Consensus Function')
        
        
    final_ensemble=np.array(ensemble(subset_base_c, n_clusters, iterations, True, n_clusters , None, data)).astype(int)
    
    return final_ensemble