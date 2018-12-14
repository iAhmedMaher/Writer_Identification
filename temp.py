import Preprocessing as pre
import FeatureExtraction as fe

if __name__ == '__main__':
    pre.store_all_texture_blocks(log_filename='texture_blocks2.txt')
    print("Finished")
    fe.store_all_feature_vectors(log_filename='CSLBCoP_feature_vectors2.txt', method='CSLBCoP')
    print("Finished")
    fe.store_all_feature_vectors(log_filename='LBP_feature_vectors2.txt', method='LBP')
    print("Finished")
    fe.store_all_feature_vectors(log_filename='LPQ_feature_vectors2.txt', method='LPQ')
    print("Finished")

