
class ComponentsFactory(object):
    def __getitem__(self, key):
        if key=="preprocessor":
            from components.insights.preprocessor import main as start_preprocessor
            return start_preprocessor
        elif key=="mongo_insights":
            from components.insights.mongo_insights import main as start_mongo
            return start_mongo
        elif key=="logger":
            from components.logger import main as start_logger
            return start_logger
        elif key=="mbti_classifier":
            from components.classifiers.c_mbti import main as start_mbti
            return start_mbti
        elif key=="ibm_ocean":
            from components.classifiers.c_ibm_ocean import main as start_ibm_ocean
            return start_ibm_ocean
        elif key=="ibm_vision":
            from components.classifiers.c_ibm_images import main as start_ibm_images
            return start_ibm_images
        elif key=="ibm_nlu":
            from components.classifiers.c_ibm_nlu import main as start_ibm_nlu
            return start_ibm_nlu
        elif key=="gender_classifier":
            from components.classifiers.c_gender import main as start_gender
            return start_gender
        elif key=="filter_db":
            from components.insights.filter_db import main as start_filter_db
            return start_filter_db
        elif key=="filter_m3":
            from components.insights.filter_m3 import main as start_filter_m3
            return start_filter_m3
        elif key=="age_torch":
            from components.classifiers.c_age_torch import main as start_age_torch
            return start_age_torch
        elif key=="img_torch":
            from components.classifiers.c_image_torch import main as start_image_torch
            return start_image_torch
        elif key=="mimosa_clustering":
            from components.clustering.mimosa_clustering import main as start_mimosa_clustering
            return start_mimosa_clustering
        elif key=="mimosa_clustering_pca":
            from components.clustering.mimosa_clustering_pca import main as start_mimosa_clustering_pca
            return start_mimosa_clustering_pca
        elif key=="sklearn_clustering_pca":
            from components.clustering.sklearn_clustering_pca import main as start_clustering
            return start_clustering
        elif key=="personas_generator":
            from components.clustering.personas_generator import main as start_pers
            return start_pers
        elif key=="tapoi_interests":
            from components.classifiers.interests_finder import main as start_interests
            return start_interests
        elif key=="twitter_collector":
            from components.insights.twitter_collector import main as start_tw_coll
            return start_tw_coll
        else:
            raise ValueError(f'not known: {key}')
