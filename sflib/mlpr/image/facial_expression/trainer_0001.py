from .base import FacialExpressionRecognizerTrainer
from ....corpus.cohn_kanade.process import AlignedFaces


class FacialExpressionRecognizerTrainer0001(FacialExpressionRecognizerTrainer):
    def __init__(self):
        super().__init__()

        self.shapes = None
        self.images = None
        self.targets = None
        self.groups = None

        self.features = None

    def generate_features(self):
        af = AlignedFaces()
        s, i, t, g = af.get_data_for_learning()

        self.shapes = s
        self.images = i
        self.targets = t


        # グループを割り振り直す
        unique_subjects = sorted(list(set(g)))
        print ("There are %d subjects." % len(unique_subjects))
        # 5分割する
        num_groups = 50
        num_subj_in_group = len(unique_subjects) / num_groups
        subj2group = {}
        for i, subj in enumerate(unique_subjects):
            subj2group[subj] =  int(i // num_subj_in_group)
        self.groups = [subj2group[subj] for subj in g]

        # import ipdb; ipdb.set_trace()

        self.features = self.recognizer.feature_extractor.calc(
            self.images, self.shapes)

    def get_train_data(self):
        if self.features is None:
            self.generate_features()
        return self.features, self.targets

    def get_validation_data(self):
        return None

    def get_all_data_with_groups(self):
        if self.features is None:
            self.generate_features()
        return self.features, self.targets, self.groups
    
