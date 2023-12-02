from weka.classifiers import Classifier, Evaluation
from weka.filters import Filter
from weka.core.dataset import Instance, missing_value, create_instances_from_matrices, create_instances_from_lists
from weka.core.converters import Loader
from javabridge.jutil import JavaException
import numpy as np
from utils.dataStructure import topKunordered, Queue
from configs.envSpecific import BASELINE


class Jrip:
    def __init__(self, topK=100, folds=3, saveRuleInterval=1000):
        self.buffer = topKunordered(maxsize=topK)
        self.jrip = Classifier(
            classname="weka.classifiers.rules.JRip", options=["-F", str(folds)])
        self.jrip_ret = 0
        self.jrip_available = False  # jrip classifier is available
        self.new_data = False  # new data is available
        self.baseline = None  # lower bound of push-into buffer
        self.num_to_nominal = Filter(
            classname="weka.filters.unsupervised.attribute.NumericToNominal", options=["-R", "last"])
        self.format = Filter(classname="weka.filters.unsupervised.attribute.Add", options=[
                             "-T", "NOM", "-C", "last"])
        self.saveRuleInterval = saveRuleInterval
        self.saveRuleCount = 0

    def update_traindata(self, trajectory, ret):
        if self.baseline is None:  # e.g. we want trajectory > -200 in mountain_car
            self.baseline = ret
        if ret > self.baseline:
            for s_a in trajectory:
                self.buffer.push((ret, s_a))
            self.baseline = self.buffer.top[0]
            self.new_data = True

    def train(self):
        if self.new_data == False:
            return
        self.new_data = False
        s, a = self.buffer.trainData
        if len(a) <= 0:
            return
        train_dataset = create_instances_from_lists(
            s, a, name="generated from lists")
        train_dataset.class_is_last()
        self.num_to_nominal.inputformat(train_dataset)
        self.train_dataset = self.num_to_nominal.filter(train_dataset)
        try:
            self.jrip.build_classifier(self.train_dataset)
            self.jrip_available = True
        except JavaException as e:
            self.jrip_available = False
            print(e)

    def action(self, state):
        """ eval propositional logic policy

        Args:
            state ( [state1, state2, ...] ): a list of states

        Returns:
            _type_: _description_
        """
        if not (isinstance(state, np.ndarray) and state.dtype == np.float32):
            state = np.array(state, dtype=np.float32)
        test = create_instances_from_matrices(state)
        self.format.inputformat(test)
        test = self.format.filter(test)
        test.class_index = test.num_attributes - 1
        evaluation = Evaluation(test)
        action = evaluation.test_model(self.jrip, test).astype(int)
        return action

    def advice(self, state):
        """ As a intrinsic reward for sampled states

        Args:
            state ( [state1, state2, ...] ): a list of states

        Returns:
            _type_: _description_
        """
        if self.jrip_available == False:
            return -np.ones(len(state))
        return self.action(state)

    @property
    def rule_nums(self):
        if self.jrip_available == False:
            return 0
        return int(self.jrip.__str__().strip().split(":")[-1])

    def saveRules(self, save_name, ret):
        with open(save_name+".txt", "w") as f:
            f.write(str(self.jrip))
            f.write("\n"+str(ret))
        self.save_model(save_name+".model")

    def saveIntervalRules(self, save_name, symbolic_return):
        self.saveRuleCount += 1
        if self.saveRuleCount % self.saveRuleInterval == 0:
            self.saveRules(save_name, symbolic_return)
            self.saveRuleCount = 0

    def saveBestRules(self, save_name, best_return):
        self.saveRules(save_name+".best", best_return)

    def load(self, path):
        """ Load train data from path file, and train ripper """
        loader = Loader(classname="weka.core.converters.CSVLoader")
        train = loader.load_file(path)
        train.class_is_last()

        self.num_to_nominal.inputformat(train)
        self.train_dataset = self.num_to_nominal.filter(train)

        self.jrip = Classifier(classname="weka.classifiers.rules.JRip")
        self.jrip.build_classifier(self.train_dataset)

    def save_model(self, path):
        self.jrip.serialize(path, header=self.train_dataset)

    def load_model(self, path):
        self.jrip, self.train_dataset = Classifier.deserialize(path)


class Ripper:
    def __init__(self, cfg):
        """Should always keep jvm.start()"""
        self.getPolicy = cfg.getPolicy
        self.lastN = cfg.lastN
        self.topK = cfg.topK
        self.trajectory = None
        self.jrip = [Jrip(topK, cfg.folds, cfg.saveRuleInterval)
                     for topK in self.topK]
        self.best_jrip_idx = 0

    def add(self, state, action):
        if self.getPolicy == "topK":
            l = len(state)
            if self.trajectory is None:
                self.trajectory = [[] for _ in range(l)]
            for idx in range(l):
                self.trajectory[idx].append([state[idx], int(action[1][idx])])
        elif self.getPolicy == "lastN":  # Deparated
            self.updateLastN(state, action[0])

    def updateTopK(self, idx, ret):
        """Store Top K transition by return

        Args:
            ret (_type_): transtion return 
            transition (_type_): list of transition pairs, [[s,a],...]
        """
        for jrip in self.jrip:
            jrip.update_traindata(self.trajectory[idx], ret)
        self.trajectory[idx] = []

    def updateLastN(self, state, action):
        self.buffer.push(state, action)

    def train(self):
        for jrip in self.jrip:
            jrip.train()
