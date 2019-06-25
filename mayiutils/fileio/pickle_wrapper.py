import pickle


class PickleWrapper(object):

    @classmethod
    def loadFromFile(cls, file, mode='rb'):
        with open(file, mode) as f:
            return pickle.load(f)
    
    @classmethod
    def dump2File(cls, o, file, mode='wb'):
        '''
        把目标对象序列化到文件
        :param o: 目标对象
        :param file: 文件
        :param mode:
        :return:
        '''
        with open(file, mode) as f:
            pickle.dump(o, f)