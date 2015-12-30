import numpy

from ..feature import ModelTransform


class Transform(ModelTransform):
    def transform(self, array):
        return self.model.fit_transform(numpy.array(array))
