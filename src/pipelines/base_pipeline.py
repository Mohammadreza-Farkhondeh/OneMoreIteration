from src.utils.logger import Logger


class BasePipeline(object):
    def __init__(self, data_loader, data_preprocessor, model):
        self.data_loader = data_loader
        self.data_preprocessor = data_preprocessor
        self.model = model
        self.logger = Logger().get_logger()

    def run(self, *args, **kwargs):
        data = self.data_loader.load_data(*args, **kwargs)
        data = self.data_preprocessor.preprocess_data(data, *args, **kwargs)
        self.model.train(data, *args, **kwargs)
        self.model.evaluate(data, *args, **kwargs)
        self.model.save(*args, **kwargs)
