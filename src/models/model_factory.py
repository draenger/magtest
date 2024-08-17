class ModelFactory:
    def __init__(self):
        self.models = {}

    def register_model(self, name, model_class, *args, **kwargs):
        self.models[name.lower()] = (model_class, args, kwargs)

    def get_model(self, name):
        model_info = self.models.get(name.lower())
        if model_info is None:
            raise ValueError(f"Model '{name}' not found")
        model_class, args, kwargs = model_info
        return model_class(*args, **kwargs)
    
    def get_registered_models(self):
        return list(self.models.keys())
