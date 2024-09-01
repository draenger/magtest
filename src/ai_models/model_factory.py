class ModelFactory:
    def __init__(self):
        self.models = {}

    def register_model(
        self,
        name,
        model_class,
        tokenizer,
        input_cost_per_million,
        output_cost_per_million,
        *args,
        **kwargs,
    ):
        self.models[name.lower()] = (
            model_class,
            tokenizer,
            input_cost_per_million,
            output_cost_per_million,
            args,
            kwargs,
        )

    def get_model(self, name):
        model_info = self.models.get(name.lower())
        if model_info is None:
            raise ValueError(f"Model '{name}' not found")
        (
            model_class,
            tokenizer,
            input_cost_per_million,
            output_cost_per_million,
            args,
            kwargs,
        ) = model_info
        return model_class(
            *args,
            tokenizer=tokenizer,
            input_cost_per_million=input_cost_per_million,
            output_cost_per_million=output_cost_per_million,
            **kwargs,
        )

    def get_registered_models(self):
        return list(self.models.keys())
