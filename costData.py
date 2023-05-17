class CostData:
    def __init__(self):
        self.finetune_per_token = finetune_per_token
        self.tuned_per_token = tuned_per_token
        self.api_per_token = api_per_token


finetune_per_token = {
    'Davinci': 0.03,
    'Ada': 0.0004,
    'Babbage': 0.0006,
    'Curie': 0.003
}

tuned_per_token ={
    'Davinci': 0.12,
    'Ada': 0.0061,
    'Babbage': 0.0024,
    'Curie': 0.012
}

api_per_token = {
    'GPT 3.5': 0.002,
    'Ada': 0.0004,
    'Babbage': 0.0005,
    'Curie': 0.002,
    'Davinci': 0.02
}

