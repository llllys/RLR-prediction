from models.RLRPredictor import RLRPredictor

model_dict = {
    'rlr_predictor': RLRPredictor
}


def build_model(args):
    model = model_dict[args.model]
    return model(args)