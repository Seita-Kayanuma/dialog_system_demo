from .base import construct_turn_detector


def train(device, const_args, train_args=None):
    args, kwargs = const_args
    td = construct_turn_detector(*args, **kwargs)
    td.to(device)
    td.load()
    args, kwargs = [], {}
    if train_args is not None:
        args, kwargs = train_args
    td.train_turn_detector(*args, **kwargs)
    td.save(upload=True)
    td.upload_csv_log()
    
