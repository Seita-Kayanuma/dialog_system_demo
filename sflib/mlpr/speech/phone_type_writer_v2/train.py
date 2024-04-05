from .base import construct_phone_type_writer


def train(device, const_args):
    args, kwargs = const_args
    ptw = construct_phone_type_writer(*args, **kwargs)
    ptw.to(device)
    ptw.train_phone_type_writer()
    ptw.save(upload=True)
    ptw.upload_csv_log()
