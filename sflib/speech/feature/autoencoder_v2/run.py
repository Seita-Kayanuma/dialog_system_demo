from speech.feature.autoencoder_v2 import base as b
from ext.torch.callbacks.early_stopper import EarlyStopper


# --- 2019.11.30 ---
# gold1
def trainSIAE23T13V00():
    ae = b.construct_autoencoder(23, 13).to('cuda')
    #ae_tmp = b.construct_autoencoder(12, 13)
    #filename = ae_tmp.get_model_filename(version=2)
    #del ae_tmp
    #ae.force_load_model_file(filename)
    #print(ae.get_model_filename(version=2))
    #print(aaaaa)
    ae.train_autoencoder()
    ae.save(version=0, overwrite=True, upload=True)
    ae.upload_csv_log()
    

# --- 2019.11.29 ---
# gold1
def trainSIAE12T13V02():
    ae = b.construct_autoencoder(12, 13).to('cuda')
    ae.load(version=1)
    ae.train_autoencoder(epoch=50,
                         additional_callbacks=[EarlyStopper(patience=10000)])
    ae.save(version=2, overwrite=True, upload=True)
    ae.upload_csv_log()


# gold1
def trainSIAE18T13V02():
    ae = b.construct_autoencoder(18, 13).to('cuda')
    ae.load(version=1)
    ae.train_autoencoder(epoch=50,
                         additional_callbacks=[EarlyStopper(patience=10000)])
    ae.save(version=2, overwrite=True, upload=True)
    ae.upload_csv_log()

    
# silver5
def trainSIAE20T11V00():
    ae = b.construct_autoencoder(20, 11).to('cuda:0')
    ae.train_autoencoder(epoch=50,
                         additional_callbacks=[EarlyStopper(patience=10000)])
    ae.save(version=0, overwrite=True, upload=True)
    ae.upload_csv_log()


# silver5
def trainSIAE20T13V00():
    ae = b.construct_autoencoder(20, 13).to('cuda:1')
    ae.train_autoencoder(epoch=50,
                         additional_callbacks=[EarlyStopper(patience=10000)])
    ae.save(version=0, overwrite=True, upload=True)
    ae.upload_csv_log()

    
# silver5
def trainSIAE21T11V00():
    ae = b.construct_autoencoder(21, 11).to('cuda:2')
    ae.train_autoencoder(epoch=50,
                         additional_callbacks=[EarlyStopper(patience=10000)])
    ae.save(version=0, overwrite=True, upload=True)
    ae.upload_csv_log()

    
# silver5
def trainSIAE21T13V00():
    ae = b.construct_autoencoder(21, 13).to('cuda:3')
    ae.train_autoencoder(epoch=50,
                         additional_callbacks=[EarlyStopper(patience=10000)])
    ae.save(version=0, overwrite=True, upload=True)
    ae.upload_csv_log()

    
# --- 2019.11.28 ---
# gold1
def trainSIAE12T13V01():
    ae = b.construct_autoencoder(12, 13).to('cuda')
    ae.load(version=0)
    ae.train_autoencoder(epoch=50,
                         additional_callbacks=[EarlyStopper(patience=10000)])
    ae.save(version=1, overwrite=True, upload=True)
    ae.upload_csv_log()


# gold1
def trainSIAE18T13V01():
    ae = b.construct_autoencoder(18, 13).to('cuda')
    ae.load(version=0)
    ae.train_autoencoder(epoch=50,
                         additional_callbacks=[EarlyStopper(patience=10000)])
    ae.save(version=1, overwrite=True, upload=True)
    ae.upload_csv_log()


# --- 2019.11.26 ---
# gold1
def trainSIAE12T11V00():
    ae = b.construct_autoencoder(12, 11).to('cuda')
    ae.train_autoencoder()
    ae.save(version=0, overwrite=True, upload=True)
    ae.upload_csv_log()


# gold1
def trainSIAE18T11V00():
    ae = b.construct_autoencoder(18, 11).to('cuda')
    ae.train_autoencoder()
    ae.save(version=0, overwrite=True, upload=True)
    ae.upload_csv_log()


# silver5
def trainSIAE12T12V00():
    ae = b.construct_autoencoder(12, 12).to('cuda:2')
    ae.train_autoencoder()
    ae.save(version=0, overwrite=True, upload=True)
    ae.upload_csv_log()

    
# gold1
def trainSIAE12T13V00():
    ae = b.construct_autoencoder(12, 13).to('cuda')
    ae.train_autoencoder()
    ae.save(version=0, overwrite=True, upload=True)
    ae.upload_csv_log()


# gold1
def trainSIAE18T13V00():
    ae = b.construct_autoencoder(18, 13).to('cuda')
    ae.train_autoencoder()
    ae.save(version=0, overwrite=True, upload=True)
    ae.upload_csv_log()

# 2019.11.25
def trainSIAE12T06V01():
    early_stopper = EarlyStopper(patience=10000, verbose=True)
    for tn in (6, 8, 9, 10):
        ae = b.construct_autoencoder(12, tn).to('cuda')
        ae.load()
        ae.train_autoencoder(epoch=20, additional_callbacks=[early_stopper])
        ae.save(overwrite=False, upload=True)
        ae.upload_csv_log()


# BEFORE 2019.11.25
def train0006():
    for an in (12, 18):
        ae = b.construct_autoencoder(an, 6).to('cuda')
        ae.train_autoencoder()
        ae.save(overwrite=False, upload=True)


def train_other():
    for tn in (8, 9, 10):
        for an in (12, 18):
            ae = b.construct_autoencoder(an, tn).to('cuda')
            ae.train_autoencoder()
            ae.save(overwrite=False, upload=True)


if __name__ == "__main__":
    trainSIAE23T13V00()
